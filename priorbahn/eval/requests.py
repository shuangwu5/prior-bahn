"""
Sample evaluation requests and route them (docs/plan.md, component 5).

A request is (start, destination, departure time) on one day of the validation or test
week. Start and destination are drawn from the busiest stations, weighted by their number
of departures, and departure times are spread evenly over the hours. Each request is routed
like in the app, and the legs of its routes are saved so every model is scored on the same
rows.

Run from the repo root: uv run --no-sync python -m priorbahn.eval.requests validation
"""

import sys
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from priorbahn.eval import paths
from priorbahn.eval.runs import train_group
from priorbahn.model.context import actual_time
from priorbahn.router import core as rc

STOPS = "data/processed/stops.parquet"
WEEKS = {
    "validation": date(2026, 9, 17),
    "test": date(2026, 9, 24),
}
N_REQUESTS = 200
N_STATIONS = 300  # busiest stations to draw start and destination from
HOURS = range(6, 23)  # departure hours, each gets about the same number of requests
N_ROUTES = 3  # alternatives per request, as in the app


def busy_stations(n: int = N_STATIONS) -> pl.DataFrame:
    """The `n` stations with the most departures in the context pool, with their counts."""
    return (
        pl.scan_parquet(STOPS)
        .filter(pl.col("split") == "context", pl.col("planned_dep").is_not_null())
        .group_by("station")
        .len("departures")
        .sort("departures", descending=True)
        .head(n)
        .collect()
    )


def draw_requests(split: str, n: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    stations = busy_stations()
    p = stations["departures"].to_numpy() / stations["departures"].sum()
    names = stations["station"].to_numpy()

    src = rng.choice(names, size=n, p=p)
    dst = rng.choice(names, size=n, p=p)
    # redraw destinations equal to the start
    while (same := src == dst).any():
        dst[same] = rng.choice(names, size=same.sum(), p=p)

    days = [WEEKS[split] + timedelta(days=int(d)) for d in rng.integers(0, 7, n)]
    hours = np.resize(np.array(HOURS), n)  # stratified: hours repeat in turn
    rng.shuffle(hours)
    minutes = rng.integers(0, 60, n)
    return pl.DataFrame(
        {
            "request_id": range(n),
            "day": days,
            "src": src,
            "dst": dst,
            "depart": [
                datetime.combine(d, datetime.min.time())
                + timedelta(hours=int(h), minutes=int(m))
                for d, h, m in zip(days, hours, minutes)
            ],
        }
    )


def find_journeys(
    tt: rc.Timetable, src: str, dst: str, depart: datetime, n: int = N_ROUTES
) -> list[list[rc.Leg]]:
    """Up to `n` routes, found by rerunning for later departures (as the app)."""
    journeys = []
    for _ in range(n):
        legs = rc.earliest_arrival(tt, src, dst, depart)
        if legs is None:
            break
        journey = rc.replay(tt, legs)
        journeys.append(journey)
        depart = journey[0].dep + timedelta(minutes=1)
    return journeys


def legs_frame(journeys: list[list[rc.Leg]], request_id: int) -> list[dict]:
    """One row per leg of each journey, with the outcome of the transfer before it."""
    rows = []
    for route, journey in enumerate(journeys):
        for i, leg in enumerate(journey):
            t = leg.transfer_before
            rows.append(
                {
                    "request_id": request_id,
                    "route": route,
                    "leg": i,
                    "run_id": leg.run_id,
                    "from_station": leg.from_station,
                    "from_stop_num": leg.from_stop_num,
                    "dep": leg.dep,
                    "to_station": leg.to_station,
                    "to_stop_num": leg.to_stop_num,
                    "arr": leg.arr,
                    "transfer_status": None if t is None else t.status,
                }
            )
    return rows


def route_request(tt: rc.Timetable, req: dict) -> list[dict]:
    """Legs of up to N_ROUTES routes for one request."""
    journeys = find_journeys(tt, req["src"], req["dst"], req["depart"])
    return legs_frame(journeys, req["request_id"])


def build(split: str, n: int = N_REQUESTS, seed: int = 0) -> tuple[pl.DataFrame, ...]:
    requests = draw_requests(split, n, seed)
    rows = []
    for day, group in requests.sort("day").group_by("day", maintain_order=True):
        tt = rc.load_timetable(day[0])
        for req in group.iter_rows(named=True):
            rows.extend(route_request(tt, req))
    return requests, pl.DataFrame(rows)


def request_rows(
    stops: pl.LazyFrame, requests: pl.DataFrame, legs: pl.DataFrame
) -> pl.DataFrame:
    """
    The stop events scored for each request, one row per request, stop and event.

    Per leg, two events: the departure at the boarding stop (`model` "dep") and the
    arrival at the alighting stop ("arr"). An event shared by several routes of one
    request appears once. "Now" is the request's departure time. Added columns:
    - `seen_delay`: the delay of the run's last event before now (null: not started yet)
    - `minutes_ahead`: planned time of the scored event minus now
    """
    keys = pl.concat(
        [
            legs.select(
                "request_id",
                "run_id",
                pl.col("from_stop_num").alias("stop_num"),
                pl.lit("dep").alias("model"),
            ),
            legs.select(
                "request_id",
                "run_id",
                pl.col("to_stop_num").alias("stop_num"),
                pl.lit("arr").alias("model"),
            ),
        ]
    ).unique()
    now = requests.select("request_id", pl.col("depart").alias("now"))
    runs = stops.filter(pl.col("run_id").is_in(keys["run_id"].unique().implode()))
    rows = (
        keys.join(now, on="request_id")
        .join(runs.collect(), on=["run_id", "stop_num"])
        .with_columns(
            minutes_ahead=(
                pl.when(pl.col("model") == "dep")
                .then(pl.col("planned_dep"))
                .otherwise(pl.col("planned_arr"))
                - pl.col("now")
            ).dt.total_minutes()
        )
    )
    return (
        rows.join(last_seen(runs, rows), on=["request_id", "run_id"], how="left")
        .with_columns(train_group(), weight=pl.lit(1.0))
        .sort("request_id", "run_id", "stop_num", "model")
    )


def last_seen(runs: pl.LazyFrame, rows: pl.DataFrame) -> pl.DataFrame:
    """
    Delay of each request's runs at their last event before now (same leak rule as the
    context: the actual time must be before now).
    """
    pairs = rows.select("request_id", "run_id", "now").unique()
    events = runs.collect().join(pairs, on="run_id")
    return (
        pl.concat(
            [
                events.select(
                    "request_id",
                    "run_id",
                    "now",
                    actual_time(e).alias("time"),
                    pl.col(f"{e}_delay").alias("seen_delay"),
                )
                for e in ("arr", "dep")
            ]
        )
        .filter(pl.col("time") < pl.col("now"))
        .sort("time")
        .group_by("request_id", "run_id")
        .agg(pl.last("seen_delay"))
    )


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    requests, legs = build(split)
    paths.router_requests(split).parent.mkdir(parents=True, exist_ok=True)
    requests.write_parquet(paths.router_requests(split))
    legs.write_parquet(paths.router_legs(split))
    rows = request_rows(pl.scan_parquet(STOPS), requests, legs)
    rows.write_parquet(paths.request_rows(split))

    routed = legs["request_id"].n_unique()
    routes = legs.select("request_id", "route").n_unique()
    print(f"{split}: {len(requests)} requests, {routed} routed, {routes} routes")
    print(f"{len(legs)} legs, {legs['run_id'].n_unique()} runs")
    print(legs.group_by("request_id", "route").len().get_column("len").describe())
    seen = rows["seen_delay"].is_not_null().sum()
    print(f"{len(rows)} scored events, {seen} on a train already running at now")


if __name__ == "__main__":
    main()
