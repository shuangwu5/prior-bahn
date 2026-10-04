"""Sample evaluation requests and route them (docs/plan.md, component 5).

A request is (start, destination, departure time) on one day of the validation or test
week. Start and destination are drawn from the busiest stations, weighted by their number
of departures, and departure times are spread evenly over the hours. Each request is routed
like in the app, and the legs of its routes are saved so every model is scored on the same
rows.

Run from the repo root: uv run --no-sync python -m dbdelay.eval.requests validation
"""

import sys
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from dbdelay.eval import paths
from dbdelay.router import core as rc

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


def route_request(tt: rc.Timetable, req: dict) -> list[dict]:
    """Legs of up to N_ROUTES routes, found by rerunning for later departures (as the app)."""
    rows = []
    depart = req["depart"]
    for route in range(N_ROUTES):
        legs = rc.earliest_arrival(tt, req["src"], req["dst"], depart)
        if legs is None:
            break
        journey = rc.replay(tt, legs)
        for i, leg in enumerate(journey):
            t = leg.transfer_before
            rows.append(
                {
                    "request_id": req["request_id"],
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
        depart = journey[0].dep + timedelta(minutes=1)
    return rows


def build(split: str, n: int = N_REQUESTS, seed: int = 0) -> tuple[pl.DataFrame, ...]:
    requests = draw_requests(split, n, seed)
    rows = []
    for day, group in requests.sort("day").group_by("day", maintain_order=True):
        tt = rc.load_timetable(day[0])
        for req in group.iter_rows(named=True):
            rows.extend(route_request(tt, req))
    return requests, pl.DataFrame(rows)


def query_rows(stops: pl.LazyFrame, legs: pl.DataFrame) -> pl.DataFrame:
    """Stop rows the models predict for each request: the boarding stop of every leg
    (departure model) and the alighting stop (arrival model). A stop shared by several
    routes of one request appears once.
    """
    keys = pl.concat(
        [
            legs.select(
                "request_id", "run_id", pl.col("from_stop_num").alias("stop_num")
            ),
            legs.select(
                "request_id", "run_id", pl.col("to_stop_num").alias("stop_num")
            ),
        ]
    ).unique()
    rows = stops.join(keys.lazy(), on=["run_id", "stop_num"]).collect()
    return rows.sort("request_id", "run_id", "stop_num")


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    requests, legs = build(split)
    paths.router_requests(split).parent.mkdir(parents=True, exist_ok=True)
    requests.write_parquet(paths.router_requests(split))
    legs.write_parquet(paths.router_legs(split))

    routed = legs["request_id"].n_unique()
    routes = legs.select("request_id", "route").n_unique()
    print(f"{split}: {len(requests)} requests, {routed} routed, {routes} routes")
    print(f"{len(legs)} legs, {legs['run_id'].n_unique()} runs")
    print(legs.group_by("request_id", "route").len().get_column("len").describe())


if __name__ == "__main__":
    main()
