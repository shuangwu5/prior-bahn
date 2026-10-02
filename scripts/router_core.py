"""Train-only routing on the dataset's own timetable, replayed against what actually happened."""

import bisect
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl

PATH = "data/monthly_processed_data/data-2026-09.parquet"
NON_TRAIN = r"(?i)^(bus|sev|bsv)$"
MIN_TRANSFER_MIN = 5
INF = 10**12


@dataclass
class Timetable:
    day_start: datetime
    conns: list[dict]
    deps: list[datetime]
    run_conns: dict[str, list[int]]
    station_eva: dict[str, str]


@dataclass
class Transfer:
    station: str
    planned_min: int | None
    actual_min: int | None
    status: str  # "held", "MISSED" or "?"


@dataclass
class Leg:
    label: str
    from_station: str
    dep: datetime
    dep_delay: int | None
    to_station: str
    arr: datetime
    arr_delay: int | None
    stops: list[str]
    transfer_before: Transfer | None


def dropped_non_train(path: str = PATH) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .filter(pl.col("train_type").str.contains(NON_TRAIN))
        .group_by("train_type")
        .len()
        .collect()
    )


def load_timetable(day: date, path: str = PATH) -> Timetable:
    day_start = datetime.combine(day, datetime.min.time())
    stops = (
        pl.scan_parquet(path)
        .filter(~pl.col("train_type").str.contains(NON_TRAIN))
        .with_columns(pl.col("id").str.extract(r"^(.*)-\d+$", 1).alias("run_id"))
        .filter(
            pl.coalesce("departure_planned_time", "arrival_planned_time").is_between(
                day_start - timedelta(hours=12), day_start + timedelta(hours=36)
            )
        )
        .sort("run_id", "train_line_station_num")
        .with_columns(
            pl.col("eva").shift(-1).over("run_id").alias("next_eva"),
            pl.col("station_name").shift(-1).over("run_id").alias("next_station"),
            pl.col("arrival_planned_time")
            .shift(-1)
            .over("run_id")
            .alias("next_arr_planned"),
            pl.col("arrival_change_time")
            .shift(-1)
            .over("run_id")
            .alias("next_arr_actual"),
            pl.col("arrival_is_canceled")
            .shift(-1)
            .over("run_id")
            .alias("next_arr_canceled"),
        )
    )
    conns = (
        stops.filter(
            pl.col("departure_planned_time").is_not_null()
            & pl.col("next_arr_planned").is_not_null()
            & (pl.col("next_arr_planned") >= pl.col("departure_planned_time"))
            & (pl.col("departure_planned_time") >= day_start)
        )
        .sort("departure_planned_time")
        .select(
            "run_id",
            "train_type",
            "train_number",
            "line_number",
            "eva",
            "station_name",
            "next_eva",
            "next_station",
            pl.col("departure_planned_time").alias("dep"),
            pl.col("departure_change_time").alias("dep_actual"),
            pl.col("departure_is_canceled").alias("dep_canceled"),
            pl.col("next_arr_planned").alias("arr"),
            pl.col("next_arr_actual").alias("arr_actual"),
            pl.col("next_arr_canceled").alias("arr_canceled"),
        )
        .collect()
    )
    C = conns.to_dicts()
    run_conns: dict[str, list[int]] = {}
    for i, c in enumerate(C):
        run_conns.setdefault(c["run_id"], []).append(i)
    return Timetable(
        day_start=day_start,
        conns=C,
        deps=[c["dep"] for c in C],
        run_conns=run_conns,
        station_eva=dict(zip(conns["station_name"], conns["eva"])),
    )


def stops_between(tt: Timetable, enter: int, leave: int) -> list[str]:
    C = tt.conns
    idx = tt.run_conns[C[enter]["run_id"]]
    seg = idx[idx.index(enter) : idx.index(leave) + 1]
    return [C[i]["station_name"] for i in seg] + [C[leave]["next_station"]]


def earliest_arrival(
    tt: Timetable, src: str, dst: str, depart_at: datetime
) -> list[tuple[int, int]] | None:
    """Connection scan: returns legs as (boarding connection, alighting connection) index pairs."""
    C = tt.conns
    best = {src: depart_at}
    via: dict[str, tuple[int, int]] = {}
    boarded: dict[str, int] = {}
    run_boardings: dict[str, int] = {}
    boardings = {src: 0}  # boardings used by the best label at each stop
    far = tt.day_start + timedelta(days=365)
    for i in range(bisect.bisect_left(tt.deps, depart_at), len(C)):
        c = C[i]
        if c["dep"] > best.get(dst, far):
            break
        run, u, v = c["run_id"], c["eva"], c["next_eva"]
        ready = best.get(u)
        if (
            ready is not None
            and ready + timedelta(minutes=0 if u == src else MIN_TRANSFER_MIN)
            <= c["dep"]
            and boardings[u] + 1 < run_boardings.get(run, INF)
        ):
            boarded[run] = i
            run_boardings[run] = boardings[u] + 1
        if run not in boarded:
            continue
        # earliest arrival first, fewer boardings (transfers) as the tie-break
        if (c["arr"], run_boardings[run]) < (best.get(v, far), boardings.get(v, INF)):
            best[v] = c["arr"]
            boardings[v] = run_boardings[run]
            via[v] = (boarded[run], i)
    if dst not in via:
        return None
    legs, stop = [], dst
    while stop != src:
        enter, leave = via[stop]
        legs.append((enter, leave))
        stop = C[enter]["eva"]
    return legs[::-1]


def delay(planned: datetime, actual: datetime | None) -> int | None:
    return None if actual is None else round((actual - planned).total_seconds() / 60)


def replay(tt: Timetable, legs: list[tuple[int, int]]) -> list[Leg]:
    """Compare the planned journey with what actually happened, transfer by transfer."""
    C = tt.conns
    out: list[Leg] = []
    prev = None
    for enter, leave in legs:
        a, b = C[enter], C[leave]
        label = f"{a['train_type']} {a['train_number']}" + (
            f" ({a['line_number']})" if a["line_number"] else ""
        )
        transfer = None
        if prev is not None:
            status = "?"
            if prev["arr_actual"] is not None and a["dep_actual"] is not None:
                status = (
                    "held"
                    if a["dep_actual"] >= prev["arr_actual"]
                    and not a["dep_canceled"]
                    and not prev["arr_canceled"]
                    else "MISSED"
                )
            transfer = Transfer(
                station=a["station_name"],
                planned_min=delay(prev["arr"], a["dep"]),
                actual_min=delay(prev["arr_actual"], a["dep_actual"])
                if prev["arr_actual"] is not None
                else None,
                status=status,
            )
        out.append(
            Leg(
                label=label,
                from_station=a["station_name"],
                dep=a["dep"],
                dep_delay=delay(a["dep"], a["dep_actual"]),
                to_station=b["next_station"],
                arr=b["arr"],
                arr_delay=delay(b["arr"], b["arr_actual"]),
                stops=stops_between(tt, enter, leave),
                transfer_before=transfer,
            )
        )
        prev = b
    return out
