"""Train-only routing on the stops table, replayed against what actually happened."""

import bisect
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl

PATH = "data/processed/stops.parquet"
MIN_TRANSFER_MIN = 5
INF = 10**12


@dataclass
class Timetable:
    day_start: datetime
    conns: list[dict]
    deps: list[datetime]
    run_conns: dict[str, list[int]]
    stations: list[str]  # names, a station with several EVA codes is one stop


@dataclass
class Transfer:
    station: str
    planned_min: int | None
    actual_min: int | None
    status: str  # "held", "MISSED" or "?"


@dataclass
class Leg:
    label: str
    run_id: str
    from_station: str
    from_stop_num: int
    dep: datetime
    dep_delay: int | None
    to_station: str
    to_stop_num: int
    arr: datetime
    arr_delay: int | None
    stops: list[str]
    transfer_before: Transfer | None


def actual(planned: str, delay: str) -> pl.Expr:
    """Actual time from the planned time and the delay in minutes (null if canceled)."""
    return pl.col(planned) + pl.duration(minutes=pl.col(delay))


def load_timetable(day: date, path: str = PATH) -> Timetable:
    """Connections between consecutive stops of each run, from the stops table.

    Stations are the merged station names of the stops table, so every leg of a route
    matches rows the delay model can predict.
    """
    day_start = datetime.combine(day, datetime.min.time())
    stops = (
        pl.scan_parquet(path)
        # runs of the day before can still be running, and routes of a late request
        # continue after midnight on runs of the next day
        .filter(
            pl.col("run_day").is_between(
                day - timedelta(days=1), day + timedelta(days=1)
            )
        )
        .filter(
            pl.coalesce("planned_dep", "planned_arr").is_between(
                day_start - timedelta(hours=12), day_start + timedelta(hours=36)
            )
        )
        .sort("run_id", "stop_num")
        .with_columns(
            *(
                pl.col(c).shift(-1).over("run_id").alias(f"next_{c}")
                for c in [
                    "station",
                    "stop_num",
                    "planned_arr",
                    "arr_delay",
                    "arr_canceled",
                ]
            )
        )
    )
    conns = (
        stops.filter(
            pl.col("planned_dep").is_not_null()
            & pl.col("next_planned_arr").is_not_null()
            & (pl.col("next_planned_arr") >= pl.col("planned_dep"))
            & (pl.col("planned_dep") >= day_start)
        )
        .sort("planned_dep")
        .select(
            "run_id",
            "train_key",
            "line_number",
            "station",
            "stop_num",
            "next_station",
            "next_stop_num",
            pl.col("planned_dep").alias("dep"),
            actual("planned_dep", "dep_delay").alias("dep_actual"),
            pl.col("dep_canceled"),
            pl.col("next_planned_arr").alias("arr"),
            actual("next_planned_arr", "next_arr_delay").alias("arr_actual"),
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
        stations=sorted(set(conns["station"]) | set(conns["next_station"])),
    )


def stops_between(tt: Timetable, enter: int, leave: int) -> list[str]:
    C = tt.conns
    idx = tt.run_conns[C[enter]["run_id"]]
    seg = idx[idx.index(enter) : idx.index(leave) + 1]
    return [C[i]["station"] for i in seg] + [C[leave]["next_station"]]


def earliest_arrival(
    tt: Timetable, src: str, dst: str, depart_at: datetime
) -> list[tuple[int, int]] | None:
    """Connection scan: returns legs as (boarding connection, alighting connection) index pairs.

    Stops are station names, as in prep.py: big stations have several EVA codes (main
    line and S-Bahn), and a journey may start, end or change trains at any of them.
    """
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
        run, u, v = c["run_id"], c["station"], c["next_station"]
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
        stop = C[enter]["station"]
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
        label = a["train_key"] + (f" ({a['line_number']})" if a["line_number"] else "")
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
                station=a["station"],
                planned_min=delay(prev["arr"], a["dep"]),
                actual_min=delay(prev["arr_actual"], a["dep_actual"])
                if prev["arr_actual"] is not None
                else None,
                status=status,
            )
        out.append(
            Leg(
                label=label,
                run_id=a["run_id"],
                from_station=a["station"],
                from_stop_num=a["stop_num"],
                dep=a["dep"],
                dep_delay=delay(a["dep"], a["dep_actual"]),
                to_station=b["next_station"],
                to_stop_num=b["next_stop_num"],
                arr=b["arr"],
                arr_delay=delay(b["arr"], b["arr_actual"]),
                stops=stops_between(tt, enter, leave),
                transfer_before=transfer,
            )
        )
        prev = b
    return out
