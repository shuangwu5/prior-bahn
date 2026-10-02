"""Prototype: train-only routing on the dataset's own timetable, replayed against what actually happened.

Usage: offline_router.py [FROM_STATION] [TO_STATION] [YYYY-MM-DD] [HH:MM]
"""

import bisect
import sys
import time
from datetime import date, datetime, timedelta

import polars as pl

PATH = "data/monthly_processed_data/data-2026-09.parquet"
NON_TRAIN = r"(?i)^(bus|sev|bsv)$"
MIN_TRANSFER_MIN = 5
INF = 10**12

src_name = sys.argv[1] if len(sys.argv) > 1 else "Heidelberg Hbf"
dst_name = sys.argv[2] if len(sys.argv) > 2 else "Lübeck Hbf"
day = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date(2026, 9, 24)
hhmm = sys.argv[4] if len(sys.argv) > 4 else "08:00"

lf = pl.scan_parquet(PATH)
print("== non-train rows dropped ==")
print(
    lf.filter(pl.col("train_type").str.contains(NON_TRAIN))
    .group_by("train_type")
    .len()
    .collect()
)

t0 = time.time()
day_start = datetime.combine(day, datetime.min.time())
stops = (
    lf.filter(~pl.col("train_type").str.contains(NON_TRAIN))
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
        pl.col("arrival_change_time").shift(-1).over("run_id").alias("next_arr_actual"),
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
print(
    f"\n== timetable for {day}: {conns.height:,} connections, {conns['run_id'].n_unique():,} runs, built in {time.time() - t0:.1f}s =="
)

names = dict(zip(conns["station_name"].to_list(), conns["eva"].to_list()))
src, dst = names[src_name], names[dst_name]
C = conns.to_dicts()
deps = [c["dep"] for c in C]
run_conns: dict[str, list[int]] = {}
for i, c in enumerate(C):
    run_conns.setdefault(c["run_id"], []).append(i)


def stops_between(enter: int, leave: int) -> list[str]:
    idx = run_conns[C[enter]["run_id"]]
    seg = idx[idx.index(enter) : idx.index(leave) + 1]
    return [C[i]["station_name"] for i in seg] + [C[leave]["next_station"]]


def earliest_arrival(depart_at: datetime) -> list[tuple[int, int]] | None:
    """Connection scan: returns legs as (boarding connection, alighting connection) index pairs."""
    best = {src: depart_at}
    via: dict[str, tuple[int, int]] = {}
    boarded: dict[str, int] = {}
    run_boardings: dict[str, int] = {}
    boardings = {src: 0}  # boardings used by the best label at each stop
    far = day_start + timedelta(days=365)
    for i in range(bisect.bisect_left(deps, depart_at), len(C)):
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


hours, minutes = map(int, hhmm.split(":"))
depart = day_start + timedelta(hours=hours, minutes=minutes)
for _ in range(3):
    t0 = time.time()
    legs = earliest_arrival(depart)
    if legs is None:
        print("no route found")
        break
    first, last = C[legs[0][0]], C[legs[-1][1]]
    print(
        f"\n== {src_name} {first['dep']:%H:%M} -> {dst_name} {last['arr']:%H:%M}, "
        f"{len(legs) - 1} transfers (query {time.time() - t0:.2f}s) =="
    )
    prev = None
    for enter, leave in legs:
        a, b = C[enter], C[leave]
        label = f"{a['train_type']} {a['train_number']}" + (
            f" ({a['line_number']})" if a["line_number"] else ""
        )
        if prev is not None:
            planned_gap = delay(prev["arr"], a["dep"])
            ok = "?"
            if prev["arr_actual"] is not None and a["dep_actual"] is not None:
                ok = (
                    "held"
                    if a["dep_actual"] >= prev["arr_actual"]
                    and not a["dep_canceled"]
                    and not prev["arr_canceled"]
                    else "MISSED"
                )
            print(
                f"     transfer at {a['station_name']}: planned {planned_gap} min, actually {delay(prev['arr_actual'], a['dep_actual'])} min -> {ok}"
            )
        print(
            f"  {label:<22} {a['station_name']} {a['dep']:%H:%M} (dep delay {delay(a['dep'], a['dep_actual'])}) -> "
            f"{b['next_station']} {b['arr']:%H:%M} (arr delay {delay(b['arr'], b['arr_actual'])})"
        )
        print(f"     stops: {' > '.join(stops_between(enter, leave))}")
        prev = b
    depart = first["dep"] + timedelta(minutes=1)
