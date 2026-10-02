"""Prototype: train-only routing on the dataset's own timetable, replayed against what actually happened.

Usage: offline_router.py [FROM_STATION] [TO_STATION] [YYYY-MM-DD] [HH:MM]
"""

import sys
import time
from datetime import date, timedelta

import router_core as rc

src_name = sys.argv[1] if len(sys.argv) > 1 else "Heidelberg Hbf"
dst_name = sys.argv[2] if len(sys.argv) > 2 else "Lübeck Hbf"
day = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date(2026, 9, 24)
hhmm = sys.argv[4] if len(sys.argv) > 4 else "08:00"

print("== non-train rows dropped ==")
print(rc.dropped_non_train())

t0 = time.time()
tt = rc.load_timetable(day)
print(
    f"\n== timetable for {day}: {len(tt.conns):,} connections, {len(tt.run_conns):,} runs, built in {time.time() - t0:.1f}s =="
)

src, dst = tt.station_eva[src_name], tt.station_eva[dst_name]
hours, minutes = map(int, hhmm.split(":"))
depart = tt.day_start + timedelta(hours=hours, minutes=minutes)
for _ in range(3):
    t0 = time.time()
    legs = rc.earliest_arrival(tt, src, dst, depart)
    if legs is None:
        print("no route found")
        break
    journey = rc.replay(tt, legs)
    print(
        f"\n== {src_name} {journey[0].dep:%H:%M} -> {dst_name} {journey[-1].arr:%H:%M}, "
        f"{len(journey) - 1} transfers (query {time.time() - t0:.2f}s) =="
    )
    for leg in journey:
        t = leg.transfer_before
        if t is not None:
            print(
                f"     transfer at {t.station}: planned {t.planned_min} min, actually {t.actual_min} min -> {t.status}"
            )
        print(
            f"  {leg.label:<22} {leg.from_station} {leg.dep:%H:%M} (dep delay {leg.dep_delay}) -> "
            f"{leg.to_station} {leg.arr:%H:%M} (arr delay {leg.arr_delay})"
        )
        print(f"     stops: {' > '.join(leg.stops)}")
    depart = journey[0].dep + timedelta(minutes=1)
