"""Router checks: a small made-up timetable, plus one real query if the data is there.

uv run --no-sync pytest tests/test_router.py
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from dbdelay.router import core as rc

RAW = Path(__file__).resolve().parents[1] / rc.PATH
DAY = datetime.combine(date(2026, 9, 24), datetime.min.time())  # naive, like the data


def conn(
    run: str, frm: str, to: str, dep: str, arr: str, eva: str = "", next_eva: str = ""
) -> dict:
    dep_t = DAY + timedelta(hours=int(dep[:2]), minutes=int(dep[3:]))
    arr_t = DAY + timedelta(hours=int(arr[:2]), minutes=int(arr[3:]))
    return {
        "run_id": run,
        "train_type": "RE",
        "train_number": run,
        "line_number": None,
        "eva": eva or frm,
        "station_name": frm,
        "next_eva": next_eva or to,
        "next_station": to,
        "dep": dep_t,
        "dep_actual": dep_t,
        "dep_canceled": False,
        "arr": arr_t,
        "arr_actual": arr_t,
        "arr_canceled": False,
    }


def timetable(conns: list[dict]) -> rc.Timetable:
    conns = sorted(conns, key=lambda c: c["dep"])
    run_conns: dict[str, list[int]] = {}
    for i, c in enumerate(conns):
        run_conns.setdefault(c["run_id"], []).append(i)
    return rc.Timetable(
        day_start=DAY,
        conns=conns,
        deps=[c["dep"] for c in conns],
        run_conns=run_conns,
        stations=sorted({c["station_name"] for c in conns}),
    )


def test_station_with_several_eva_codes_is_one_stop():
    # "Hub" stands for a station like Hamburg Hbf, whose main-line and S-Bahn platforms
    # have different EVA codes. The router only sees names, so a change at Hub works
    # whichever platform each train uses, and a detour back to Hub is never needed.
    tt = timetable(
        [
            conn("1", "A", "Hub", "08:00", "08:30", next_eva="Hub main line"),
            conn("2", "Hub", "B", "08:40", "09:00", eva="Hub S-Bahn"),
            conn("3", "Hub", "Side", "08:35", "08:40", eva="Hub main line"),
            conn("4", "Side", "Hub", "08:45", "08:50", next_eva="Hub S-Bahn"),
        ]
    )
    legs = rc.earliest_arrival(tt, "A", "B", DAY + timedelta(hours=8))
    assert [tt.conns[e]["run_id"] for e, _ in legs] == ["1", "2"]
    legs = rc.earliest_arrival(tt, "A", "Hub", DAY + timedelta(hours=8))
    assert [tt.conns[e]["run_id"] for e, _ in legs] == ["1"]


def test_hamburg_to_munich_has_no_detour():
    if not RAW.exists():
        pytest.skip(f"{rc.PATH} is not there")
    tt = rc.load_timetable(DAY.date(), str(RAW))
    legs = rc.earliest_arrival(
        tt, "Hamburg Hbf", "München Hbf", tt.day_start + timedelta(hours=8)
    )
    journey = rc.replay(tt, legs)
    # with one EVA code per name, this was 5 legs: an S-Bahn to Dammtor, and a detour
    # via Pasing to reach the S-Bahn platforms of München Hbf at 15:19
    assert len(journey) == 1
    assert journey[-1].arr == tt.day_start + timedelta(hours=13, minutes=50)
