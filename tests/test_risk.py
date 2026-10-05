"""Checks of the careful transfer rule and the route ranking on made-up legs.

uv run --no-sync pytest tests/test_risk.py
"""

from datetime import date, datetime, time

import polars as pl

from dbdelay import risk


def at(hour: int, minute: int = 0) -> datetime:
    return datetime.combine(date(2026, 9, 20), time(hour, minute))


# request 1 has two routes:
# - route 0: arrives at B 14:00, connection leaves B 14:10, then arrives at C 15:00
# - route 1: one leg, arrives at C 15:05
LEGS = pl.DataFrame(
    [
        (1, 0, 0, "r1", "A", at(13), "B", 5, at(14)),
        (1, 0, 1, "r2", "B", at(14, 10), "C", 9, at(15)),
        (1, 1, 0, "r3", "A", at(13, 5), "C", 7, at(15, 5)),
    ],
    schema=[
        "request_id",
        "route",
        "leg",
        "run_id",
        "from_station",
        "dep",
        "to_station",
        "to_stop_num",
        "arr",
    ],
    orient="row",
)


def legs_with(q95_at_b: float) -> pl.DataFrame:
    pred = pl.DataFrame(
        {
            "request_id": [1, 1, 1],
            "run_id": ["r1", "r2", "r3"],
            "stop_num": [5, 9, 7],
            "q50": [1.0, 1.0, 1.0],
            "q80": [2.0, 2.0, 2.0],
            "q95": [q95_at_b, 4.0, 4.0],
        }
    )
    return risk.with_arrival_delays(LEGS, pred)


def test_transfer_is_safe_with_5_min_left_at_q95():
    # 14:10 - (14:00 + 5) = 5 min: safe. With q95 = 6 only 4 min are left: at risk.
    assert risk.transfers(legs_with(5.0))["safe"].to_list() == [True]
    assert risk.transfers(legs_with(6.0))["safe"].to_list() == [False]


def test_routes_rank_transfers_at_risk_last():
    # route 0 arrives earlier at q80 (15:02 against 15:07), but its transfer is at risk
    ranked = risk.routes(legs_with(6.0))
    assert ranked.select("route", "at_risk", "rank").rows() == [(1, 0, 1), (0, 1, 2)]
    ranked = risk.routes(legs_with(5.0))
    assert ranked.select("route", "at_risk", "rank").rows() == [(0, 0, 1), (1, 0, 2)]
