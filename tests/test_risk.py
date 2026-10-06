"""
Checks of the transfer levels and the route ranking on made-up legs.

uv run --no-sync pytest tests/test_risk.py
"""

from datetime import date, datetime, time

import polars as pl
import pytest

from priorbahn import risk


def at(hour: int, minute: int = 0) -> datetime:
    return datetime.combine(date(2026, 9, 20), time(hour, minute))


# request 1 has two routes:
# - route 0: arrives at B 14:00, connection leaves B 14:10, then arrives at C 15:00
# - route 1: one leg, arrives at C 15:05
LEGS = pl.DataFrame(
    [
        (1, 0, 0, "r1", "A", 1, at(13), "B", 5, at(14)),
        (1, 0, 1, "r2", "B", 3, at(14, 10), "C", 9, at(15)),
        (1, 1, 0, "r3", "A", 1, at(13, 5), "C", 7, at(15, 5)),
    ],
    schema=[
        "request_id",
        "route",
        "leg",
        "run_id",
        "from_station",
        "from_stop_num",
        "dep",
        "to_station",
        "to_stop_num",
        "arr",
    ],
    orient="row",
)


def legs_with(q50: float, q80: float, q95: float) -> pl.DataFrame:
    """
    The legs, with the given arrival delays at B and small ones elsewhere.
    """
    pred = pl.DataFrame(
        {
            "request_id": [1, 1, 1],
            "run_id": ["r1", "r2", "r3"],
            "stop_num": [5, 9, 7],
            "q50": [float(q50), 1.0, 1.0],
            "q80": [float(q80), 2.0, 2.0],
            "q95": [float(q95), 4.0, 4.0],
        }
    )
    return risk.with_arrival_delays(LEGS, pred)


@pytest.mark.parametrize(
    ("q50", "q80", "q95", "level"),
    [
        # 10 min planned, 2 to change trains: 8 min of room
        (1, 3, 5, 1),  # 8 is 3 min above q95: over 99%
        (1, 4, 8, 2),  # 8 is q95: 95%
        (1, 12, 15, 3),  # 8 is between q50 and q80: about 69%
        (9, 12, 15, 4),  # 8 is below q50
    ],
)
def test_transfer_level_depends_on_the_room_at_each_quantile(
    q50: int, q80: int, q95: int, level: int
) -> None:
    assert risk.transfers(legs_with(q50, q80, q95))["level"].to_list() == [level]


def test_chance_meets_the_quantiles_and_rises() -> None:
    df = pl.DataFrame({"q50": [2.0], "q80": [5.0], "q95": [11.0]})
    x = [-30.0, 0.0, 2.0, 3.5, 5.0, 8.0, 11.0, 20.0, 60.0]
    out = df.join(pl.DataFrame({"x": x}), how="cross").select(
        risk.chance(pl.col("x")).alias("chance")
    )["chance"]
    assert out.gather([2, 4, 6]).to_list() == pytest.approx([0.5, 0.8, 0.95])
    assert out.is_sorted() and out.min() > 0 and out.max() < 1


def test_a_late_connecting_train_gives_more_time() -> None:
    legs = legs_with(1, 8, 12)
    on_time = risk.transfers(legs)
    dep = pl.DataFrame(
        {"request_id": [1], "run_id": ["r2"], "stop_num": [3], "q50": [4.0]}
    )
    late = risk.transfers(risk.with_departure_delays(legs, dep))
    assert (on_time["wait_min"][0], late["wait_min"][0]) == (8, 12)
    assert late["chance"][0] == pytest.approx(0.95)


def test_levels_cut_the_chance() -> None:
    chance = pl.Series("c", [0.99, 0.97, 0.9, 0.8, 0.79, 0.5, 0.3, None])
    out = pl.DataFrame(chance).select(risk.level(pl.col("c")).alias("level"))
    assert out["level"].to_list() == [1, 1, 2, 2, 3, 3, 4, None]


def test_routes_rank_by_arrival_then_transfers() -> None:
    # route 0 is planned to arrive earlier (15:00 against 15:05), so it comes first despite
    # its transfer; the transfer level does not change the order
    for q in [(9, 12, 15), (1, 4, 8)]:
        ranked = risk.routes(legs_with(*q))
        assert ranked.select("route", "transfers", "rank").rows() == [
            (0, 1, 1),
            (1, 0, 2),
        ]


def test_routes_arriving_at_the_same_time_rank_by_transfers() -> None:
    # route 1 (direct) is planned to arrive at 15:00 like route 0: fewer transfers win
    legs = legs_with(1, 2, 4).with_columns(
        pl.when(pl.col("route") == 1).then(at(15)).otherwise(pl.col("arr")).alias("arr")
    )
    ranked = risk.routes(legs)
    assert ranked.select("route", "rank").rows() == [(1, 1), (0, 2)]


def test_routes_with_the_same_transfers_rank_by_planned_arrival() -> None:
    # two direct routes: the one planned to arrive first comes first, even though its
    # predicted delay makes it 10 min later at the 80% level
    legs = pl.DataFrame(
        [
            (1, 0, 0, "a", "A", 1, at(13), "C", 4, at(15)),
            (1, 1, 0, "b", "A", 1, at(13, 5), "C", 4, at(15, 5)),
        ],
        schema=LEGS.columns,
        orient="row",
    )
    pred = pl.DataFrame(
        {
            "request_id": [1, 1],
            "run_id": ["a", "b"],
            "stop_num": [4, 4],
            "q50": [10.0, 0.0],
            "q80": [15.0, 0.0],
            "q95": [20.0, 1.0],
        }
    )
    ranked = risk.routes(risk.with_arrival_delays(legs, pred))
    assert ranked.select("route", "rank").rows() == [(0, 1), (1, 2)]


def test_held_needs_the_change_time_and_no_cancellation() -> None:
    df = pl.DataFrame(
        {
            "arr": [at(14)] * 4,
            "arr_delay": [7, 9, 0, None],
            "arr_canceled": [False, False, True, False],
            "dep": [at(14, 10)] * 4,
            "dep_delay": [0, 0, 0, 0],
            "dep_canceled": [False] * 4,
        }
    )
    out = df.select(risk.held(*(pl.col(c) for c in df.columns)).alias("held"))[
        "held"
    ].to_list()
    # 3 min left, 1 min left, canceled, unknown delay
    assert out == [True, False, False, None]
