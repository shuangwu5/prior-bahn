"""Checks of the last known delay on small made-up runs.

uv run --no-sync pytest tests/test_last_known.py
"""

from datetime import date, datetime, time, timedelta

import polars as pl

from dbdelay.model.last_known import with_last_known

DAY = date(2026, 9, 20)
WEEK_AGO = DAY - timedelta(days=7)
NOW = datetime.combine(DAY, time(14))


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute))


def run(run_id: str, day: date, delays: list[tuple]) -> list[tuple]:
    """Four stops A to D with fixed planned times on `day` and the given
    (arr_delay, dep_delay) per stop."""
    planned = [
        (None, at(day, 13)),
        (at(day, 13, 40), at(day, 13, 50)),
        (at(day, 14, 30), at(day, 14, 32)),
        (at(day, 15), None),
    ]
    return [
        (run_id, i + 1, arr, dep, *delay)
        for i, ((arr, dep), delay) in enumerate(zip(planned, delays))
    ]


STOPS = pl.DataFrame(
    # today: left A 4 min late, arrived at B 6 min late, leaves B at 14:02 (after now)
    run("today", DAY, [(None, 4), (6, 12), (13, 13), (15, None)])
    # a week ago: arrived at B at 13:45, left B at 13:55
    + run("week ago", WEEK_AGO, [(None, 2), (5, 5), (3, 3), (2, None)])
    # today, a later train that has not started at now
    + [
        ("later", 1, None, at(DAY, 15), None, 0),
        ("later", 2, at(DAY, 16), None, 1, None),
    ],
    schema={
        "run_id": pl.String,
        "stop_num": pl.Int32,
        "planned_arr": pl.Datetime("ns"),
        "planned_dep": pl.Datetime("ns"),
        "arr_delay": pl.Int16,
        "dep_delay": pl.Int16,
    },
    orient="row",
)


def last_known(run_id: str, stop_num: int, model: str, replay: bool) -> tuple:
    rows = STOPS.filter(pl.col("run_id") == run_id, pl.col("stop_num") == stop_num)
    out = with_last_known(rows, STOPS.lazy(), NOW, model, replay)
    return out.select(
        "last_known_delay", "minutes_since_known", "stops_since_known"
    ).row(0)


def test_query_row_uses_only_events_before_now():
    # arrival at C: the departure from B (14:02) is after now, so B's arrival counts
    assert last_known("today", 3, "arr", replay=False) == (6, 50, 1)


def test_departure_uses_the_arrival_at_the_same_stop():
    assert last_known("today", 2, "dep", replay=False) == (6, 10, 0)


def test_train_not_started_has_no_last_known_delay():
    assert last_known("later", 2, "arr", replay=False) == (None, None, None)


def test_context_row_replays_its_day_at_the_same_clock_time():
    # a week ago at 14:00, the train had left B (13:55, 5 min late)
    assert last_known("week ago", 3, "arr", replay=True) == (5, 40, 1)


def test_context_row_before_that_clock_time_looks_before_its_own_event():
    # the arrival at B a week ago (13:45) came before 14:00: the last known delay is the
    # departure from A
    assert last_known("week ago", 2, "arr", replay=True) == (2, 40, 1)
