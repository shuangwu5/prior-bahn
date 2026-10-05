"""
Context builder checks on the real stops table and on small made-up ones (no API calls).

uv run --no-sync pytest tests/test_context.py
"""

from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl
import pytest

from priorbahn.model.context import build_context
from priorbahn.model.features import (
    feature_columns,
    shared_categories,
    to_frame,
    usable_rows,
)

STOPS = Path(__file__).resolve().parents[1] / "data/processed/stops.parquet"
DAY = date(2026, 9, 20)
START = datetime.combine(DAY, time())  # "now" at the start of DAY


@pytest.fixture(scope="module")
def stops() -> pl.LazyFrame:
    if not STOPS.exists():
        pytest.skip("run priorbahn/data/prep.py first")
    return pl.scan_parquet(STOPS)


@pytest.fixture(scope="module")
def query(stops: pl.LazyFrame) -> pl.DataFrame:
    query = (
        stops.filter(
            pl.col("run_day") == pl.lit(DAY), pl.col("station") == "Berlin Hauptbahnhof"
        )
        .head(30)
        .collect()
    )
    assert len(query) > 0, "empty query would make the tests below meaningless"
    return query


def test_context_only_uses_earlier_days(
    stops: pl.LazyFrame, query: pl.DataFrame
) -> None:
    context = build_context(stops, query, START, size=500)
    assert 0 < len(context) <= 500
    assert context["run_day"].max() < pl.Series([DAY]).cast(pl.Datetime("ns"))[0]
    assert context["run_id"].is_in(query["run_id"].implode()).sum() == 0
    assert (
        not context.select(pl.struct("run_id", "stop_num").is_duplicated())
        .to_series()
        .any()
    )


def made_up_stops(rows: list[tuple]) -> pl.DataFrame:
    """
    A small stops table with the columns build_context reads, one row per stop.

    Each row is (run_id, run_day, train_key, planned_arr, planned_dep, arr_delay,
    dep_delay). All rows are at station "S". The train type is the first word of
    train_key. Hours and weekday follow prep.py.
    """
    df = pl.DataFrame(
        rows,
        schema={
            "run_id": pl.String,
            "run_day": pl.Date,
            "train_key": pl.String,
            "planned_arr": pl.Datetime("ns"),
            "planned_dep": pl.Datetime("ns"),
            "arr_delay": pl.Int16,
            "dep_delay": pl.Int16,
        },
        orient="row",
    )
    planned = pl.coalesce("planned_dep", "planned_arr")
    return df.with_columns(
        run_day=pl.col("run_day").cast(pl.Datetime("ns")),
        train_type=pl.col("train_key").str.split(" ").list.first(),
        station=pl.lit("S"),
        stop_num=pl.lit(1, pl.Int32),
        arr_hour=pl.col("planned_arr").dt.hour().cast(pl.Float64),
        arr_minute=pl.col("planned_arr").dt.minute().cast(pl.Float64),
        dep_hour=pl.col("planned_dep").dt.hour().cast(pl.Float64),
        dep_minute=pl.col("planned_dep").dt.minute().cast(pl.Float64),
        weekday=planned.dt.weekday().cast(pl.UInt8),  # 1 is Monday
    )


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute))


def test_window_covers_the_last_7_days_and_wraps_at_midnight() -> None:
    # query at 00:20 by an RE that is not in the context: the window is 23:20 to 23:59 and
    # 00:00 to 00:19 on each of the last 7 days. The context rows are RB trains, so the
    # general group (query train types only) adds nothing.
    query = made_up_stops([("q", DAY, "RE 9", None, at(DAY, 0, 20), None, None)])
    d2, d3, d9 = (DAY - timedelta(days=n) for n in (2, 3, 9))
    stops = made_up_stops(
        [
            ("23:30", d2, "RB 1", None, at(d2, 23, 30), None, 0),
            ("00:10", d3, "RB 1", None, at(d3, 0, 10), None, 0),
            ("23:10", d2, "RB 1", None, at(d2, 23, 10), None, 0),
            ("00:25", d3, "RB 1", None, at(d3, 0, 25), None, 0),
            ("older than 7 days", d9, "RB 1", None, at(d9, 23, 30), None, 0),
        ]
    ).lazy()
    context = build_context(stops, query, START, 10)
    assert sorted(context["run_id"]) == ["00:10", "23:30"]


def test_no_event_on_the_query_day_leaks() -> None:
    # query at 00:30, at the same station and by the same train as the context rows
    query = made_up_stops([("q", DAY, "RE 1", None, at(DAY, 0, 30), None, None)])
    eve = DAY - timedelta(days=1)
    stops = made_up_stops(
        [
            (
                "earlier day",
                eve - timedelta(days=1),
                "RE 1",
                None,
                at(eve, 10),
                None,
                5,
            ),
            ("planned after midnight", eve, "RE 1", None, at(DAY, 1), None, 0),
            ("late after midnight", eve, "RE 1", None, at(eve, 23, 55), None, 10),
            ("late arrival", eve, "RE 1", at(eve, 23, 50), None, 15, None),
            ("before midnight", eve, "RE 1", at(eve, 23, 40), at(eve, 23, 45), 2, 3),
        ]
    ).lazy()
    context = build_context(stops, query, START, 10)
    assert sorted(context["run_id"]) == ["before midnight", "earlier day"]


def test_rows_of_the_request_day_count_only_before_now() -> None:
    # now is 14:00, the query train leaves at 16:00, so the earlier-day slot is 15:00 to
    # 15:59 and the "right now" window is 13:00 to 13:59
    query = made_up_stops([("q", DAY, "ICE 9", None, at(DAY, 16), None, None)])
    stops = made_up_stops(
        [
            ("left 13:35", DAY, "RE 1", None, at(DAY, 13, 30), None, 5),
            ("left before the window", DAY, "RE 1", None, at(DAY, 12, 30), None, 0),
            ("late, not left yet", DAY, "RE 1", None, at(DAY, 13, 50), None, 15),
            (
                "arrived, not left yet",
                DAY,
                "RE 1",
                at(DAY, 13, 40),
                at(DAY, 14, 10),
                0,
                0,
            ),
            ("in the slot, after now", DAY, "RE 1", None, at(DAY, 15, 30), None, 0),
        ]
    ).lazy()
    context = build_context(stops, query, at(DAY, 14), 10)
    assert context.sort("run_id").select("run_id", "arr_delay", "dep_delay").rows() == [
        ("arrived, not left yet", 0, None),
        ("left 13:35", None, 5),
    ]


def test_same_train_group_takes_the_stops_passed_today_first() -> None:
    # now is 14:00. The query is stop 3 of run "q". Stops 1 and 2 were passed before now,
    # stop 2 only arrived (it leaves late, after now). Stop 4 has not happened yet. A week
    # ago, the whole ride counts, also stop 1 at A, which is not a query station.
    week_ago = DAY - timedelta(days=7)
    run = made_up_stops(
        [
            ("q", DAY, "ICE 9", None, at(DAY, 13), None, 4),
            ("q", DAY, "ICE 9", at(DAY, 13, 40), at(DAY, 13, 50), 6, 12),
            ("q", DAY, "ICE 9", at(DAY, 14, 30), at(DAY, 14, 32), None, None),
            ("q", DAY, "ICE 9", at(DAY, 15), None, None, None),
            ("week ago", week_ago, "ICE 9", None, at(week_ago, 13), None, 2),
            ("week ago", week_ago, "ICE 9", at(week_ago, 14, 30), None, 3, None),
        ]
    ).with_columns(
        stop_num=pl.Series([1, 2, 3, 4, 1, 3], dtype=pl.Int32),
        station=pl.Series(["A", "B", "C", "D", "A", "C"]),
    )
    query = run.filter(pl.col("run_id") == "q", pl.col("stop_num") == 3)
    context = build_context(run.lazy(), query, at(DAY, 14), 10)
    assert context.select("run_id", "stop_num", "arr_delay", "dep_delay").rows() == [
        ("q", 2, 6, None),
        ("q", 1, None, 4),
        ("week ago", 1, None, 2),
        ("week ago", 3, 3, None),
    ]


def test_same_station_group_is_split_by_train_type() -> None:
    # 2 ICE rows in the slot: the other 4 come from RE, never from S-Bahn, since the
    # query has no S-Bahn train
    query = made_up_stops([("q", DAY, "ICE 9", None, at(DAY, 10), None, None)])
    week_ago = DAY - timedelta(days=7)
    stops = made_up_stops(
        [
            (f"{key} {i}", week_ago, key, None, at(week_ago, 9, 30), None, 0)
            for key, n in [("ICE 7", 2), ("RE 1", 10), ("S 1", 10)]
            for i in range(n)
        ]
    ).lazy()
    context = build_context(stops, query, START, 6)
    assert context["train_type"].value_counts().sort("train_type").rows() == [
        ("ICE", 2),
        ("RE", 4),
    ]


def test_general_group_is_split_evenly_over_query_train_types() -> None:
    # S-Bahn has ten times more rows than NJ, ICE is not in the query. The rows are at
    # another station than the query, so only the general group can take them.
    query = made_up_stops(
        [
            ("q1", DAY, "S 1", None, at(DAY, 8), None, None),
            ("q2", DAY, "NJ 40", None, at(DAY, 22), None, None),
        ]
    )
    earlier = DAY - timedelta(days=3)
    stops = (
        made_up_stops(
            [
                (f"s{i}", earlier, "S 2", None, at(earlier, 9), None, 0)
                for i in range(40)
            ]
            + [
                (f"n{i}", earlier, "NJ 41", None, at(earlier, 21), None, 5)
                for i in range(4)
            ]
            + [
                (f"i{i}", earlier, "ICE 5", None, at(earlier, 12), None, 1)
                for i in range(4)
            ]
        )
        .with_columns(station=pl.lit("X"))
        .lazy()
    )
    context = build_context(stops, query, START, 8)
    assert context["train_type"].value_counts().sort("train_type").rows() == [
        ("NJ", 4),
        ("S", 4),
    ]


@pytest.mark.parametrize("model", ["arr", "dep"])
def test_features_share_categories(
    stops: pl.LazyFrame, query: pl.DataFrame, model: str
) -> None:
    context = build_context(stops, query, START, size=500)
    train = usable_rows(context, model)
    cats = shared_categories(train, query)
    X_train, X_query = to_frame(train, model, cats), to_frame(query, model, cats)
    assert list(X_train.columns) == feature_columns(model)
    assert X_train["station"].cat.categories.equals(X_query["station"].cat.categories)
    assert X_query["station"].notna().all()


def test_same_train_looks_further_back_than_the_other_groups() -> None:
    # 10 days ago: the query's own train is used (14 days back), another RE at the same
    # station is not (7 days back)
    query = made_up_stops([("q", DAY, "RE 9", None, at(DAY, 10), None, None)])
    ten_days_ago = DAY - timedelta(days=10)
    stops = made_up_stops(
        [
            (
                "same train",
                ten_days_ago,
                "RE 9",
                None,
                at(ten_days_ago, 9, 30),
                None,
                1,
            ),
            (
                "other train",
                ten_days_ago,
                "RE 1",
                None,
                at(ten_days_ago, 9, 30),
                None,
                2,
            ),
        ]
    ).lazy()
    context = build_context(stops, query, START, 10, same_train_days=14)
    assert context["run_id"].to_list() == ["same train"]
