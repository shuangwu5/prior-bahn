"""Context builder checks on the real stops table and on small made-up ones (no API calls).

uv run --no-sync pytest tests/test_context.py
"""

from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl
import pytest

from dbdelay.model.context import build_context
from dbdelay.model.features import (
    feature_columns,
    shared_categories,
    to_frame,
    usable_rows,
)

STOPS = Path(__file__).resolve().parents[1] / "data/processed/stops.parquet"
DAY = date(2026, 9, 20)


@pytest.fixture(scope="module")
def stops() -> pl.LazyFrame:
    if not STOPS.exists():
        pytest.skip("run dbdelay/data/prep.py first")
    return pl.scan_parquet(STOPS)


@pytest.fixture(scope="module")
def query(stops) -> pl.DataFrame:
    query = (
        stops.filter(
            pl.col("run_day") == pl.lit(DAY), pl.col("station") == "Berlin Hauptbahnhof"
        )
        .head(30)
        .collect()
    )
    assert len(query) > 0, "empty query would make the tests below meaningless"
    return query


def test_context_only_uses_earlier_days(stops, query):
    context = build_context(stops, query, DAY, size=500)
    assert 0 < len(context) <= 500
    assert context["run_day"].max() < pl.Series([DAY]).cast(pl.Datetime("ns"))[0]
    assert context["run_id"].is_in(query["run_id"].implode()).sum() == 0
    assert (
        not context.select(pl.struct("run_id", "stop_num").is_duplicated())
        .to_series()
        .any()
    )


def made_up_stops(rows: list[tuple]) -> pl.DataFrame:
    """A small stops table with the columns build_context reads, one row per stop.

    Each row is (run_id, run_day, train_key, planned_arr, planned_dep, arr_delay,
    dep_delay). All rows are at station "S". Hours and weekday follow prep.py.
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
        station=pl.lit("S"),
        stop_num=pl.lit(1, pl.Int32),
        arr_hour=pl.col("planned_arr").dt.hour().cast(pl.Float64),
        dep_hour=pl.col("planned_dep").dt.hour().cast(pl.Float64),
        weekday=planned.dt.weekday().cast(pl.UInt8),  # 1 is Monday
    )


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute))


def test_hour_window_wraps_at_midnight():
    # query on a Sunday at 23:xx, by a train that is not in the context
    query = made_up_stops([("q", DAY, "RE 9", None, at(DAY, 23, 30), None, None)])
    sunday, monday = DAY - timedelta(days=7), DAY - timedelta(days=6)
    stops = made_up_stops(
        [
            ("next day 00h", monday, "RE 1", None, at(monday, 0, 10), None, 0),
            ("same day 22h", sunday, "RE 1", None, at(sunday, 22, 10), None, 0),
            ("same day 00h", sunday, "RE 1", None, at(sunday, 0, 10), None, 0),
            ("next day 01h", monday, "RE 1", None, at(monday, 1, 10), None, 0),
        ]
    ).lazy()
    context = build_context(stops, query, DAY, 10, (0, 1, 0))
    assert sorted(context["run_id"]) == ["next day 00h", "same day 22h"]


def test_no_event_on_the_query_day_leaks():
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
    context = build_context(stops, query, DAY, 10, (1, 0, 0))
    assert sorted(context["run_id"]) == ["before midnight", "earlier day"]


@pytest.mark.parametrize("model", ["arr", "dep"])
def test_features_share_categories(stops, query, model):
    context = build_context(stops, query, DAY, size=500)
    train = usable_rows(context, model)
    cats = shared_categories(train, query)
    X_train, X_query = to_frame(train, model, cats), to_frame(query, model, cats)
    assert list(X_train.columns) == feature_columns(model)
    assert X_train["station"].cat.categories.equals(X_query["station"].cat.categories)
    assert X_query["station"].notna().all()
