"""Context builder checks on the real stops table (no API calls).

uv run --no-sync pytest tests/test_context.py
"""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from delaymodel.context import build_context
from delaymodel.features import (
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
        pytest.skip("run scripts/prep_data.py first")
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


@pytest.mark.parametrize("model", ["arr", "dep"])
def test_features_share_categories(stops, query, model):
    context = build_context(stops, query, DAY, size=500)
    train = usable_rows(context, model)
    cats = shared_categories(train, query)
    X_train, X_query = to_frame(train, model, cats), to_frame(query, model, cats)
    assert list(X_train.columns) == feature_columns(model)
    assert X_train["station"].cat.categories.equals(X_query["station"].cat.categories)
    assert X_query["station"].notna().all()
