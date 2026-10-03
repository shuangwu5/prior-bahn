"""Feature lists and the conversion of stop rows into TabPFN input.

The lists follow "Feature lists (decided)" in docs/data-prep-plan.md.
"""

import pandas as pd
import polars as pl

CATEGORICAL = ["train_type", "station", "train_key", "line_number"]
SHARED_NUMERIC = ["weekday", "run_planned_min", "stop_num", "n_stops", "stop_frac"]

# model name -> columns of the stops table that describe that model's event
EVENTS = {
    "arr": {
        "target": "arr_delay",
        "canceled": "arr_canceled",
        "numeric": ["arr_hour", "arr_minute"],
    },
    "dep": {
        "target": "dep_delay",
        "canceled": "dep_canceled",
        "numeric": ["dep_hour", "dep_minute", "dwell_planned_min"],
    },
}


def feature_columns(model: str) -> list[str]:
    return CATEGORICAL + SHARED_NUMERIC + EVENTS[model]["numeric"]


def usable_rows(df: pl.DataFrame, model: str) -> pl.DataFrame:
    """Rows that can serve as context for `model`: event exists, not canceled, target known."""
    event = EVENTS[model]
    return df.filter(pl.col(event["target"]).is_not_null() & ~pl.col(event["canceled"]))


def to_frame(df: pl.DataFrame, model: str, categories: dict[str, list]) -> pd.DataFrame:
    """Pandas frame for TabPFN with the categorical columns as `category` dtype.

    `categories` is built once from the context and query rows together (see
    `shared_categories`), so both frames use the same category list. TabPFN
    treats a column declared as category as one at any size.
    """
    X = df.select(feature_columns(model)).to_pandas()
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col], categories=categories[col])
    return X


def shared_categories(*frames: pl.DataFrame) -> dict[str, list]:
    stacked = pl.concat([f.select(CATEGORICAL) for f in frames])
    return {c: stacked[c].drop_nulls().unique().sort().to_list() for c in CATEGORICAL}
