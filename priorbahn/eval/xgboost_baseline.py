"""
XGBoost baseline: one quantile model per request day, trained on the history.

For query rows of day D, the model learns from the usable rows of the HISTORY_DAYS days
before D (the same history as the count-based baselines), with at most CAP_PER_TYPE rows
per train type, so that S-Bahn and regional trains do not take up most of the model.

The features are those of TabPFN plus the last known delay, and the target is the change
from it, as in the final TabPFN setup. A training row has no single "now", so each gets
its own: a random time up to MAX_AHEAD minutes before its planned event. Query rows use
their request's "now".
"""

from datetime import date

import numpy as np
import polars as pl
import xgboost as xgb

from priorbahn.eval.baselines import _history
from priorbahn.model import last_known
from priorbahn.model.features import EVENTS, shared_categories, to_frame
from priorbahn.model.predict import QUANTILES

CAP_PER_TYPE = 200_000
MAX_AHEAD = 360  # minutes; 75% of the scored validation arrivals are under 274 ahead
ROUNDS = 300
PARAMS = {
    "objective": "reg:quantileerror",
    "quantile_alpha": QUANTILES,
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
}
EXTRA = last_known.COLUMNS


def training_rows(
    stops: pl.LazyFrame, day: date, model: str, seed: int = 0
) -> pl.DataFrame:
    """The capped history of `day` with the last known delay at a random "now"."""
    rows = (
        _history(stops, day, model)
        .filter(
            pl.int_range(pl.len()).shuffle(seed=seed).over("train_type") < CAP_PER_TYPE
        )
        .collect()
    )
    rng = np.random.default_rng(seed)
    ahead = pl.Series(rng.uniform(0, MAX_AHEAD, len(rows)))
    rows = rows.with_columns(
        _cutoff=pl.col(f"planned_{model}") - pl.duration(minutes=1) * ahead
    )
    # one day at a time: the join with all events of the runs is large
    parts = [
        last_known.at_cutoff(part, stops, pl.col("_cutoff"), model)
        for part in rows.partition_by("run_day", maintain_order=True)
    ]
    return pl.concat(parts).drop("_cutoff")


def _change_target(rows: pl.DataFrame, model: str) -> np.ndarray:
    base = rows["last_known_delay"].fill_null(0).cast(pl.Float64)
    return (rows[EVENTS[model]["target"]].cast(pl.Float64) - base).to_numpy()


def xgboost_quantiles(
    stops: pl.LazyFrame, query: pl.DataFrame, model: str
) -> pl.DataFrame:
    """q50, q80, q95 per query row, in query order, null where the event does not exist."""
    qcols = [f"q{round(q * 100)}" for q in QUANTILES]
    query = last_known.at_cutoff(query, stops, pl.col("now"), model).with_row_index(
        "_row"
    )
    has_event = pl.col(EVENTS[model]["numeric"][0]).is_not_null()
    out = []
    # one model per request day: a run that starts after midnight belongs to the request
    # of the day before, whose history must not include that day
    for (day,), rows in query.filter(has_event).group_by(
        pl.col("now").dt.date(), maintain_order=True
    ):
        train = training_rows(stops, day, model)
        categories = shared_categories(train, rows)
        dtrain = xgb.DMatrix(
            to_frame(train, model, categories, EXTRA),
            _change_target(train, model),
            enable_categorical=True,
        )
        booster = xgb.train(PARAMS, dtrain, num_boost_round=ROUNDS)
        dquery = xgb.DMatrix(
            to_frame(rows, model, categories, EXTRA), enable_categorical=True
        )
        # separate quantile trees can cross: sort each row's quantiles
        pred = np.sort(booster.predict(dquery), axis=1).astype(np.float64)
        pred += (
            rows["last_known_delay"].fill_null(0).cast(pl.Float64).to_numpy()[:, None]
        )
        out.append(
            rows.select("_row").hstack(pl.DataFrame(pred, schema=qcols, orient="row"))
        )
        print(f"xgboost: {day:%Y-%m-%d} done, {len(train):,} training rows", flush=True)

    return (
        query.select("_row")
        .join(pl.concat(out), on="_row", how="left")
        .sort("_row")
        .select(qcols)
    )
