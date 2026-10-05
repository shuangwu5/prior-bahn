"""Count-based baselines: empirical delay quantiles from the recent history.

For query rows of day D, the history is every usable row of the HISTORY_DAYS run days
before D whose events all happen before D starts, the same rule as the TabPFN context
(which also looks back 14 days for the same train). Each baseline returns the same frame as
`dbdelay.model.predict.predict_delays`: q50, q80, q95 per query row, in query order,
null where the model's event does not exist.
"""

from datetime import datetime, time, timedelta

import polars as pl

from dbdelay.model.context import last_event_time
from dbdelay.model.features import EVENTS
from dbdelay.model.predict import QUANTILES

HISTORY_DAYS = 14  # the same as SAME_TRAIN_DAYS in dbdelay/model/request.py
MIN_COUNT = 10  # fewer past rows than this and the next, coarser level is used


def _quantiles(target: str) -> list[pl.Expr]:
    return [
        pl.col(target).quantile(q, interpolation="linear").alias(f"q{round(q * 100)}")
        for q in QUANTILES
    ] + [pl.len().alias("count")]


def _history(stops: pl.LazyFrame, day, model: str) -> pl.LazyFrame:
    event = EVENTS[model]
    return stops.filter(
        pl.col("run_day") < pl.lit(day),
        pl.col("run_day") >= pl.lit(day - timedelta(days=HISTORY_DAYS)),
        # a run of an earlier day can go past midnight: keep its events of day D out
        last_event_time() < datetime.combine(day, time()),
        pl.col(event["target"]).is_not_null(),
        ~pl.col(event["canceled"]),
    ).with_columns(pl.col(event["target"]).cast(pl.Float64))


def _per_day(
    stops: pl.LazyFrame, query: pl.DataFrame, model: str, levels
) -> pl.DataFrame:
    """Quantiles for each query row from the first level with enough history.

    `levels` is a list of key lists, from the finest to the coarsest; an empty list means
    all of the history.
    """
    event = EVENTS[model]
    hour = event["numeric"][0]  # arr_hour or dep_hour
    target = event["target"]
    qcols = [f"q{round(q * 100)}" for q in QUANTILES]

    out = []
    query = query.with_row_index("_row").with_columns(pl.col(hour).cast(pl.Int8))
    for (day,), rows in query.group_by("run_day", maintain_order=True):
        history = _history(stops, day, model).with_columns(pl.col(hour).cast(pl.Int8))
        filled = rows.select("_row").with_columns(
            *(pl.lit(None, pl.Float64).alias(c) for c in qcols)
        )
        for keys in levels:
            if keys:
                # read only the history of the keys this day's rows need
                needed = rows.select(keys).unique().drop_nulls()
                stats = (
                    history.join(needed.lazy(), on=keys)
                    .group_by(keys)
                    .agg(_quantiles(target))
                    .filter(pl.col("count") >= MIN_COUNT)
                    .collect()
                )
                level = rows.select("_row", *keys).join(stats, on=keys, how="left")
            else:
                stats = history.select(_quantiles(target)).collect()
                level = rows.select("_row").join(stats, how="cross")
            # keep values found at a finer level, fill the rest from this level
            filled = filled.join(level.select("_row", *qcols), on="_row", suffix="_l")
            filled = filled.select(
                "_row", *(pl.coalesce(c, f"{c}_l").alias(c) for c in qcols)
            )
        out.append(filled)

    pred = query.select("_row", hour).join(pl.concat(out), on="_row", how="left")
    # no event (arrival at a first stop, departure at a last stop): no prediction
    return (
        pred.sort("_row")
        .with_columns(
            pl.when(pl.col(hour).is_not_null()).then(pl.col(c)).alias(c) for c in qcols
        )
        .select(qcols)
    )


def global_quantiles(
    stops: pl.LazyFrame, query: pl.DataFrame, model: str
) -> pl.DataFrame:
    """The same delay quantiles for every row: those of all past rows."""
    return _per_day(stops, query, model, levels=[[]])


def train_station_quantiles(
    stops: pl.LazyFrame, query: pl.DataFrame, model: str
) -> pl.DataFrame:
    """Quantiles of the same train at the same station on past days. With fewer than
    MIN_COUNT past rows, fall back to the station at the same planned hour, then to all
    past rows."""
    hour = EVENTS[model]["numeric"][0]
    return _per_day(
        stops, query, model, levels=[["train_key", "station"], ["station", hour], []]
    )


def carry_forward(stops: pl.LazyFrame, query: pl.DataFrame, model: str) -> pl.DataFrame:
    """ "The delay stays the same": the median is the delay where the train was last seen
    before now (`seen_delay`, see `dbdelay.eval.requests.request_rows`). The upper
    quantiles add the usual spread of `train_station_quantiles` (its q80 and q95 minus its
    q50). A train not seen yet gets the `train_station_quantiles` values."""
    usual = train_station_quantiles(stops, query, model)
    seen = query["seen_delay"].cast(pl.Float64)
    return usual.select(
        pl.when(seen.is_not_null() & pl.col("q50").is_not_null())
        .then(seen + pl.col(c) - pl.col("q50"))
        .otherwise(pl.col(c))
        .alias(c)
        for c in ("q50", "q80", "q95")
    )
