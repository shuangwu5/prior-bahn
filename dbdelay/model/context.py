"""Build the one shared context for a request (docs/plan.md, component 3).

The context is made of rows of the stops table from days before the request day, in
three groups: the same train at the same station, other trains at those stations around
the same hour and weekday, and a small general sample. The same context serves both the
arrival model and the departure model.
"""

from datetime import date

import polars as pl

DEFAULT_SIZE = 2000
DEFAULT_SHARES = (0.4, 0.4, 0.2)  # same train, same station and time, general
HOUR_WINDOW = 1  # hours either side of the query hour
# The general group is a random sample of the whole pool (about 28M rows). To avoid loading
# all of it, we first keep only the runs whose hashed run_id falls in this many per mille
# (2 per mille is about 50k rows), then sample the final rows from those.
GENERAL_HASH_SHARE = 2


def with_hour(lf: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """Hour of the event the row is about (departure, or arrival at the last stop)."""
    return lf.with_columns(
        pl.coalesce("dep_hour", "arr_hour").cast(pl.Int8).alias("hour")
    )


def build_context(
    stops: pl.LazyFrame,
    query: pl.DataFrame,
    day: date,
    size: int = DEFAULT_SIZE,
    shares: tuple[float, float, float] = DEFAULT_SHARES,
    seed: int = 0,
) -> pl.DataFrame:
    """Context rows for `query` (rows of the stops table), using only run days before `day`.

    Rows of the query's own runs are excluded, since a run that started the evening before
    can still be running on `day`.
    """
    n_same_train, n_same_slot, n_general = (round(size * s) for s in shares)
    before = (
        stops.filter(pl.col("run_day") < pl.lit(day))
        .filter(~pl.col("run_id").is_in(query["run_id"].unique().implode()))
        .pipe(with_hour)
    )

    # one collect for the two station-based groups: only the query's stations are read
    local = before.filter(
        pl.col("station").is_in(query["station"].unique().implode())
    ).collect()

    pairs = query.select("train_key", "station").unique()
    same_train = (
        local.join(pairs, on=["train_key", "station"])
        .sort("run_day", descending=True)  # the most recent days are the best match
        .head(n_same_train)
    )

    # query slots widened by the hour window, same weekday
    slots = (
        with_hour(query)
        .select("station", "weekday", "hour")
        .drop_nulls()
        .unique()
        .join(
            pl.DataFrame({"shift": range(-HOUR_WINDOW, HOUR_WINDOW + 1)}), how="cross"
        )
        .select("station", "weekday", (pl.col("hour") + pl.col("shift")).alias("hour"))
        .unique()
        .with_columns(pl.col("hour").cast(pl.Int8))
    )
    same_slot = _sample(
        local.join(slots, on=["station", "weekday", "hour"]), n_same_slot, seed
    )

    # hash() turns each run_id into a fixed integer, so "hash % 1000 < 2" keeps the same
    # ~0.2% of runs on every call and drops the rest while scanning. Whole runs are kept or
    # dropped, and the kept runs are spread over all days (no head(), the file is day-sorted).
    general_pool = before.filter(
        pl.col("run_id").hash(seed) % 1000 < GENERAL_HASH_SHARE
    ).collect()
    general = _sample(general_pool, n_general, seed)

    context = pl.concat([same_train, same_slot, general], how="vertical")
    return context.unique(
        ["run_id", "stop_num"], keep="first", maintain_order=True
    ).drop("hour")


def _sample(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    return df.sample(min(n, len(df)), seed=seed) if len(df) else df
