"""Build the one shared context for a request (docs/plan.md, component 3).

The context is made of rows of the stops table from days before the request day, in
three groups: the same train at the same station, other trains at those stations around
the same hour and weekday, and a small general sample from all stations, split evenly
over the train types of the query. The same context serves both the
arrival model and the departure model.
"""

import math
from datetime import date, datetime, time

import polars as pl

DEFAULT_SIZE = 2000
DEFAULT_SHARES = (0.4, 0.4, 0.2)  # same train, same station and time, general
HOUR_WINDOW = 1  # hours either side of the query hour
# The general group is split evenly over the train types of the query. To avoid loading
# all rows of a type (S-Bahn alone has about 13M), we first keep only the runs whose hashed
# run_id falls under a per mille share chosen per type, so that each type gives about this
# many rows. Then the final rows are sampled from those.
GENERAL_POOL_ROWS_PER_TYPE = 5000


def with_hour(lf: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """Hour of the event the row is about (departure, or arrival at the last stop)."""
    return lf.with_columns(
        pl.coalesce("dep_hour", "arr_hour").cast(pl.Int8).alias("hour")
    )


def last_event_time() -> pl.Expr:
    """Latest planned or actual (planned + delay) time of the row's arrival and departure."""
    return pl.max_horizontal(
        "planned_arr",
        "planned_dep",
        pl.col("planned_arr") + pl.duration(minutes=pl.col("arr_delay")),
        pl.col("planned_dep") + pl.duration(minutes=pl.col("dep_delay")),
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

    A run that started the evening before can still be running on `day`. So a row is kept
    only if all its events, planned and actual, happened before `day` began: what happened
    after the query time must not leak into the context. Rows of the query's own runs are
    excluded too.
    """
    n_same_train, n_same_slot, n_general = (round(size * s) for s in shares)
    before = (
        stops.filter(pl.col("run_day") < pl.lit(day))
        .filter(last_event_time() < datetime.combine(day, time()))
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

    # query slots widened by the hour window, same weekday. The window wraps at
    # midnight: hour 23 also takes hour 0 of the next weekday, hour 0 also takes hour 23
    # of the weekday before (weekday 1 is Monday, 7 is Sunday).
    hours = pl.col("hour") + pl.col("shift")
    weekday = pl.col("weekday").cast(pl.Int16) - 1 + hours // 24
    slots = (
        with_hour(query)
        .select("station", "weekday", "hour")
        .drop_nulls()
        .unique()
        .join(
            pl.DataFrame({"shift": range(-HOUR_WINDOW, HOUR_WINDOW + 1)}), how="cross"
        )
        .select(
            "station",
            (weekday % 7 + 1).cast(pl.UInt8).alias("weekday"),
            (hours % 24).cast(pl.Int8).alias("hour"),
        )
        .unique()
    )
    same_slot = _sample(
        local.join(slots, on=["station", "weekday", "hour"]), n_same_slot, seed
    )

    general = _general_sample(stops, before, query, n_general, seed)

    context = pl.concat([same_train, same_slot, general], how="vertical")
    return context.unique(
        ["run_id", "stop_num"], keep="first", maintain_order=True
    ).drop("hour")


def _general_sample(
    stops: pl.LazyFrame, before: pl.LazyFrame, query: pl.DataFrame, n: int, seed: int
) -> pl.DataFrame:
    """`n` rows from all stations, split evenly over the train types of the query.

    A uniform sample would be about half S-Bahn whatever the request is about, since
    S-Bahn has 46% of all rows. Here each train type of the request gets the same number
    of rows, so a rare type like NJ gets as many as S.
    """
    if n == 0:
        return before.head(0).collect()
    types = sorted(query["train_type"].unique().drop_nulls())
    if not types:
        return before.head(0).collect()
    of_types = pl.col("train_type").is_in(types)

    # per mille of runs to keep per type, so that each type gives about
    # GENERAL_POOL_ROWS_PER_TYPE rows (all runs of a type with fewer rows)
    counts = stops.filter(of_types).group_by("train_type").len().collect()
    per_mille = {
        t: min(1000, max(1, math.ceil(1000 * GENERAL_POOL_ROWS_PER_TYPE / n_rows)))
        for t, n_rows in counts.rows()
    }
    # hash() turns each run_id into a fixed integer, so "hash % 1000 < per_mille" keeps the
    # same runs on every call and drops the rest while scanning. Whole runs are kept or
    # dropped, and the kept runs are spread over all days (no head(), the file is day-sorted).
    # A plain filter (not a join) keeps this a scan-time filter.
    pool = before.filter(
        of_types,
        pl.col("run_id").hash(seed) % 1000
        < pl.col("train_type").replace_strict(
            per_mille, default=0, return_dtype=pl.UInt64
        ),
    ).collect()

    # n split evenly, the first types get one row more when n does not divide
    quota, extra = divmod(n, len(types))
    return pl.concat(
        [
            _sample(pool.filter(pl.col("train_type") == t), quota + (i < extra), seed)
            for i, t in enumerate(types)
        ],
        how="vertical",
    )


def _sample(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    return df.sample(min(n, len(df)), seed=seed) if len(df) else df
