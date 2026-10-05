"""
Build the one shared context for a request (docs/plan.md, component 3).

The context is made of rows of the stops table that are known at "now", the time the user
picks (also the earliest departure). It uses the request day before "now" and the
DAYS_BACK days before it, in three groups:
1. the same train: first the stops it passed today before "now" (the strongest signal),
   then its whole rides on the earlier days, most recent day first
2. other trains at the query's stations: first the rows of the 60 minutes before "now",
   which show what is going on right now, then rows of the earlier days in the 60 minutes
   before each query row's planned time
3. a general sample from all stations that fills the rest of the context
Groups 1 and 2 take every row they find, as long as the context has room. Groups 2 and 3
are split evenly over the train types of the query. The same context serves both the
arrival model and the departure model.

S-Bahn delays behave differently from those of other trains. When no query train is an
S-Bahn, S-Bahn rows are left out of the second group. The third group has no S-Bahn rows
then anyway, since it only takes the train types of the query.
"""

import math
from datetime import datetime, time, timedelta

import polars as pl

DEFAULT_SIZE = 10_000
DAYS_BACK = 7  # earlier days the context looks at, besides the request day
SAME_TRAIN_DAYS = 7  # earlier days of the same train's rides (group 1)
WINDOW_MIN = 60  # context rows come from the 60 minutes before the query time
# The general group is split evenly over the train types of the query. To avoid loading
# all rows of a type (S-Bahn alone has about 13M), we first keep only the runs whose hashed
# run_id falls under a per mille share chosen per type, so that each type gives about
# twice the rows it needs (at least this many). Then the final rows are sampled from those.
GENERAL_POOL_ROWS_PER_TYPE = 5000


def with_minute_of_day(
    lf: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame | pl.DataFrame:
    """Planned minute of the day of the event the row is about (departure, or arrival at the last stop)."""
    return lf.with_columns(
        pl.coalesce(
            pl.col("dep_hour") * 60 + pl.col("dep_minute"),
            pl.col("arr_hour") * 60 + pl.col("arr_minute"),
        )
        .cast(pl.Int16)
        .alias("minute_of_day")
    )


def last_event_time() -> pl.Expr:
    """Latest planned or actual (planned + delay) time of the row's arrival and departure."""
    return pl.max_horizontal(
        "planned_arr",
        "planned_dep",
        pl.col("planned_arr") + pl.duration(minutes=pl.col("arr_delay")),
        pl.col("planned_dep") + pl.duration(minutes=pl.col("dep_delay")),
    )


def actual_time(event: str) -> pl.Expr:
    """Actual time of the row's arrival ("arr") or departure ("dep"): planned plus delay."""
    return pl.col(f"planned_{event}") + pl.duration(minutes=pl.col(f"{event}_delay"))


def known_at(stops: pl.LazyFrame, now: datetime) -> pl.LazyFrame:
    """
    The stops table as it was known at `now`.

    An event is known only if its actual time is before `now`. A planned time before `now`
    is not enough: a late train may not have left yet. The delay of an event that is not
    known is set to empty, and a row with no known delay is dropped.
    """
    return (
        stops.filter(pl.col("run_day") < now)
        .with_columns(
            pl.when(actual_time(e) < now).then(f"{e}_delay").alias(f"{e}_delay")
            for e in ("arr", "dep")
        )
        .filter(pl.col("arr_delay").is_not_null() | pl.col("dep_delay").is_not_null())
    )


def build_context(
    stops: pl.LazyFrame,
    query: pl.DataFrame,
    now: datetime,
    size: int = DEFAULT_SIZE,
    days_back: int = DAYS_BACK,
    same_train_days: int = SAME_TRAIN_DAYS,
    seed: int = 0,
    skip_s_bahn: bool = True,
) -> pl.DataFrame:
    """
    Context rows for `query` (rows of the stops table), as known at `now`.

    Only events with an actual time before `now` are used (see `known_at`), from runs that
    started at most `days_back` days before the request day (`same_train_days` for the
    rides of the same train). The query's own runs appear only in the first group, with the
    stops they passed before `now`.
    """
    # filter for the second group, see the module docstring
    others = (
        pl.col("train_type") != "S"
        if skip_s_bahn and not (query["train_type"] == "S").any()
        else pl.lit(True)
    )
    types = sorted(query["train_type"].unique().drop_nulls())
    of_query_runs = pl.col("run_id").is_in(query["run_id"].unique().implode())

    def since(days: int) -> pl.Expr:
        return pl.col("run_day") >= datetime.combine(
            now.date() - timedelta(days=days), time()
        )

    known = known_at(stops.filter(since(max(days_back, same_train_days))), now).pipe(
        with_minute_of_day
    )

    # one collect for the first two groups: only the query's stations and trains are read
    near = known.filter(
        pl.col("station").is_in(query["station"].unique().implode())
        | pl.col("train_key").is_in(query["train_key"].unique().implode())
    ).collect()
    # the query's own runs today, at the stops they passed before now (not the query rows)
    own = near.filter(of_query_runs).join(
        query.select("run_id", "stop_num"), on=["run_id", "stop_num"], how="anti"
    )
    others_near = near.filter(~of_query_runs)

    same_train = pl.concat(
        [
            # the latest stops are the best match
            own.sort("stop_num", descending=True),
            # whole rides on the earlier days, the most recent days first
            others_near.filter(
                pl.col("train_key").is_in(query["train_key"].unique().implode()),
                since(same_train_days),
            ).sort("run_day", "run_id", "stop_num", descending=[True, False, False]),
        ],
        how="vertical",
    ).head(size)

    local = others_near.filter(
        pl.col("station").is_in(query["station"].unique().implode()),
        others,
        since(days_back),
    )
    # rows of the last 60 minutes before now: the latest known event happened then
    latest = pl.max_horizontal(actual_time("arr"), actual_time("dep"))
    recent = local.filter(latest >= now - timedelta(minutes=WINDOW_MIN))

    # earlier days: for a query row at 14:40, the slot is 13:40 to 14:39 of every day.
    # Past midnight (a query at 00:20) it is 23:20 to 23:59 and 00:00 to 00:19.
    slots = (
        with_minute_of_day(query)
        .select("station", "minute_of_day")
        .drop_nulls()
        .unique()
        .join(pl.DataFrame({"back": range(1, WINDOW_MIN + 1)}), how="cross")
        .select(
            "station",
            ((pl.col("minute_of_day") - pl.col("back")) % 1440)
            .cast(pl.Int16)
            .alias("minute_of_day"),
        )
        .unique()
    )
    in_slot = local.join(slots, on=["station", "minute_of_day"]).join(
        recent.select("run_id", "stop_num"), on=["run_id", "stop_num"], how="anti"
    )

    # the recent rows first, the earlier days next, as long as there is room
    room = size - len(same_train)
    now_rows = _split_by_type(recent, types, room, seed)
    same_slot = pl.concat(
        [now_rows, _split_by_type(in_slot, types, room - len(now_rows), seed)],
        how="vertical",
    )
    first = pl.concat([same_train, same_slot], how="vertical").unique(
        ["run_id", "stop_num"], keep="first", maintain_order=True
    )

    # the general group fills the rest, without rows already taken
    general = _general_sample(
        stops.filter(since(days_back)),
        known.filter(~of_query_runs, since(days_back)),
        types,
        size - len(first),
        seed,
    ).join(first.select("run_id", "stop_num"), on=["run_id", "stop_num"], how="anti")

    return pl.concat([first, general], how="vertical").drop("minute_of_day")


def _general_sample(
    stops: pl.LazyFrame, known: pl.LazyFrame, types: list[str], n: int, seed: int
) -> pl.DataFrame:
    """
    `n` rows from all stations, split evenly over the train types of the query.

    A uniform sample would be about half S-Bahn whatever the request is about, since
    S-Bahn has 46% of all rows. Here each train type of the request gets the same number
    of rows, so a rare type like NJ gets as many as S.
    """
    if n == 0 or not types:
        return known.head(0).collect()
    of_types = pl.col("train_type").is_in(types)

    # per mille of runs to keep per type, so that each type gives about `target` rows (all
    # runs of a type with fewer rows). `stops` holds only the days of the context window.
    counts = stops.filter(of_types).group_by("train_type").len().collect()
    target = max(GENERAL_POOL_ROWS_PER_TYPE, 2 * math.ceil(n / len(types)))
    per_mille = {
        t: min(1000, max(1, math.ceil(1000 * target / n_rows)))
        for t, n_rows in counts.rows()
    }
    # hash() turns each run_id into a fixed integer, so "hash % 1000 < per_mille" keeps the
    # same runs on every call and drops the rest while scanning. Whole runs are kept or
    # dropped, and the kept runs are spread over all days (no head(), the file is day-sorted).
    # A plain filter (not a join) keeps this a scan-time filter.
    pool = known.filter(
        of_types,
        pl.col("run_id").hash(seed) % 1000
        < pl.col("train_type").replace_strict(
            per_mille, default=0, return_dtype=pl.UInt64
        ),
    ).collect()
    return _split_by_type(pool, types, n, seed)


def _split_by_type(
    pool: pl.DataFrame, types: list[str], n: int, seed: int
) -> pl.DataFrame:
    """
    `n` rows of `pool`, the same number for each of `types`.

    When a type has too few rows, the rest is filled with other rows of the pool, of any
    type. When n does not divide, the first types get one row more.
    """
    if n <= 0 or pool.is_empty():
        return pool.head(0)
    pool = pool.with_row_index("_row")
    quota, extra = divmod(n, len(types)) if types else (0, 0)
    picked = pl.concat(
        [pool.head(0)]
        + [
            _sample(pool.filter(pl.col("train_type") == t), quota + (i < extra), seed)
            for i, t in enumerate(types)
        ],
        how="vertical",
    )
    rest = pool.join(picked.select("_row"), on="_row", how="anti")
    filled = _sample(rest, n - len(picked), seed)
    return pl.concat([picked, filled], how="vertical").drop("_row")


def _sample(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    return df.sample(min(n, len(df)), seed=seed) if len(df) else df
