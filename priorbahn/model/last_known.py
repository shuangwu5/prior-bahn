"""
The last known delay of a train before "now".

For every row and one event ("arr" or "dep"), three columns are added:
- `last_known_delay`: the delay at the train's last event before the cutoff (empty: the
  train had not started)
- `minutes_since_known`: planned minutes from that event to the row's event
- `stops_since_known`: stops from that event's stop to the row's stop (0 for a departure
  predicted from the arrival at the same stop)

Query rows use "now" as the cutoff. Context rows replay their own day at the same clock
time as "now", and never look past "now" or past their own event.
"""

from datetime import datetime

import polars as pl

from priorbahn.model.context import actual_time

COLUMNS = ("last_known_delay", "minutes_since_known", "stops_since_known")


def with_last_known(
    rows: pl.DataFrame,
    stops: pl.LazyFrame,
    now: datetime,
    model: str,
    replay: bool,
) -> pl.DataFrame:
    """
    `rows` with the COLUMNS for their `model` event.

    `replay` is False for query rows (the cutoff is `now`) and True for context rows (the
    cutoff is the earliest of `now`, the same clock time on the day of the row's event, and
    the row's own event).
    """
    planned = pl.col(f"planned_{model}")
    if replay:
        clock = now - datetime.combine(now.date(), datetime.min.time())
        cutoff = pl.min_horizontal(
            pl.lit(now),
            planned.dt.truncate("1d") + pl.lit(clock),
            actual_time(model),
        )
    else:
        cutoff = pl.lit(now)
    return at_cutoff(rows, stops, cutoff, model)


def at_cutoff(rows: pl.DataFrame, stops: pl.LazyFrame, cutoff: pl.Expr, model: str) -> pl.DataFrame:
    """
    `rows` with the COLUMNS for their `model` event, known before `cutoff`, an
    expression on `rows` that can differ per row.
    """
    planned = pl.col(f"planned_{model}")
    # all stops of the rows' runs, numbered 1, 2, 3, ... by planned time within each run
    runs = (
        stops.filter(pl.col("run_id").is_in(rows["run_id"].unique().implode()))
        .collect()
        .with_columns(pl.coalesce("planned_arr", "planned_dep").rank("ordinal").over("run_id").alias("stop_index"))
    )
    events = pl.concat(
        [
            runs.select(
                "run_id",
                pl.col("stop_index").alias("known_index"),
                pl.lit(e == "arr").alias("known_is_arr"),
                actual_time(e).alias("known_time"),
                pl.col(f"planned_{e}").alias("known_planned"),
                pl.col(f"{e}_delay").alias("last_known_delay"),
            )
            for e in ("arr", "dep")
        ]
    ).filter(pl.col("known_time").is_not_null())

    targets = rows.with_row_index("_row").join(
        runs.select("run_id", "stop_num", "stop_index"),
        on=["run_id", "stop_num"],
        how="left",
    )
    earlier_stop = pl.col("known_index") < pl.col("stop_index")
    # a departure may be predicted from the arrival at the same stop
    same_stop_arrival = (pl.col("known_index") == pl.col("stop_index")) & pl.col("known_is_arr")
    if model == "arr":
        same_stop_arrival = pl.lit(False)
    last = (
        targets.select("_row", "run_id", "stop_index", planned, cutoff.alias("cutoff"))
        .join(events, on="run_id")
        .filter(pl.col("known_time") < pl.col("cutoff"), earlier_stop | same_stop_arrival)
        .sort("known_time")
        .group_by("_row")
        .agg(pl.all().last())
        .select(
            "_row",
            "last_known_delay",
            (planned - pl.col("known_planned")).dt.total_minutes().alias("minutes_since_known"),
            (pl.col("stop_index") - pl.col("known_index")).alias("stops_since_known"),
        )
    )
    return targets.join(last, on="_row", how="left").sort("_row").drop("_row", "stop_index")
