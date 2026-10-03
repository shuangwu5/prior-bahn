"""Polars version of scripts/prep_data.py, kept for reference only.

It builds the same `stops` table (checked column by column against the pandas version) and
is much faster and lighter on memory (about 30 seconds and 7 GB, against 2 minutes and 9 GB).
The pandas version is the one the project uses. This one writes to a different file so it
never overwrites the pandas output.

Run from the repo root: uv run --no-sync python scripts/reference/prep_data_polars.py
"""

from datetime import date
from pathlib import Path

import polars as pl

RAW = "data/monthly_processed_data/data-2026-{month}.parquet"
MONTHS = ["08", "09"]
OUT = Path("data/processed/stops_polars.parquet")
NON_TRAIN = r"(?i)^(bus|sev|bsv)$"

# split by the day a run starts, see plan.md
FIRST_DAY, LAST_DAY = date(2026, 8, 1), date(2026, 9, 30)
VALIDATION_START, TEST_START = date(2026, 9, 17), date(2026, 9, 24)


def minutes(later: str, earlier: str) -> pl.Expr:
    return (pl.col(later) - pl.col(earlier)).dt.total_minutes()


def build() -> pl.LazyFrame:
    # both months are read together: a run that starts on Aug 31 spills into the September file
    raw = pl.concat([pl.scan_parquet(RAW.format(month=m)) for m in MONTHS])
    planned_arr, planned_dep = pl.col("planned_arr"), pl.col("planned_dep")
    run_day = pl.col("run_day")
    # stations are merged by name, the few without a name keep their EVA code
    station = pl.coalesce("station_name", "EVA " + pl.col("eva"))

    kept = (
        raw.filter(
            # a null train_type makes this condition null, and filter drops those rows
            ~pl.col("train_type").str.contains(NON_TRAIN),
            ~pl.col("is_replacement_train"),
            ~pl.col("is_additional_stop"),
            pl.col("arrival_planned_time").is_not_null()
            | pl.col("departure_planned_time").is_not_null(),
        )
        .with_columns(
            # `id` is <ride id>-<run start YYMMDDHHMM>-<stop number>
            pl.col("id").str.extract(r"^(.*)-\d+$", 1).alias("run_id"),
            pl.col("id")
            .str.extract(r"-(\d{6})\d{4}-\d+$", 1)
            .str.to_date("%y%m%d")
            .alias("run_day"),
        )
        .filter(run_day.is_between(FIRST_DAY, LAST_DAY))
    )

    # The source writes destinations like xml_station_name ("Frankfurt(Main)Süd"), while
    # station_name has "Frankfurt (Main) Süd". So each destination is translated through
    # the xml_station_name -> station pairs seen in the data. An xml name that belongs to
    # two different stations cannot be translated and is left out of the lookup.
    xml_to_station = (
        kept.select(
            pl.col("xml_station_name").alias("destination_xml"),
            station.alias("destination_station"),
        )
        .unique()
        .filter(pl.len().over("destination_xml") == 1)
    )

    return (
        kept.join(
            xml_to_station,
            left_on="final_destination_station",
            right_on="destination_xml",
            how="left",
        )
        .select(
            "run_id",
            "run_day",
            pl.when(run_day < VALIDATION_START)
            .then(pl.lit("context"))
            .when(run_day < TEST_START)
            .then(pl.lit("validation"))
            .otherwise(pl.lit("test"))
            .alias("split"),
            "train_type",
            (pl.col("train_type") + " " + pl.col("train_number")).alias("train_key"),
            "line_number",
            # places that are not stops in our data (Basel SBB, Enschede, bus stops) keep
            # their source spelling, and the source leaves the destination empty at the
            # last stop of a run, where it is the station itself
            pl.coalesce(
                "destination_station", "final_destination_station", station
            ).alias("final_destination"),
            station.alias("station"),
            pl.col("train_line_station_num").alias("stop_num"),
            pl.col("arrival_planned_time").alias("planned_arr"),
            pl.col("departure_planned_time").alias("planned_dep"),
            pl.col("arrival_is_canceled").alias("arr_canceled"),
            pl.col("departure_is_canceled").alias("dep_canceled"),
            # delay_in_min is the delay of the row's own event (see the exploration notebook),
            # so both delays are recomputed. Canceled events have no usable delay.
            pl.when(~pl.col("arrival_is_canceled"))
            .then(minutes("arrival_change_time", "arrival_planned_time"))
            .cast(pl.Int16)
            .alias("arr_delay"),
            pl.when(~pl.col("departure_is_canceled"))
            .then(minutes("departure_change_time", "departure_planned_time"))
            .cast(pl.Int16)
            .alias("dep_delay"),
        )
        .sort("run_day", "run_id", "stop_num")
        .with_columns(
            # the source has missing stops (4% of runs), so the previous row is only the
            # previous stop when the stop numbers are adjacent
            (
                pl.col("stop_num") - pl.col("stop_num").shift(1).over("run_id") == 1
            ).alias("prev_adjacent"),
            pl.col("station").shift(1).over("run_id").alias("prev_station"),
            planned_dep.shift(1).over("run_id").alias("prev_planned_dep"),
            (planned_dep - planned_arr)
            .dt.total_minutes()
            .cast(pl.Int16)
            .alias("dwell_planned_min"),
            pl.col("stop_num").max().over("run_id").alias("n_stops"),
            pl.coalesce(planned_dep, planned_arr).alias("planned_time"),
        )
        .with_columns(
            pl.when(pl.col("prev_adjacent"))
            .then(pl.col("prev_station"))
            .alias("prev_station"),
            pl.when(pl.col("prev_adjacent"))
            .then((planned_arr - pl.col("prev_planned_dep")).dt.total_minutes())
            .cast(pl.Int16)
            .alias("run_planned_min"),
            (pl.col("stop_num") / pl.col("n_stops")).alias("stop_frac"),
            pl.col("planned_time").dt.hour().cast(pl.UInt8).alias("hour"),
            pl.col("planned_time").dt.weekday().cast(pl.UInt8).alias("weekday"),
            (pl.col("planned_time").dt.weekday() >= 6).alias("is_weekend"),
        )
        .drop("planned_time", "prev_adjacent", "prev_planned_dep")
    )


def main() -> None:
    n_raw = sum(
        pl.scan_parquet(RAW.format(month=m)).select(pl.len()).collect().item()
        for m in MONTHS
    )
    df = build().collect()

    n_unique = df.select(pl.struct("run_id", "stop_num").n_unique()).item()
    assert n_unique == df.height, "(run_id, stop_num) is not unique"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT, compression="zstd")
    print(f"{n_raw:,} raw rows -> {df.height:,} rows, {df['run_id'].n_unique():,} runs")
    print(df.group_by("split").len().sort("split"))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
