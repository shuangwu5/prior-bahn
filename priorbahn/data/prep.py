"""
Build the transformed `stops` table from the raw monthly files.

Follows docs/data-prep-plan.md: schedule-only columns, no lagged history features.
Run from the repo root: uv run --no-sync python -m priorbahn.data.prep
"""

from datetime import date
from pathlib import Path

import polars as pl

RAW = "data/monthly_processed_data/data-2026-{month}.parquet"
MONTHS = ["09"]  # TabPFN and the baselines look back at most 14 days
OUT = Path("data/processed/stops.parquet")
NON_TRAIN = "^(?:bus|sev|bsv)$"  # matched without regard to case
# historic, special and one-off trains with no stable history, plus a few malformed
# type labels, see docs/data-prep-plan.md section 4. UEX is kept: holiday night trains.
DROP_TYPES = [
    # historic and non-passenger
    "DB", "PRE", "MBB", "P", "SDG", "DPN", "ÖBA", "KTB", "UEF",
    # one-off specials and charters (1 to 40 days, few trains)
    "DRC-L", "DRC", "CLB", "Bvs", "L-S", "DBK", "MSM", "RBP", "DRB-G", "SPNV", "SVG",
    "LEO", "Sp",
    # malformed labels of regular trains
    "RE1", "RE 4", "2", ".",
]  # fmt: skip
# Delays and planned legs longer than this are date errors of one day in the source
# (real values stay under 900 min, the errors are close to 1440)
MAX_PLAUSIBLE_MIN = 1200

# split by the day a run starts, see docs/plan.md
FIRST_DAY, LAST_DAY = date(2026, 9, 1), date(2026, 9, 30)
VALIDATION_START, TEST_START = date(2026, 9, 17), date(2026, 9, 24)

# only these raw columns are needed, loading fewer columns saves a lot of memory
RAW_COLUMNS = [
    "id",
    "station_name",
    "xml_station_name",
    "eva",
    "train_number",
    "line_number",
    "final_destination_station",
    "train_type",
    "is_replacement_train",
    "is_additional_stop",
    "arrival_is_canceled",
    "departure_is_canceled",
    "train_line_station_num",
    "arrival_planned_time",
    "arrival_change_time",
    "departure_planned_time",
    "departure_change_time",
]


def minutes_between(later: pl.Expr, earlier: pl.Expr) -> pl.Expr:
    """
    Whole minutes from `earlier` to `later`. Missing times give null.
    """
    return (later - earlier).dt.total_minutes().cast(pl.Int16)


def null_date_errors(minutes: pl.Expr) -> pl.Expr:
    """
    Set values that are off by about a day to null.

    The source sometimes has a planned or actual time on the wrong day. We cannot tell
    which side is wrong, so the value is not fixed but left empty.
    """
    return pl.when(minutes.abs() <= MAX_PLAUSIBLE_MIN).then(minutes)


def load_month(month: str) -> pl.LazyFrame:
    """
    Read one raw file, drop rows we never use and add run_id and run_day.
    """
    df = pl.scan_parquet(RAW.format(month=month)).select(RAW_COLUMNS)

    # rows without a train type are dropped too (about 37k rows, none has a planned time)
    train_type = pl.col("train_type")
    is_train = (
        train_type.is_not_null()
        & ~train_type.str.contains(f"(?i){NON_TRAIN}")
        & ~train_type.is_in(DROP_TYPES)
    )
    has_planned_time = (
        pl.col("arrival_planned_time").is_not_null()
        | pl.col("departure_planned_time").is_not_null()
    )
    keep = (
        is_train
        & ~pl.col("is_replacement_train")
        & ~pl.col("is_additional_stop")
        & has_planned_time
    )
    df = df.filter(keep)

    # `id` is <ride id>-<run start YYMMDDHHMM>-<stop number>.
    # run_id is everything before the stop number, run_day is the date inside it.
    pattern = r"^(.*-(\d{6})\d{4})-\d+$"
    df = df.with_columns(
        run_id=pl.col("id").str.extract(pattern, 1),
        run_day=pl.col("id")
        .str.extract(pattern, 2)
        .str.strptime(pl.Datetime("ns"), "%y%m%d"),
    )
    return df.drop("id")


def destination_in_station_spelling(df: pl.DataFrame) -> pl.Series:
    """
    Final destination of each row, spelled like its `station`.

    The source writes destinations like xml_station_name ("Frankfurt(Main)Süd"), while
    station_name has "Frankfurt (Main) Süd". So each destination is translated through
    the xml_station_name -> station pairs seen in the data. An xml name that belongs to
    two different stations cannot be translated and is left out of the lookup.
    """
    pairs = (
        df.select(xml="xml_station_name", translated="station")
        .unique()
        .filter(pl.len().over("xml") == 1)
    )
    # places that are not stops in our data (Basel SBB, Enschede, bus stops) keep their
    # source spelling, and the source leaves the destination empty at the last stop of a
    # run, where it is the station itself
    return (
        df.select("final_destination_station", "station")
        .join(
            pairs,
            left_on="final_destination_station",
            right_on="xml",
            how="left",
            nulls_equal=True,
            maintain_order="left",
        )
        .select(pl.coalesce("translated", "final_destination_station", "station"))
        .to_series()
    )


def build() -> pl.DataFrame:
    # a run that starts on Aug 31 is in this file too, with its September stops: it is cut
    # below, since a run belongs to the day it starts
    df = (
        pl.concat([load_month(m) for m in MONTHS])
        .filter(pl.col("run_day").is_between(FIRST_DAY, LAST_DAY))
        # stations are merged by name, the few without a name keep their EVA code
        .with_columns(
            station=pl.coalesce("station_name", pl.lit("EVA ") + pl.col("eva"))
        )
        .collect()
    )

    arr_canceled, dep_canceled = (
        pl.col("arrival_is_canceled"),
        pl.col("departure_is_canceled"),
    )
    out = df.select(
        "run_id",
        "run_day",
        "train_type",
        train_key=pl.col("train_type") + " " + pl.col("train_number"),
        line_number="line_number",
        final_destination=destination_in_station_spelling(df),
        station="station",
        stop_num="train_line_station_num",
        planned_arr="arrival_planned_time",
        planned_dep="departure_planned_time",
        arr_canceled=arr_canceled,
        dep_canceled=dep_canceled,
        # delay_in_min is the delay of the row's own event (see the exploration
        # notebook), so both delays are recomputed from the time columns.
        # A canceled event has no usable delay, so it becomes null.
        arr_delay=pl.when(~arr_canceled).then(
            null_date_errors(
                minutes_between(
                    pl.col("arrival_change_time"), pl.col("arrival_planned_time")
                )
            )
        ),
        dep_delay=pl.when(~dep_canceled).then(
            null_date_errors(
                minutes_between(
                    pl.col("departure_change_time"), pl.col("departure_planned_time")
                )
            )
        ),
    )

    # sort so that the rows of one run are next to each other, in stop order
    out = out.sort("run_day", "run_id", "stop_num", nulls_last=True)

    # `shift()` moves a column down by one row, so each row can see the row above it.
    # The row above is the previous stop only if it is in the same run and the stop
    # numbers are adjacent (the source has missing stops in about 4% of runs).
    same_run = pl.col("run_id") == pl.col("run_id").shift()
    prev_is_adjacent = same_run & (pl.col("stop_num") - pl.col("stop_num").shift() == 1)
    out = out.with_columns(
        prev_station=pl.when(prev_is_adjacent).then(pl.col("station").shift()),
        run_planned_min=pl.when(prev_is_adjacent).then(
            null_date_errors(
                minutes_between(pl.col("planned_arr"), pl.col("planned_dep").shift())
            )
        ),
        dwell_planned_min=minutes_between(pl.col("planned_dep"), pl.col("planned_arr")),
        # highest stop number seen in each run
        n_stops=pl.col("stop_num").max().over("run_id"),
    )
    # where this stop sits in the run
    out = out.with_columns(stop_frac=pl.col("stop_num") / pl.col("n_stops"))

    # calendar features. Hour and minute are taken from the planned time of each
    # event, since the arrival model and the departure model use different ones.
    # They are missing where the event does not exist (first and last stop). They are
    # floats, as the pandas version of this script wrote them.
    out = out.with_columns(
        arr_hour=pl.col("planned_arr").dt.hour().cast(pl.Float64),
        arr_minute=pl.col("planned_arr").dt.minute().cast(pl.Float64),
        dep_hour=pl.col("planned_dep").dt.hour().cast(pl.Float64),
        dep_minute=pl.col("planned_dep").dt.minute().cast(pl.Float64),
        # weekday from the planned time (departure, or arrival at the last stop)
        weekday=pl.coalesce("planned_dep", "planned_arr")
        .dt.weekday()
        .cast(pl.UInt8),  # 1 is Monday
        # the split is decided by the day the run starts
        split=pl.when(pl.col("run_day") < VALIDATION_START)
        .then(pl.lit("context"))
        .when(pl.col("run_day") < TEST_START)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("test")),
    )

    return out.select(
        "run_id",
        "run_day",
        "split",
        "train_type",
        "train_key",
        "line_number",
        "final_destination",
        "station",
        "stop_num",
        "planned_arr",
        "planned_dep",
        "arr_canceled",
        "dep_canceled",
        "arr_delay",
        "dep_delay",
        "prev_station",
        "dwell_planned_min",
        "n_stops",
        "run_planned_min",
        "stop_frac",
        "arr_hour",
        "arr_minute",
        "dep_hour",
        "dep_minute",
        "weekday",
    )


def main() -> None:
    n_raw = sum(
        pl.scan_parquet(RAW.format(month=m)).select(pl.len()).collect().item()
        for m in MONTHS
    )
    df = build()

    assert not df.select(
        pl.struct("run_id", "stop_num").is_duplicated().any()
    ).item(), "(run_id, stop_num) repeats"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT, compression="zstd")
    print(f"{n_raw:,} raw rows -> {len(df):,} rows, {df['run_id'].n_unique():,} runs")
    print(df["split"].value_counts().sort("split"))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
