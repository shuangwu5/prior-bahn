"""Build the transformed `stops` table from the raw monthly files.

Follows docs/data-prep-plan.md: schedule-only columns, no lagged history features.
Run from the repo root: uv run --no-sync python -m dbdelay.data.prep
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

RAW = "data/monthly_processed_data/data-2026-{month}.parquet"
MONTHS = ["08", "09"]
OUT = Path("data/processed/stops.parquet")
NON_TRAIN = "^(?:bus|sev|bsv)$"  # matched without regard to case

# split by the day a run starts, see docs/plan.md
FIRST_DAY, LAST_DAY = pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-30")
VALIDATION_START, TEST_START = pd.Timestamp("2026-09-17"), pd.Timestamp("2026-09-24")

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


def minutes_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Whole minutes from `earlier` to `later`. Missing times give <NA>."""
    return ((later - earlier).dt.total_seconds() / 60).astype("Int16")


def load_month(month: str) -> pd.DataFrame:
    """Read one raw file, drop rows we never use and add run_id and run_day."""
    df = pd.read_parquet(RAW.format(month=month), columns=RAW_COLUMNS)

    # rows without a train type are dropped too (about 37k rows, none has a planned time)
    is_train = df["train_type"].notna() & ~df["train_type"].str.contains(
        NON_TRAIN, case=False, na=False
    )
    has_planned_time = (
        df["arrival_planned_time"].notna() | df["departure_planned_time"].notna()
    )
    keep = (
        is_train
        & ~df["is_replacement_train"]
        & ~df["is_additional_stop"]
        & has_planned_time
    )
    df = df[keep]

    # `id` is <ride id>-<run start YYMMDDHHMM>-<stop number>.
    # run_id is everything before the stop number, run_day is the date inside it.
    parts = df["id"].str.extract(r"^(.*-(\d{6})\d{4})-\d+$")
    df = df.assign(
        run_id=parts[0],
        run_day=pd.to_datetime(parts[1], format="%y%m%d"),
    )
    return df.drop(columns="id")


def destination_in_station_spelling(df: pd.DataFrame, station: pd.Series) -> pd.Series:
    """Final destination of each row, spelled like `station`.

    The source writes destinations like xml_station_name ("Frankfurt(Main)Süd"), while
    station_name has "Frankfurt (Main) Süd". So each destination is translated through
    the xml_station_name -> station pairs seen in the data. An xml name that belongs to
    two different stations cannot be translated and is left out of the lookup.
    """
    pairs = pd.DataFrame(
        {"xml": df["xml_station_name"], "station": station}
    ).drop_duplicates()
    pairs = pairs[~pairs["xml"].duplicated(keep=False)]
    xml_to_station = pairs.set_index("xml")["station"]

    source = df["final_destination_station"]
    destination = source.map(xml_to_station)
    # places that are not stops in our data (Basel SBB, Enschede, bus stops) keep their
    # source spelling, and the source leaves the destination empty at the last stop of a
    # run, where it is the station itself
    return destination.fillna(source).fillna(station)


def build() -> pd.DataFrame:
    # both months are loaded together: a run that starts on Aug 31 spills into September
    df = pd.concat([load_month(m) for m in MONTHS], ignore_index=True)
    df = df[df["run_day"].between(FIRST_DAY, LAST_DAY)]

    # stations are merged by name, the few without a name keep their EVA code
    station = df["station_name"].fillna("EVA " + df["eva"])

    out = pd.DataFrame(
        {
            "run_id": df["run_id"],
            "run_day": df["run_day"],
            "train_type": df["train_type"],
            "train_key": df["train_type"] + " " + df["train_number"],
            "line_number": df["line_number"],
            "final_destination": destination_in_station_spelling(df, station),
            "station": station,
            "stop_num": df["train_line_station_num"],
            "planned_arr": df["arrival_planned_time"],
            "planned_dep": df["departure_planned_time"],
            "arr_canceled": df["arrival_is_canceled"],
            "dep_canceled": df["departure_is_canceled"],
            # delay_in_min is the delay of the row's own event (see the exploration
            # notebook), so both delays are recomputed from the time columns.
            # A canceled event has no usable delay, so it becomes <NA>.
            "arr_delay": minutes_between(
                df["arrival_change_time"], df["arrival_planned_time"]
            ).where(~df["arrival_is_canceled"]),
            "dep_delay": minutes_between(
                df["departure_change_time"], df["departure_planned_time"]
            ).where(~df["departure_is_canceled"]),
        }
    )

    # sort so that the rows of one run are next to each other, in stop order
    out = out.sort_values(["run_day", "run_id", "stop_num"], ignore_index=True)

    # `shift()` moves a column down by one row, so each row can see the row above it.
    # The row above is the previous stop only if it is in the same run and the stop
    # numbers are adjacent (the source has missing stops in about 4% of runs).
    same_run = out["run_id"] == out["run_id"].shift()
    prev_is_adjacent = same_run & (out["stop_num"] - out["stop_num"].shift() == 1)
    out["prev_station"] = out["station"].shift().where(prev_is_adjacent)
    out["run_planned_min"] = minutes_between(
        out["planned_arr"], out["planned_dep"].shift()
    ).where(prev_is_adjacent)

    out["dwell_planned_min"] = minutes_between(out["planned_dep"], out["planned_arr"])

    # highest stop number seen in each run, and where this stop sits in the run
    out["n_stops"] = out.groupby("run_id")["stop_num"].transform("max")
    out["stop_frac"] = out["stop_num"] / out["n_stops"]

    # calendar features. Hour and minute are taken from the planned time of each
    # event, since the arrival model and the departure model use different ones.
    # They are missing where the event does not exist (first and last stop).
    out["arr_hour"] = out["planned_arr"].dt.hour
    out["arr_minute"] = out["planned_arr"].dt.minute
    out["dep_hour"] = out["planned_dep"].dt.hour
    out["dep_minute"] = out["planned_dep"].dt.minute
    # weekday from the planned time (departure, or arrival at the last stop)
    planned_time = out["planned_dep"].fillna(out["planned_arr"])
    out["weekday"] = (planned_time.dt.dayofweek + 1).astype("uint8")  # 1 is Monday

    # the split is decided by the day the run starts
    out["split"] = "test"
    out.loc[out["run_day"] < TEST_START, "split"] = "validation"
    out.loc[out["run_day"] < VALIDATION_START, "split"] = "context"

    return out[
        [
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
        ]
    ]


def main() -> None:
    n_raw = sum(pq.ParquetFile(RAW.format(month=m)).metadata.num_rows for m in MONTHS)
    df = build()

    assert not df.duplicated(["run_id", "stop_num"]).any(), "(run_id, stop_num) repeats"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, compression="zstd", index=False)
    print(f"{n_raw:,} raw rows -> {len(df):,} rows, {df['run_id'].nunique():,} runs")
    print(df["split"].value_counts().sort_index())
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
