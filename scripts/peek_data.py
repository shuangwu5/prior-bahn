"""Quick look at one month of the Deutsche Bahn dataset."""

import sys

import polars as pl

path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "data/monthly_processed_data/data-2026-09.parquet"
)
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)
pl.Config.set_tbl_width_chars(250)

lf = pl.scan_parquet(path)
schema = lf.collect_schema()
print("== schema ==")
for name, dtype in schema.items():
    print(f"  {name}: {dtype}")

print("\n== sample rows ==")
print(lf.head(5).collect())

print("\n== size and cardinalities ==")
print(
    lf.select(
        pl.len().alias("rows"),
        pl.col("eva").n_unique().alias("stations"),
        pl.col("train_type").n_unique().alias("train_types"),
        pl.col("train_number").n_unique().alias("train_numbers"),
        pl.col("line_number").n_unique().alias("line_numbers"),
        pl.col("final_destination_station").n_unique().alias("destinations"),
        pl.col("train_line_ride_id").n_unique().alias("rides"),
        pl.col("id").n_unique().alias("ids"),
        pl.col("time").min().alias("time_min"),
        pl.col("time").max().alias("time_max"),
    ).collect()
)

print("\n== null fraction per column ==")
print(lf.select(pl.all().is_null().mean()).collect().transpose(include_header=True))

print("\n== delay distribution (minutes) ==")
d = pl.col("delay_in_min")
print(
    lf.select(
        d.mean().alias("mean"),
        d.std().alias("std"),
        d.min().alias("min"),
        d.quantile(0.5).alias("p50"),
        d.quantile(0.75).alias("p75"),
        d.quantile(0.9).alias("p90"),
        d.quantile(0.99).alias("p99"),
        d.max().alias("max"),
        (d == 0).mean().alias("frac_zero"),
        (d < 0).mean().alias("frac_negative"),
        (d >= 6).mean().alias("frac_ge6"),
        (d >= 15).mean().alias("frac_ge15"),
        (d >= 60).mean().alias("frac_ge60"),
        pl.col("arrival_is_canceled").mean().alias("arr_canceled"),
        pl.col("departure_is_canceled").mean().alias("dep_canceled"),
    )
    .collect()
    .transpose(include_header=True)
)

print("\n== by train type (top 20 by rows) ==")
print(
    lf.group_by("train_type")
    .agg(
        pl.len().alias("rows"),
        d.mean().alias("mean_delay"),
        d.quantile(0.9).alias("p90"),
        (d >= 6).mean().alias("frac_ge6"),
        pl.col("arrival_is_canceled").mean().alias("arr_canceled"),
        pl.col("line_number").is_null().mean().alias("null_line"),
    )
    .sort("rows", descending=True)
    .head(20)
    .collect()
)

print("\n== by hour of planned departure ==")
print(
    lf.with_columns(
        pl.coalesce("departure_planned_time", "arrival_planned_time")
        .dt.hour()
        .alias("hour")
    )
    .group_by("hour")
    .agg(
        pl.len().alias("rows"),
        d.mean().alias("mean_delay"),
        (d >= 6).mean().alias("frac_ge6"),
    )
    .sort("hour")
    .collect()
)

print("\n== by day ==")
print(
    lf.with_columns(pl.col("time").dt.date().alias("day"))
    .group_by("day")
    .agg(
        pl.len().alias("rows"),
        d.mean().alias("mean_delay"),
        (d >= 6).mean().alias("frac_ge6"),
    )
    .sort("day")
    .collect()
)

print("\n== stops per ride ==")
per_ride = lf.group_by("train_line_ride_id").agg(
    pl.len().alias("stops"),
    pl.col("train_line_station_num").min().alias("first_num"),
    pl.col("train_line_station_num").max().alias("last_num"),
)
print(
    per_ride.select(
        pl.col("stops").mean().alias("mean_stops"),
        pl.col("stops").quantile(0.5).alias("p50_stops"),
        pl.col("stops").max().alias("max_stops"),
        (pl.col("stops") == 1).mean().alias("frac_single_stop"),
        (pl.col("first_num") == 1).mean().alias("frac_starts_at_1"),
    ).collect()
)

print("\n== one long-distance ride, in stop order ==")
ride = (
    lf.filter(pl.col("train_type") == "ICE")
    .group_by("train_line_ride_id")
    .agg(pl.len().alias("n"))
    .sort("n", descending=True)
    .head(1)
    .collect()["train_line_ride_id"][0]
)
print(
    lf.filter(pl.col("train_line_ride_id") == ride)
    .sort("train_line_station_num")
    .select(
        "train_line_station_num",
        "station_name",
        "train_type",
        "train_number",
        "arrival_planned_time",
        "arrival_change_time",
        "departure_planned_time",
        "departure_change_time",
        "delay_in_min",
        "arrival_is_canceled",
    )
    .collect()
)

print("\n== signal check: how much does the previous stop's delay explain? ==")
seq = (
    lf.filter(~pl.col("arrival_is_canceled").fill_null(False))
    .sort("train_line_ride_id", "train_line_station_num")
    .with_columns(
        d.shift(1).over("train_line_ride_id").alias("prev_delay"),
        d.shift(3).over("train_line_ride_id").alias("prev3_delay"),
    )
)
print(
    seq.select(
        pl.corr("delay_in_min", "prev_delay").alias("corr_prev_stop"),
        pl.corr("delay_in_min", "prev3_delay").alias("corr_3_stops_back"),
        (d - pl.col("prev_delay")).abs().mean().alias("mae_persistence_1"),
        (d - pl.col("prev3_delay")).abs().mean().alias("mae_persistence_3"),
        (d - d.median()).abs().mean().alias("mae_global_median"),
    ).collect()
)
