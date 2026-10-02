"""Second look: rebuild per-day train runs from `id` and check how much signal simple features carry."""

import sys

import polars as pl

path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "data/monthly_processed_data/data-2026-09.parquet"
)
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)
pl.Config.set_tbl_width_chars(250)

d = pl.col("delay_in_min")
lf = pl.scan_parquet(path)

print("== id format ==")
print(lf.select("id", "train_line_ride_id", "train_line_station_num").head(5).collect())

# id looks like "<ride id>-<YYMMDDHHMM of run start>-<station num>"
runs = lf.with_columns(
    pl.col("id").str.extract(r"^(.*)-\d+$", 1).alias("run_id"),
    pl.col("id").str.extract(r"-(\d{10})-\d+$", 1).alias("run_start"),
    pl.coalesce("arrival_planned_time", "departure_planned_time").alias("planned_time"),
).filter(
    ~pl.col("arrival_is_canceled") & ~pl.col("departure_is_canceled") & d.is_not_null()
)

print("\n== runs ==")
per_run = runs.group_by("run_id").agg(
    pl.len().alias("stops"),
    pl.col("train_line_station_num").n_unique().alias("distinct_nums"),
    pl.col("train_line_station_num").max().alias("max_num"),
)
print(
    per_run.select(
        pl.len().alias("runs"),
        pl.col("stops").mean().alias("mean_stops_seen"),
        pl.col("stops").quantile(0.5).alias("p50_stops_seen"),
        pl.col("stops").max().alias("max_stops_seen"),
        (pl.col("stops") == pl.col("distinct_nums"))
        .mean()
        .alias("frac_no_dup_station_num"),
        (pl.col("stops") / pl.col("max_num")).mean().alias("mean_coverage_of_route"),
        pl.col("run_id").is_null().sum().alias("null_run_id"),
    ).collect()
)

print("\n== persistence within a run (previous observed stop) ==")
seq = runs.sort("run_id", "train_line_station_num").with_columns(
    d.shift(1).over("run_id").alias("prev_delay"),
    d.shift(3).over("run_id").alias("prev3_delay"),
    d.first().over("run_id").alias("first_delay"),
    (
        pl.col("train_line_station_num")
        - pl.col("train_line_station_num").shift(1).over("run_id")
    ).alias("gap"),
)
print(
    seq.filter(pl.col("prev_delay").is_not_null())
    .select(
        pl.len().alias("rows"),
        pl.corr("delay_in_min", "prev_delay").alias("corr_prev"),
        pl.corr("delay_in_min", "prev3_delay").alias("corr_prev3"),
        pl.corr("delay_in_min", "first_delay").alias("corr_first"),
        (d - pl.col("prev_delay")).abs().mean().alias("mae_prev"),
        (d - pl.col("prev3_delay")).abs().mean().alias("mae_prev3"),
        (d - pl.col("first_delay")).abs().mean().alias("mae_first"),
        (d - d.median()).abs().mean().alias("mae_median"),
        pl.col("gap").mean().alias("mean_station_gap"),
    )
    .collect()
)

print("\n== no live info: train on days 1-23, test on days 24-30 ==")
df = runs.select(
    "eva",
    "train_type",
    "train_number",
    "line_number",
    "planned_time",
    "delay_in_min",
    "train_line_station_num",
).collect()
train = df.filter(pl.col("planned_time").dt.day() <= 23)
test = df.filter(pl.col("planned_time").dt.day() >= 24)
gmed = train["delay_in_min"].median()
gmean = train["delay_in_min"].mean()


def hist(keys: list[str], name: str) -> pl.DataFrame:
    return train.group_by(keys).agg(
        d.mean().alias(f"{name}_mean"),
        d.median().alias(f"{name}_med"),
        pl.len().alias(f"{name}_n"),
    )


test = (
    test.join(hist(["train_type"], "type"), on=["train_type"], how="left")
    .join(hist(["eva"], "station"), on=["eva"], how="left")
    .join(
        hist(["train_type", "train_number", "eva"], "train_station"),
        on=["train_type", "train_number", "eva"],
        how="left",
    )
)
t = pl.col("delay_in_min")
print(
    test.select(
        pl.len().alias("test_rows"),
        (t - gmed).abs().mean().alias("mae_global_median"),
        (t - pl.col("type_med").fill_null(gmed)).abs().mean().alias("mae_type_median"),
        (t - pl.col("station_med").fill_null(gmed))
        .abs()
        .mean()
        .alias("mae_station_median"),
        (t - pl.col("train_station_med").fill_null(gmed))
        .abs()
        .mean()
        .alias("mae_train_station_median"),
        ((t - gmean) ** 2).mean().sqrt().alias("rmse_global_mean"),
        ((t - pl.col("train_station_mean").fill_null(gmean)) ** 2)
        .mean()
        .sqrt()
        .alias("rmse_train_station_mean"),
        pl.col("train_station_n").is_null().mean().alias("frac_unseen_train_station"),
        pl.corr("delay_in_min", "train_station_mean").alias("corr_train_station_mean"),
    )
)

print("\n== long-distance only (ICE/IC/EC) ==")
ld = runs.filter(pl.col("train_type").is_in(["ICE", "IC", "EC"]))
print(
    ld.select(
        pl.len().alias("rows"),
        pl.col("run_id").n_unique().alias("runs"),
        pl.col("eva").n_unique().alias("stations"),
        pl.col("train_number").n_unique().alias("train_numbers"),
        d.mean().alias("mean_delay"),
        d.quantile(0.5).alias("p50"),
        d.quantile(0.9).alias("p90"),
        (d >= 6).mean().alias("frac_ge6"),
        (d >= 15).mean().alias("frac_ge15"),
    ).collect()
)

print("\n== busiest stations ==")
print(
    runs.group_by("station_name")
    .agg(
        pl.len().alias("rows"),
        d.mean().alias("mean_delay"),
        (d >= 6).mean().alias("frac_ge6"),
    )
    .sort("rows", descending=True)
    .head(15)
    .collect()
)
