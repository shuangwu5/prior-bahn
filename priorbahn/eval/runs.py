"""
Sample whole runs of the validation or test week for the per-leg evaluation.

Every method (TabPFN and the baselines) is scored on the same sampled runs, using all stop
rows of each run. Runs are drawn in equal numbers from a few train groups, so that rare
groups such as long-distance trains get enough runs. Overall scores are weighted back to
each group's real share of rows (see `group_weights`).

Run from the repo root: uv run --no-sync python -m priorbahn.eval.runs validation
"""

import sys

import polars as pl

from priorbahn.eval import paths

STOPS = "data/processed/stops.parquet"
RUNS_PER_GROUP = 60
LONG_DISTANCE = ["ICE", "IC", "EC", "ECE", "RJ", "RJX", "NJ", "EN", "FLX", "TGV", "ES"]


def train_group() -> pl.Expr:
    return (
        pl.when(pl.col("train_type").is_in(LONG_DISTANCE))
        .then(pl.lit("long-distance"))
        .when(pl.col("train_type").is_in(["S", "RE", "RB"]))
        .then(pl.col("train_type"))
        .otherwise(pl.lit("other regional"))
        .alias("group")
    )


def week_runs(stops: pl.LazyFrame, split: str) -> pl.DataFrame:
    """One row per run of the week: its day, group and number of stop rows."""
    return (
        stops.filter(pl.col("split") == split)
        .group_by("run_id")
        .agg(pl.first("run_day"), pl.first("train_type"), pl.len().alias("rows"))
        .with_columns(train_group())
        .collect()
    )


def sample_runs(
    stops: pl.LazyFrame, split: str, per_group: int = RUNS_PER_GROUP, seed: int = 0
) -> pl.DataFrame:
    # sorted so the sample is reproducible
    runs = week_runs(stops, split).sort("run_id")
    return pl.concat(
        g.sample(min(per_group, len(g)), seed=seed)
        for _, g in runs.group_by("group", maintain_order=True)
    ).sort("group", "run_id")


def group_weights(
    stops: pl.LazyFrame, split: str, sample: pl.DataFrame
) -> pl.DataFrame:
    """Weight per sampled row so that each group counts with its share of all week rows."""
    week = week_runs(stops, split).group_by("group").agg(pl.col("rows").sum())
    drawn = sample.group_by("group").agg(pl.col("rows").sum().alias("drawn"))
    return week.join(drawn, on="group").select(
        "group",
        (pl.col("rows") / pl.col("rows").sum() / pl.col("drawn")).alias("weight"),
    )


def query_rows(stops: pl.LazyFrame, sample: pl.DataFrame) -> pl.DataFrame:
    """All stop rows of the sampled runs, with their group."""
    rows = stops.join(sample.select("run_id", "group").lazy(), on="run_id").collect()
    return rows.sort("run_id", "stop_num")


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    stops = pl.scan_parquet(STOPS)
    sample = sample_runs(stops, split)
    weights = group_weights(stops, split, sample)
    rows = query_rows(stops, sample).join(weights, on="group")

    out = paths.sample_runs(split)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(out)
    print(
        rows.group_by("group")
        .agg(
            pl.col("run_id").n_unique().alias("runs"),
            pl.len().alias("rows"),
            pl.col("arr_delay").is_not_null().sum().alias("arr_targets"),
            pl.col("dep_delay").is_not_null().sum().alias("dep_targets"),
            pl.first("weight"),
        )
        .sort("group")
    )


if __name__ == "__main__":
    main()
