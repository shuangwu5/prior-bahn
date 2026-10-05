"""
Check the transfer levels of priorbahn.risk against what actually happened.

Uses the router requests of one week and the arrival predictions of each method (TabPFN
from its cache, see priorbahn.eval.run_requests; the baselines are computed). A transfer
held if the connecting train left at least risk.CHANGE_MIN minutes after the incoming
train arrived (a cancellation is a miss). Transfers with an unknown delay are left out.

A good method puts many transfers in level 1, and the held rate of each level falls in
its range (risk.LEVELS).

Run from the repo root:
uv run --no-sync python -m priorbahn.eval.transfers validation [--methods ...]
"""

import argparse

import polars as pl

from priorbahn import risk
from priorbahn.eval import paths
from priorbahn.eval.run_requests import predict

STOPS = "data/processed/stops.parquet"
METHODS = ["tabpfn_14d_5k_last_known", "train_station", "carry_forward", "global"]


def with_outcome(transfers: pl.DataFrame, stops: pl.LazyFrame) -> pl.DataFrame:
    """
    `transfers` (see risk.transfers) with `held`, from the actual delays.
    """
    actual = stops.select(
        "run_id",
        "stop_num",
        "arr_delay",
        "dep_delay",
        "arr_canceled",
        "dep_canceled",
    ).collect()
    return (
        transfers.join(
            actual.select(
                "run_id",
                pl.col("stop_num").alias("to_stop_num"),
                "arr_delay",
                "arr_canceled",
            ),
            on=["run_id", "to_stop_num"],
            how="left",
        )
        .join(
            actual.select(
                pl.col("run_id").alias("next_run_id"),
                pl.col("stop_num").alias("next_stop_num"),
                "dep_delay",
                "dep_canceled",
            ),
            on=["next_run_id", "next_stop_num"],
            how="left",
        )
        .with_columns(
            held=risk.held(
                pl.col("arr"),
                pl.col("arr_delay"),
                pl.col("arr_canceled"),
                pl.col("next_dep"),
                pl.col("dep_delay"),
                pl.col("dep_canceled"),
            )
        )
    )


def scored_transfers(split: str, method: str) -> pl.DataFrame:
    """
    Every transfer with a known outcome, with its level by `method`.
    """
    stops = pl.scan_parquet(STOPS)
    legs = pl.read_parquet(paths.router_legs(split))
    rows = pl.read_parquet(paths.request_rows(split)).filter(pl.col("model") == "arr")
    pred = predict(method, stops, rows, split, ["arr"], local=False, workers=4)
    pred = rows.select("request_id", "run_id", "stop_num").hstack(
        pred.select("q50", "q80", "q95")
    )
    transfers = risk.transfers(risk.with_arrival_delays(legs, pred))
    return (
        with_outcome(transfers, stops)
        .filter(pl.col("held").is_not_null())
        .with_columns(method=pl.lit(method))
    )


def summary(scored: pl.DataFrame) -> pl.DataFrame:
    """
    Per method and level: share of transfers and how often they held.
    """
    return (
        scored.group_by("method", "level")
        .agg(pl.len().alias("transfers"), pl.col("held").mean().alias("held"))
        .with_columns(
            share=pl.col("transfers") / pl.col("transfers").sum().over("method")
        )
        .sort("method", "level")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["validation", "test"])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    args = parser.parse_args()
    scored = pl.concat(scored_transfers(args.split, m) for m in args.methods)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)
    print(
        f"{args.split}: {scored.filter(pl.col('method') == args.methods[0]).height} transfers"
    )
    print(summary(scored))


if __name__ == "__main__":
    main()
