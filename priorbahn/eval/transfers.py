"""
Check the transfer probabilities of priorbahn.risk against what actually happened.

Uses the router requests of one week and the arrival predictions of each method (TabPFN
from its cache, see priorbahn.eval.run_requests; XGBoost from its saved scores, since
predicting again would train its daily models again; the other baselines are computed).
The connecting train's departure delay always comes from `carry_forward`, as in the app. A
transfer held if the connecting train left at least risk.CHANGE_MIN minutes after the
incoming train arrived. Transfers with a canceled train or an unknown delay are left out,
since the probability is for trains that run.

Scores per method: the Brier score (mean squared difference between the probability and the
outcome, 1 for held and 0 for missed; lower is better) and its difference to the first
method with a 95% range, and per level the share of transfers, the mean probability and how
often they held.

Run from the repo root:
uv run --no-sync python -m priorbahn.eval.transfers validation|test [--methods ...]
"""

import argparse

import numpy as np
import polars as pl

from priorbahn import risk
from priorbahn.eval import baselines, paths
from priorbahn.eval.run_requests import QCOLS, predict

STOPS = "data/processed/stops.parquet"
METHODS = [
    "tabpfn_14d_5k_last_known",
    "xgboost",
    "carry_forward",
    "train_station",
    "global",
]


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


def arrival_quantiles(method: str, stops: pl.LazyFrame, arr: pl.DataFrame, split: str) -> pl.DataFrame:
    """
    q50, q80 and q95 of `method` for the arrival rows `arr`, with request_id, run_id and
    stop_num. XGBoost's come from its saved scores (all arrivals with a known delay).
    """
    keys = arr.select("request_id", "run_id", "stop_num")
    if method == "xgboost":
        saved = pl.read_parquet(paths.scores(split, "requests", method))
        return keys.join(
            saved.filter(pl.col("model") == "arr").select(*keys.columns, *QCOLS),
            on=keys.columns,
            how="left",
            maintain_order="left",
        )
    pred = predict(method, stops, arr, split, ["arr"], local=False, workers=4)
    return keys.hstack(pred.select(QCOLS))


def scored_transfers(split: str, method: str) -> pl.DataFrame:
    """
    Every transfer of both running trains with a known outcome, with its probability and
    level by `method`.
    """
    stops = pl.scan_parquet(STOPS)
    legs = pl.read_parquet(paths.router_legs(split))
    rows = pl.read_parquet(paths.request_rows(split))
    arr = rows.filter(pl.col("model") == "arr")
    pred = arrival_quantiles(method, stops, arr, split)
    dep = rows.filter(pl.col("model") == "dep")
    dep_pred = dep.select("request_id", "run_id", "stop_num").hstack(
        baselines.carry_forward(stops, dep, "dep").select("q50")
    )
    legs = risk.with_departure_delays(risk.with_arrival_delays(legs, pred), dep_pred)
    return (
        with_outcome(risk.transfers(legs), stops)
        .filter(
            pl.col("held").is_not_null(),
            ~pl.col("arr_canceled"),
            ~pl.col("dep_canceled"),
        )
        .with_columns(method=pl.lit(method))
    )


def summary(scored: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Per method: the Brier score. Per method and level: share of transfers, mean probability
    and how often they held.
    """
    brier = (
        scored.group_by("method")
        .agg(
            pl.len().alias("transfers"),
            ((pl.col("probability") - pl.col("held").cast(pl.Float64)) ** 2).mean().alias("brier"),
        )
        .sort("brier")
    )
    levels = (
        scored.group_by("method", "level")
        .agg(
            pl.len().alias("transfers"),
            pl.col("probability").mean(),
            pl.col("held").mean(),
        )
        .with_columns(share=pl.col("transfers") / pl.col("transfers").sum().over("method"))
        .sort("method", "level")
    )
    return brier, levels


def differences(scored: pl.DataFrame, reference: str, n: int = 2000) -> pl.DataFrame:
    """
    Brier score of each method minus that of `reference`, on the same transfers. The 95%
    range comes from drawing the requests again with replacement `n` times: a range that
    includes 0 means no clear difference.
    """
    wide = scored.with_columns(b=(pl.col("probability") - pl.col("held").cast(pl.Float64)) ** 2).pivot(
        on="method", index=["request_id", "route", "leg"], values="b"
    )
    methods = [m for m in wide.columns[3:] if m != reference]
    per_request = wide.group_by("request_id").agg(pl.len(), *(pl.col(m) - pl.col(reference) for m in methods))
    counts = per_request["len"].to_numpy()
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(per_request), (n, len(per_request)))
    out = []
    for m in methods:
        diff = per_request[m].list.sum().to_numpy()
        means = diff[draws].sum(axis=1) / counts[draws].sum(axis=1)
        low, high = np.percentile(means, [2.5, 97.5])
        out.append((m, diff.sum() / counts.sum(), low, high))
    return pl.DataFrame(out, schema=["method", f"minus_{reference}", "low", "high"], orient="row")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["validation", "test"])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    args = parser.parse_args()
    scored = pl.concat(scored_transfers(args.split, m) for m in args.methods)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)
    print(f"{args.split}: {scored['request_id'].n_unique()} requests")
    for table in summary(scored):
        print(table)
    print(differences(scored, args.methods[0]))


if __name__ == "__main__":
    main()
