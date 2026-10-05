"""Check the "safe" transfers of dbdelay.risk against what actually happened.

Uses the router requests of one week and the arrival predictions of each method (TabPFN
from its cache, see dbdelay.eval.run_requests; the baselines are computed). The actual
outcome comes from the router: "held" if the connecting train left after the incoming
train arrived, "MISSED" if not (or if one of them was canceled). Transfers with an
unknown outcome are left out.

A good method marks most transfers safe, and almost all of its safe transfers held (the
rule aims at about 95%, since it uses the q95 delay).

Run from the repo root:
uv run --no-sync python -m dbdelay.eval.transfers validation [--methods ...]
"""

import argparse

import polars as pl

from dbdelay import risk
from dbdelay.eval import paths
from dbdelay.eval.run_requests import predict

STOPS = "data/processed/stops.parquet"
METHODS = ["tabpfn_14d_5k_last_known", "train_station", "carry_forward", "global"]


def scored_transfers(split: str, method: str) -> pl.DataFrame:
    """Every transfer with a known outcome, marked safe or at risk by `method`."""
    stops = pl.scan_parquet(STOPS)
    legs = pl.read_parquet(paths.router_legs(split))
    rows = pl.read_parquet(paths.request_rows(split)).filter(pl.col("model") == "arr")
    pred = predict(method, stops, rows, split, ["arr"], local=False, workers=4)
    pred = rows.select("request_id", "run_id", "stop_num").hstack(
        pred.select("q50", "q80", "q95")
    )
    return (
        risk.transfers(risk.with_arrival_delays(legs, pred))
        .join(
            legs.select(
                *risk.ROUTE,
                (pl.col("leg") - 1).alias("leg"),
                pl.col("transfer_status").alias("outcome"),
            ),
            on=[*risk.ROUTE, "leg"],
        )
        .filter(pl.col("outcome").is_in(["held", "MISSED"]))
        .with_columns(method=pl.lit(method), held=pl.col("outcome") == "held")
    )


def summary(scored: pl.DataFrame) -> pl.DataFrame:
    """Per method: transfers, share marked safe, how often safe and at-risk transfers
    held."""
    return (
        scored.group_by("method")
        .agg(
            pl.len().alias("transfers"),
            pl.col("held").mean().alias("held_overall"),
            pl.col("safe").mean().alias("share_safe"),
            pl.col("held").filter(pl.col("safe")).mean().alias("held_if_safe"),
            pl.col("held").filter(~pl.col("safe")).mean().alias("held_if_at_risk"),
            pl.col("safe").is_null().sum().alias("no_prediction"),
        )
        .sort("method")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["validation", "test"])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    args = parser.parse_args()
    scored = pl.concat(scored_transfers(args.split, m) for m in args.methods)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    print(summary(scored))


if __name__ == "__main__":
    main()
