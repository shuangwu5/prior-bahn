"""Route risk from predicted arrival delays (docs/plan.md, component 4).

A transfer is "safe" if it still works when the incoming train arrives at its q95 delay
and the connecting train leaves on time: planned departure minus (planned arrival plus
q95 arrival delay) is at least MIN_TRANSFER_MIN minutes. The connecting train is taken as
on time because trains almost never leave early (0.14% of departures in the validation
week), so a late departure only helps. Otherwise the transfer is "at risk".

Routes are ranked by their number of transfers at risk, then by the arrival time they
reach with 80% certainty (planned arrival plus the q80 arrival delay of the last leg).
"""

import polars as pl

from dbdelay.router.core import MIN_TRANSFER_MIN

ROUTE = ["request_id", "route"]


def with_arrival_delays(legs: pl.DataFrame, pred: pl.DataFrame) -> pl.DataFrame:
    """`legs` (one row per leg, see dbdelay.eval.requests.route_request) with the predicted
    arrival delay at each leg's alighting stop. `pred` has request_id, run_id, stop_num and
    q50, q80, q95."""
    return legs.join(
        pred.select(
            "request_id",
            "run_id",
            pl.col("stop_num").alias("to_stop_num"),
            "q50",
            "q80",
            "q95",
        ),
        on=["request_id", "run_id", "to_stop_num"],
        how="left",
    )


def transfers(legs: pl.DataFrame) -> pl.DataFrame:
    """One row per transfer, between leg `leg` and the next leg of the same route, with
    `careful_gap_min` (the transfer time when the incoming train is at its q95 delay)
    and `safe`. Needs the columns of `with_arrival_delays`."""
    nxt = legs.select(
        *ROUTE,
        (pl.col("leg") - 1).alias("leg"),
        pl.col("dep").alias("next_dep"),
        pl.col("from_station").alias("station"),
    )
    gap = (
        pl.col("next_dep") - pl.col("arr") - pl.duration(minutes=pl.col("q95"))
    ).dt.total_seconds() / 60
    return (
        legs.join(nxt, on=[*ROUTE, "leg"])
        .with_columns(careful_gap_min=gap)
        .with_columns(safe=pl.col("careful_gap_min") >= MIN_TRANSFER_MIN)
    )


def routes(legs: pl.DataFrame) -> pl.DataFrame:
    """One row per route: number of transfers, how many are at risk, the planned arrival
    and the arrival reached with 50%, 80% and 95% certainty, ranked within each request
    (`rank` 1 is the best). Needs the columns of `with_arrival_delays`."""
    at_risk = (
        transfers(legs)
        .group_by(ROUTE)
        .agg(
            pl.len().alias("transfers"),
            (~pl.col("safe")).sum().alias("at_risk"),
            pl.col("safe").is_null().sum().alias("unknown"),
        )
    )
    last = legs.sort("leg").group_by(ROUTE).last()
    arrival = last.select(
        *ROUTE,
        pl.col("arr").alias("planned_arrival"),
        *(
            (pl.col("arr") + pl.duration(minutes=pl.col(q).ceil())).alias(
                f"arrival_{q}"
            )
            for q in ("q50", "q80", "q95")
        ),
    )
    return (
        arrival.join(at_risk, on=ROUTE, how="left")
        .with_columns(
            pl.col("transfers", "at_risk", "unknown").fill_null(0),
        )
        .sort("request_id", "at_risk", "arrival_q80")
        .with_columns(rank=pl.int_range(1, pl.len() + 1).over("request_id"))
    )
