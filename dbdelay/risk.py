"""Route risk from predicted arrival delays (docs/plan.md, component 4).

Each transfer gets one of four levels. The question is whether the transfer still leaves
CHANGE_MIN minutes to change trains when the incoming train is as late as its predicted
q95, q80 or q50 delay, with the connecting train on time:

1. very likely: still works at q95 (over 90% of such transfers held in the validation
   and test weeks)
2. likely: works at q80, not at q95 (80 to 90%)
3. uncertain: works at q50, not at q80 (50 to 80%)
4. unlikely: fails even at q50 (below 50%)

The connecting train is taken as on time because trains almost never leave early (0.14%
of departures in the validation week), so a late departure only helps.

Routes are ranked by their weakest transfer, then by how many transfers have that level
(very likely transfers are not counted), then by the arrival time they reach with 80%
certainty (planned arrival plus the q80 arrival delay of the last leg).
"""

import polars as pl

CHANGE_MIN = 2  # minutes needed to change trains, also used for the actual outcome
LEVELS = {
    1: ("very likely", "over 90%"),
    2: ("likely", "80 to 90%"),
    3: ("uncertain", "50 to 80%"),
    4: ("unlikely", "below 50%"),
}
ROUTE = ["request_id", "route"]


def with_arrival_delays(legs: pl.DataFrame, pred: pl.DataFrame) -> pl.DataFrame:
    """`legs` (one row per leg, see dbdelay.eval.requests.legs_frame) with the predicted
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
    `room_min` (planned transfer time minus CHANGE_MIN) and `level` (1 to 4, see LEVELS;
    empty without a prediction). Needs the columns of `with_arrival_delays`."""
    nxt = legs.select(
        *ROUTE,
        (pl.col("leg") - 1).alias("leg"),
        pl.col("dep").alias("next_dep"),
        pl.col("from_station").alias("station"),
        pl.col("run_id").alias("next_run_id"),
        pl.col("from_stop_num").alias("next_stop_num"),
    )
    room = (pl.col("next_dep") - pl.col("arr")).dt.total_minutes() - CHANGE_MIN
    level = (
        pl.when(pl.col("q95").is_null())
        .then(None)
        .when(pl.col("room_min") >= pl.col("q95"))
        .then(1)
        .when(pl.col("room_min") >= pl.col("q80"))
        .then(2)
        .when(pl.col("room_min") >= pl.col("q50"))
        .then(3)
        .otherwise(4)
    )
    return (
        legs.join(nxt, on=[*ROUTE, "leg"])
        .with_columns(room_min=room)
        .with_columns(level=level.cast(pl.Int8))
    )


def routes(legs: pl.DataFrame) -> pl.DataFrame:
    """One row per route: number of transfers, the weakest transfer level (1 for a direct
    route) and how many transfers have it, the planned arrival and the arrival reached with
    50%, 80% and 95% certainty, ranked within each request (`rank` 1 is the best). Needs
    the columns of `with_arrival_delays`."""
    weakest = (
        transfers(legs)
        .group_by(ROUTE)
        .agg(
            pl.len().alias("transfers"),
            pl.col("level").max().alias("weakest"),
            (pl.col("level") == pl.col("level").max()).sum().alias("at_weakest"),
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
        arrival.join(weakest, on=ROUTE, how="left")
        .with_columns(
            pl.col("transfers", "at_weakest").fill_null(0),
            pl.col("weakest").fill_null(1),
        )
        # a very likely transfer does not count against a route
        .with_columns(
            at_weakest=pl.when(pl.col("weakest") == 1)
            .then(0)
            .otherwise(pl.col("at_weakest"))
        )
        .sort("request_id", "weakest", "at_weakest", "arrival_q80")
        .with_columns(rank=pl.int_range(1, pl.len() + 1).over("request_id"))
    )


def held(
    arr: pl.Expr,
    arr_delay: pl.Expr,
    arr_canceled: pl.Expr,
    dep: pl.Expr,
    dep_delay: pl.Expr,
    dep_canceled: pl.Expr,
) -> pl.Expr:
    """Whether a transfer actually worked: the connecting train left at least CHANGE_MIN
    minutes after the incoming train arrived. A canceled arrival or departure is a missed
    transfer; an unknown delay gives an empty result."""
    gap = (
        dep + pl.duration(minutes=dep_delay) - arr - pl.duration(minutes=arr_delay)
    ).dt.total_minutes()
    return pl.when(arr_canceled | dep_canceled).then(False).otherwise(gap >= CHANGE_MIN)
