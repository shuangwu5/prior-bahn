"""
Route risk from predicted delays.

Each transfer gets a chance that it holds, if both trains run (cancellations are left
out). The chance is read from the incoming train's predicted arrival delay (q50, q80,
q95) at `wait_min`: the planned transfer time minus CHANGE_MIN, plus the connecting
train's typical (median) departure delay. The connecting train's delay comes from the
`carry_forward` baseline, so a search still needs only one TabPFN call.

The four levels are cut points on the chance (LEVELS and CUTS). In the validation and
test weeks, the transfers of each level held about as often as its range says.

Routes are ranked by their planned arrival, then by their number of transfers, so the
order matches the times on the cards. The predicted delays and the weakest transfer level
are shown next to each route, but do not change the order.
"""

import math

import polars as pl

CHANGE_MIN = 2  # minutes needed to change trains, also used for the actual outcome
# level: (label, range of the chance)
LEVELS = {
    1: ("almost sure", "97% or more"),
    2: ("likely", "80 to 97%"),
    3: ("uncertain", "50 to 80%"),
    4: ("unlikely", "below 50%"),
}
# the lowest chance of each level, chosen so that each color means what people expect:
# green is only for transfers that almost never fail, yellow starts where about 1 in 5
# fail
CUTS = {1: 0.97, 2: 0.8, 3: 0.5}
ROUTE = ["request_id", "route"]


def with_arrival_delays(legs: pl.DataFrame, pred: pl.DataFrame) -> pl.DataFrame:
    """
    `legs` (one row per leg, see priorbahn.eval.requests.legs_frame) with the predicted
    arrival delay at each leg's alighting stop. `pred` has request_id, run_id, stop_num and
    q50, q80, q95.
    """
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


def with_departure_delays(legs: pl.DataFrame, pred: pl.DataFrame) -> pl.DataFrame:
    """
    `legs` with `dep_q50`, the typical departure delay at each leg's boarding stop.
    `pred` has request_id, run_id, stop_num and q50.
    """
    return legs.join(
        pred.select(
            "request_id",
            "run_id",
            pl.col("stop_num").alias("from_stop_num"),
            pl.col("q50").alias("dep_q50"),
        ),
        on=["request_id", "run_id", "from_stop_num"],
        how="left",
    )


def probability(x: pl.Expr) -> pl.Expr:
    """
    Share of the predicted arrival delay at or below `x` minutes, from q50, q80 and q95:
    straight lines between them, and exponential tails below q50 and above q95 (with
    the slopes of the parts next to them). Gaps under half a minute are widened.
    """
    q50 = pl.col("q50")
    q80 = pl.max_horizontal(pl.col("q80"), q50 + 0.5)
    q95 = pl.max_horizontal(pl.col("q95"), q80 + 0.5)
    return (
        pl.when(x < q50)
        .then(0.5 * ((x - q50) / ((q80 - q50) / math.log(2.5))).exp())
        .when(x < q80)
        .then(0.5 + 0.3 * (x - q50) / (q80 - q50))
        .when(x < q95)
        .then(0.8 + 0.15 * (x - q80) / (q95 - q80))
        .otherwise(1 - 0.05 * (-(x - q95) / ((q95 - q80) / math.log(4))).exp())
    )


def level(p: pl.Expr) -> pl.Expr:
    """
    The level (1 to 4, see LEVELS) of a chance; empty without a chance.
    """
    out = pl.when(p.is_null()).then(None)
    for k, cut in CUTS.items():
        out = out.when(p >= cut).then(k)
    return out.otherwise(4).cast(pl.Int8)


def transfers(legs: pl.DataFrame) -> pl.DataFrame:
    """
    One row per transfer, between leg `leg` and the next leg of the same route, with
    `room_min` (planned transfer time minus CHANGE_MIN), `wait_min` (the room plus the
    connecting train's typical departure delay), `probability` and `level` (1 to 4, see
    LEVELS; both empty without a prediction). Needs the columns of
    `with_arrival_delays`, and `dep_q50` of `with_departure_delays` (taken as 0 where
    missing).
    """
    if "dep_q50" not in legs.columns:
        legs = legs.with_columns(dep_q50=pl.lit(None, pl.Float64))
    nxt = legs.select(
        *ROUTE,
        (pl.col("leg") - 1).alias("leg"),
        pl.col("dep").alias("next_dep"),
        pl.col("from_station").alias("station"),
        pl.col("run_id").alias("next_run_id"),
        pl.col("from_stop_num").alias("next_stop_num"),
        pl.col("dep_q50").alias("next_dep_q50"),
    )
    room = (pl.col("next_dep") - pl.col("arr")).dt.total_minutes() - CHANGE_MIN
    # trains almost never leave early (0.14% of departures), so a negative median is 0
    wait = pl.col("room_min") + pl.col("next_dep_q50").fill_null(0).clip(lower_bound=0)
    return (
        legs.join(nxt, on=[*ROUTE, "leg"])
        .with_columns(room_min=room)
        .with_columns(wait_min=wait)
        .with_columns(probability=probability(pl.col("wait_min")))
        .with_columns(level=level(pl.col("probability")))
    )


def routes(legs: pl.DataFrame) -> pl.DataFrame:
    """
    One row per route: number of transfers, the weakest transfer level (1 for a direct
    route), the planned arrival and the arrival reached with 50%, 80% and 95% certainty,
    ranked within each request (`rank` 1 is the best). Needs the columns of
    `transfers`.
    """
    weakest = (
        transfers(legs)
        .group_by(ROUTE)
        .agg(
            pl.len().alias("transfers"),
            pl.col("level").max().alias("weakest"),
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
            pl.col("transfers").fill_null(0),
            pl.col("weakest").fill_null(1),
        )
        .sort("request_id", "planned_arrival", "transfers")
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
    """
    Whether a transfer actually worked: the connecting train left at least CHANGE_MIN
    minutes after the incoming train arrived. A canceled arrival or departure is a missed
    transfer; an unknown delay gives an empty result.
    """
    gap = (
        dep + pl.duration(minutes=dep_delay) - arr - pl.duration(minutes=arr_delay)
    ).dt.total_minutes()
    return pl.when(arr_canceled | dep_canceled).then(False).otherwise(gap >= CHANGE_MIN)
