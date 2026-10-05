"""
Per-leg scores for predicted delay quantiles.

Every method returns the same frame: one row per query row with columns q50, q80, q95
(see `priorbahn.model.predict.QUANTILES`). Scores use only rows with a known target, and
weight each row with its group weight (see `priorbahn.eval.runs.group_weights`).
"""

import polars as pl

from priorbahn.model.features import EVENTS
from priorbahn.model.predict import QUANTILES

QCOLS = {q: f"q{round(q * 100)}" for q in QUANTILES}


def pinball(q: float) -> pl.Expr:
    """Pinball (quantile) loss of the column for quantile `q` against the target `y`."""
    diff = pl.col("y") - pl.col(QCOLS[q])
    return pl.max_horizontal(q * diff, (q - 1) * diff)


def scored_rows(
    rows: pl.DataFrame, pred: pl.DataFrame, model: str, extra: tuple[str, ...] = ()
) -> pl.DataFrame:
    """
    Query rows next to their predictions, kept only where the target is known.

    `extra` names more columns of `rows` to keep, for breakdowns.
    """
    return (
        pl.concat(
            [
                rows.select(
                    "run_id",
                    "train_key",
                    "station",
                    "stop_num",
                    "group",
                    "weight",
                    "dep_hour",
                    "arr_hour",
                    *extra,
                ),
                pred,
            ],
            how="horizontal",
        )
        .with_columns(rows[EVENTS[model]["target"]].cast(pl.Float64).alias("y"))
        .filter(pl.col("y").is_not_null() & pl.col(QCOLS[0.5]).is_not_null())
    )


def score(df: pl.DataFrame, by: str | list[str] | None = None) -> pl.DataFrame:
    """
    Weighted scores, overall or per value of `by`.

    - pinball_qXX: mean pinball loss at that quantile, and pinball the mean over quantiles
    - mae: mean absolute error of the median
    - cover_qXX: share of targets at or below the predicted quantile (ideal: XX%)
    """
    w = pl.col("weight")

    def mean(e: pl.Expr) -> pl.Expr:
        return (e * w).sum() / w.sum()

    aggs = [pl.len().alias("n")]
    aggs += [mean(pinball(q)).alias(f"pinball_{c}") for q, c in QCOLS.items()]
    aggs.append(mean(pl.mean_horizontal(pinball(q) for q in QCOLS)).alias("pinball"))
    aggs.append(mean((pl.col("y") - pl.col(QCOLS[0.5])).abs()).alias("mae"))
    aggs += [
        mean((pl.col("y") <= pl.col(c)).cast(pl.Float64)).alias(f"cover_{c}")
        for c in QCOLS.values()
    ]
    if by is None:
        return df.select(aggs)
    return df.group_by(by).agg(aggs).sort(by)
