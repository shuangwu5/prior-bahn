"""Predict the arrival delays of one request with the chosen setup (docs/plan.md, Decisions).

The setup is the evaluation variant `tabpfn_14d_5k_last_known`: a context of up to 5,000
rows known at "now" (7 days back, 14 days for the same train), the timetable features,
`days_ago` and the last known delay. TabPFN predicts the change from the last known delay.
"""

from datetime import datetime

import polars as pl

from dbdelay.model import last_known
from dbdelay.model.context import build_context
from dbdelay.model.features import with_days_ago
from dbdelay.model.predict import VERSION, predict_delays

SIZE = 5_000
SAME_TRAIN_DAYS = 14


def predict_arrivals(
    stops: pl.LazyFrame,
    keys: pl.DataFrame,
    now: datetime,
    local: bool = False,
    version: str = VERSION,
) -> pl.DataFrame:
    """Arrival delay quantiles (q50, q80, q95) for the stops in `keys` (run_id, stop_num),
    with their `last_known_delay` and `minutes_since_known` (empty: not started)."""
    query = stops.join(
        keys.select("run_id", "stop_num").unique().lazy(), on=["run_id", "stop_num"]
    ).collect()
    context = build_context(
        stops, query, now, size=SIZE, same_train_days=SAME_TRAIN_DAYS
    )
    context = last_known.with_last_known(
        with_days_ago(context, now.date()), stops, now, "arr", replay=True
    )
    query = last_known.with_last_known(
        with_days_ago(query, now.date()), stops, now, "arr", replay=False
    )
    pred = predict_delays(
        context,
        query,
        "arr",
        local=local,
        extra=("days_ago", *last_known.COLUMNS),
        change_from="last_known_delay",
        version=version,
    )
    return query.select(
        "run_id", "stop_num", "last_known_delay", "minutes_since_known"
    ).hstack(pred)
