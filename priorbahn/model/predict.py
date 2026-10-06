"""
Fit TabPFN on a shared context and predict delay quantiles for the query rows.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from dotenv import load_dotenv

from priorbahn.model.features import EVENTS, shared_categories, to_frame, usable_rows

if TYPE_CHECKING:
    import tabpfn
    import tabpfn_client

VERSION = "v3.5"  # "v3.5-fast" is the faster variant
QUANTILES = [0.5, 0.8, 0.95]
ROOT = Path(__file__).resolve().parents[2]


def load_token() -> None:
    """
    Read the Prior Labs key from .env. Both packages read it from TABPFN_TOKEN.
    Keep TABPFN_TOKEN if it is set, else fall back to PRIORLABS_API_KEY, else "".
    """
    load_dotenv(ROOT / ".env", override=True)
    os.environ.setdefault("TABPFN_TOKEN", os.environ.get("PRIORLABS_API_KEY", ""))


def regressor(local: bool = False, version: str = VERSION) -> "tabpfn.TabPFNRegressor | tabpfn_client.TabPFNRegressor":
    if local:
        from tabpfn import TabPFNRegressor
    else:
        from tabpfn_client import TabPFNRegressor
    return TabPFNRegressor.create_default_for_version(version)


def predict_delays(
    context: pl.DataFrame,
    query: pl.DataFrame,
    model: str,
    quantiles: list[float] = QUANTILES,
    local: bool = False,
    extra: tuple[str, ...] = (),
    change_from: str | None = None,
    version: str = VERSION,
) -> pl.DataFrame:
    """
    Quantiles of the `model` delay ("arr" or "dep") for every query row.

    `extra` names feature columns used on top of the standard list, such as "days_ago"
    (see `features.with_days_ago`); both frames must have them.

    With `change_from` (a column such as "last_known_delay"), TabPFN learns the change in
    delay from that column, and the column is added back to the predicted quantiles. Rows
    where it is empty learn and predict the delay itself.

    Query rows keep their order. Rows where the model's event does not exist (the arrival
    at a first stop, the departure at a last stop) have no input, and get null quantiles.
    """
    has_event = query[EVENTS[model]["numeric"][0]].is_not_null()
    rows = query.filter(has_event)
    if rows.is_empty():  # e.g. a run with one stop in the data has no arrival
        return pl.DataFrame(
            {f"q{round(q * 100)}": [None] * len(query) for q in quantiles},
            schema={f"q{round(q * 100)}": pl.Float64 for q in quantiles},
        )
    train = usable_rows(context, model)

    categories = shared_categories(train, rows)
    target = train[EVENTS[model]["target"]].to_numpy().astype(float)
    if change_from is not None:
        target = target - train[change_from].fill_null(0).to_numpy().astype(float)
    reg = regressor(local, version)
    reg.fit(to_frame(train, model, categories, extra), target)
    # the client needs plain Python floats for the quantile levels
    out = reg.predict(
        to_frame(rows, model, categories, extra),
        output_type="quantiles",
        quantiles=[float(q) for q in quantiles],
    )
    # put the predictions back at the rows that have an event, null elsewhere
    mask = has_event.to_numpy()
    base = rows[change_from].fill_null(0).to_numpy().astype(float) if change_from is not None else 0.0
    cols = {}
    for q, p in zip(quantiles, out):
        full = np.full(len(query), np.nan)
        full[mask] = np.asarray(p) + base
        cols[f"q{round(q * 100)}"] = pl.Series(full).fill_nan(None)
    return pl.DataFrame(cols)
