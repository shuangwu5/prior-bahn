"""Fit TabPFN on a shared context and predict delay quantiles for the query rows."""

import os
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv

from delaymodel.features import EVENTS, shared_categories, to_frame, usable_rows

VERSION = "v3.5"
QUANTILES = [0.5, 0.8, 0.95]
ROOT = Path(__file__).resolve().parents[1]


def load_token() -> None:
    """Read the Prior Labs key from .env. Both packages read it from TABPFN_TOKEN."""
    load_dotenv(ROOT / ".env", override=True)
    os.environ["TABPFN_TOKEN"] = os.environ["PRIORLABS_API_KEY"]


def regressor(local: bool = False):
    if local:
        from tabpfn import TabPFNRegressor
    else:
        from tabpfn_client import TabPFNRegressor
    return TabPFNRegressor.create_default_for_version(VERSION)


def predict_delays(
    context: pl.DataFrame,
    query: pl.DataFrame,
    model: str,
    quantiles: list[float] = QUANTILES,
    local: bool = False,
) -> pl.DataFrame:
    """Quantiles of the `model` delay ("arr" or "dep") for every query row.

    Query rows keep their order. Rows where the model's event does not exist (the arrival
    at a first stop, the departure at a last stop) have no input, and get null quantiles.
    """
    has_event = query[EVENTS[model]["numeric"][0]].is_not_null()
    rows = query.filter(has_event)
    train = usable_rows(context, model)

    categories = shared_categories(train, rows)
    reg = regressor(local)
    reg.fit(
        to_frame(train, model, categories),
        train[EVENTS[model]["target"]].to_numpy().astype(float),
    )
    # the client needs plain Python floats for the quantile levels
    out = reg.predict(
        to_frame(rows, model, categories),
        output_type="quantiles",
        quantiles=[float(q) for q in quantiles],
    )
    # put the predictions back at the rows that have an event, null elsewhere
    mask = has_event.to_numpy()
    cols = {}
    for q, p in zip(quantiles, out):
        full = np.full(len(query), np.nan)
        full[mask] = np.asarray(p)
        cols[f"q{round(q * 100)}"] = pl.Series(full).fill_nan(None)
    return pl.DataFrame(cols)
