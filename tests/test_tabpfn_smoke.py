"""Smoke test for TabPFN-3.5 on the processed stops table.

Fits 50 context rows and predicts 10 validation rows, through the Prior Labs
API (client) and with local weights (local).

    uv run --no-sync pytest tests/test_tabpfn_smoke.py
    uv run --no-sync pytest tests/test_tabpfn_smoke.py -k local -s
"""

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
STOPS = ROOT / "data/processed/stops.parquet"
NUM = [
    "stop_num",
    "n_stops",
    "stop_frac",
    "arr_hour",
    "arr_minute",
    "weekday",
    "dwell_planned_min",
    "run_planned_min",
]
CAT = ["train_type", "station"]
VERSION = "v3.5"
N_TRAIN, N_VAL = 50, 10


def load_sample(split: str, n: int) -> pl.DataFrame:
    df = (
        pl.scan_parquet(STOPS)
        .filter(pl.col("arr_delay").is_not_null() & ~pl.col("arr_canceled"))
        .filter(pl.col("split") == split)
        .select(["arr_delay", *NUM, *CAT])
        .head(200_000)
        .collect()
    )
    return df.sample(n, seed=0)


def to_X(df: pl.DataFrame, cats: dict):
    X = df.select(NUM).to_pandas()
    for c in CAT:
        X[c] = df[c].replace_strict(cats[c], default=-1).to_numpy()
    return X


@pytest.fixture(scope="module")
def token() -> None:
    load_dotenv(ROOT / ".env", override=True)
    if "PRIORLABS_API_KEY" not in os.environ:
        pytest.skip("PRIORLABS_API_KEY not set")
    # both packages read the token from TABPFN_TOKEN
    os.environ["TABPFN_TOKEN"] = os.environ["PRIORLABS_API_KEY"]


@pytest.fixture(scope="module")
def data():
    if not STOPS.exists():
        pytest.skip("run scripts/prep_data.py first")
    train, val = load_sample("context", N_TRAIN), load_sample("validation", N_VAL)
    cats = {c: {v: i for i, v in enumerate(train[c].unique().to_list())} for c in CAT}
    return (
        to_X(train, cats),
        train["arr_delay"].to_numpy().astype(float),
        to_X(val, cats),
        val["arr_delay"].to_numpy().astype(float),
    )


@pytest.mark.parametrize("mode", ["client", "local"])
def test_regressor_smoke(mode, token, data):
    if mode == "client":
        from tabpfn_client import TabPFNRegressor
    else:
        from tabpfn import TabPFNRegressor
    X_tr, y_tr, X_te, y_te = data

    reg = TabPFNRegressor.create_default_for_version(VERSION)
    reg.fit(X_tr, y_tr)
    pred = np.asarray(reg.predict(X_te))

    print(f"\n{mode} {VERSION}")
    print("pred:", np.round(pred, 2))
    print("true:", y_te)
    print("MAE tabpfn:", np.abs(pred - y_te).mean())
    print("MAE mean baseline:", np.abs(y_tr.mean() - y_te).mean())
    assert pred.shape == y_te.shape
    assert np.isfinite(pred).all()
