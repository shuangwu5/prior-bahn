"""Baseline checks on a small made-up stops table (no API calls).

uv run --no-sync pytest tests/test_baselines.py
"""

from datetime import date, datetime, time, timedelta

import polars as pl

from priorbahn.eval.baselines import global_quantiles

DAY = date(2026, 9, 20)


def made_up_stops(rows: list[tuple]) -> pl.LazyFrame:
    """One arrival per row: (run_id, run_day, planned_arr, arr_delay), all at station "S"."""
    df = pl.DataFrame(
        rows,
        schema={
            "run_id": pl.String,
            "run_day": pl.Date,
            "planned_arr": pl.Datetime("ns"),
            "arr_delay": pl.Int16,
        },
        orient="row",
    )
    return df.with_columns(
        run_day=pl.col("run_day").cast(pl.Datetime("ns")),
        planned_dep=pl.lit(None, pl.Datetime("ns")),
        dep_delay=pl.lit(None, pl.Int16),
        station=pl.lit("S"),
        arr_hour=pl.col("planned_arr").dt.hour().cast(pl.Float64),
        arr_canceled=pl.lit(False),
    ).lazy()


def test_history_leaves_out_events_of_the_query_day():
    day_before = DAY - timedelta(days=1)
    rows = [
        (f"r{i}", day_before, datetime.combine(day_before, time(10, i)), 1)
        for i in range(10)
    ]
    # an earlier-day run that arrives after midnight, on the query day: with it in the
    # history, q95 of these 11 delays would be 50
    rows += [("late", day_before, datetime.combine(DAY, time(0, 30)), 99)]
    stops = made_up_stops(rows)
    query = (
        stops.filter(pl.col("run_id") == "r0")
        .collect()
        .with_columns(run_day=pl.lit(DAY).cast(pl.Datetime("ns")))
    )

    pred = global_quantiles(stops, query, "arr")
    assert pred["q95"].to_list() == [1.0]
