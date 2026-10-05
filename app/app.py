"""
Streamlit app: plan a train journey and see how reliable each route is.

The user picks a day of the data, two stations and "now" (also the earliest departure).
The app finds up to N routes, predicts the arrival delay of every leg with TabPFN from
what was known at "now", gives each transfer a level (priorbahn/risk.py) and ranks the
routes. A switch on the page reveals what actually happened that day. The cards are drawn by
app/render.py.

Run from the repo root: uv run --no-sync streamlit run app/app.py
"""

import sys
import time as clock
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render

from priorbahn import risk
from priorbahn.eval import requests as rq
from priorbahn.model.predict import load_token
from priorbahn.model.request import predict_arrivals
from priorbahn.router import core as rc

STOPS = "data/processed/stops.parquet"
DATA_DAYS = (date(2026, 9, 15), date(2026, 9, 30))  # the context needs 14 earlier days
# more routes cost little: TabPFN's time depends on the context size, not on the number
# of stops predicted. The evaluation used rq.N_ROUTES (3).
MAX_ROUTES = 10
# 5% to 95% in steps of 5%, for the delay curve at each transfer. It includes the 50%,
# 80% and 95% that the transfer levels use, and costs about the same time as those three.
QUANTILES = [round(0.05 * i, 2) for i in range(1, 20)]
MODELS = {"TabPFN 3.5": "v3.5", "TabPFN 3.5 Fast": "v3.5-fast"}
# run TabPFN on this machine; False uses the Prior Labs API (costs credits)
LOCAL = True
STOP_COLUMNS = [
    "run_id",
    "stop_num",
    "station",
    "planned_arr",
    "planned_dep",
    "arr_delay",
    "dep_delay",
    "arr_canceled",
    "dep_canceled",
    "final_destination",
]


@st.cache_resource(show_spinner="Building timetable for the day...")
def get_timetable(day: date) -> rc.Timetable:
    return rc.load_timetable(day)


def short_label(label: str) -> str:
    """
    The line ("S3") if the label has one ("S 38318 (S3)"), else the train ("ECE 8").
    """
    return label[label.index("(") + 1 : -1] if "(" in label else label


@st.cache_data(show_spinner=False)
def plan(
    day: date, src: str, dst: str, now_time: time, version: str
) -> tuple[list[dict], float]:
    """
    The ranked routes (up to MAX_ROUTES), each a dict for render.card, and the seconds
    TabPFN took.
    """
    tt = get_timetable(day)
    now = datetime.combine(day, now_time)
    journeys = rq.find_journeys(tt, src, dst, now, MAX_ROUTES)
    if not journeys:
        return [], 0.0
    legs = pl.DataFrame(rq.legs_frame(journeys, request_id=0))
    stops = pl.scan_parquet(STOPS)
    load_token()
    started = clock.perf_counter()
    pred = predict_arrivals(
        stops,
        legs.select("run_id", pl.col("to_stop_num").alias("stop_num")),
        now,
        local=LOCAL,
        version=version,
        quantiles=QUANTILES,
    ).with_columns(request_id=pl.lit(0, pl.Int64))
    seconds = clock.perf_counter() - started
    print(
        f"{version} {'local' if LOCAL else 'API'}: {src} -> {dst} at {now:%Y-%m-%d %H:%M}, "
        f"{len(legs)} legs, prediction {seconds:.1f}s",
        flush=True,
    )
    legs = risk.with_arrival_delays(legs, pred)
    transfers = risk.transfers(legs)
    summary = risk.routes(legs)

    detail = (
        stops.filter(pl.col("run_id").is_in(legs["run_id"].unique().implode()))
        .select(STOP_COLUMNS)
        .collect()
    )
    q_cols = [f"q{round(q * 100)}" for q in QUANTILES]
    known = pred.select("run_id", "stop_num", "last_known_delay", *q_cols)
    out = []
    for r in summary.iter_rows(named=True):
        journey = journeys[r["route"]]
        route_legs = []
        for i, leg in enumerate(journey):
            leg_stops = detail.filter(
                pl.col("run_id") == leg.run_id,
                pl.col("stop_num").is_between(leg.from_stop_num, leg.to_stop_num),
            ).sort("stop_num")
            row = legs.filter(pl.col("route") == r["route"], pl.col("leg") == i).row(
                0, named=True
            )
            at_end = known.filter(
                pl.col("run_id") == leg.run_id, pl.col("stop_num") == leg.to_stop_num
            )
            end = at_end.row(0, named=True) if len(at_end) else {}
            route_legs.append(
                {
                    "label": leg.label,
                    "short": short_label(leg.label),
                    "destination": leg_stops["final_destination"][0],
                    "minutes": max(int((leg.arr - leg.dep).total_seconds() // 60), 1),
                    "stops": leg_stops.to_dicts(),
                    "q50": row["q50"],
                    "q80": row["q80"],
                    "last_known_delay": end.get("last_known_delay"),
                    # (level, minutes) for the delay curve, empty without a prediction
                    "quantiles": [
                        (q, end[c])
                        for q, c in zip(QUANTILES, q_cols)
                        if end.get(c) is not None
                    ],
                    "transfer_after": None,
                }
            )
        for t in transfers.filter(pl.col("route") == r["route"]).iter_rows(named=True):
            incoming = route_legs[t["leg"]]["stops"][-1]
            outgoing = route_legs[t["leg"] + 1]["stops"][0]
            route_legs[t["leg"]]["transfer_after"] = {
                "level": t["level"],
                "planned_min": int((t["next_dep"] - t["arr"]).total_seconds() // 60),
                "room_min": t["room_min"],
                "held": transfer_held(t["arr"], incoming, t["next_dep"], outgoing),
            }
        out.append(
            {
                **{
                    k: r[k]
                    for k in (
                        "rank",
                        "weakest",
                        "planned_arrival",
                        "arrival_q80",
                        "arrival_q95",
                    )
                },
                "planned_departure": journey[0].dep,
                "legs": route_legs,
            }
        )
    return out, seconds


def transfer_held(
    arr: datetime, incoming: dict, dep: datetime, outgoing: dict
) -> bool | None:
    """
    Whether the transfer actually worked, by the same rule as the evaluation.
    """
    df = pl.DataFrame(
        {
            "arr": [arr],
            "arr_delay": [incoming["arr_delay"]],
            "arr_canceled": [incoming["arr_canceled"]],
            "dep": [dep],
            "dep_delay": [outgoing["dep_delay"]],
            "dep_canceled": [outgoing["dep_canceled"]],
        },
        schema_overrides={"arr_delay": pl.Int16, "dep_delay": pl.Int16},
    )
    return df.select(risk.held(*(pl.col(c) for c in df.columns)))[0, 0]


st.set_page_config(page_title="Prior Bahn", layout="wide")
# the form and the route cards share one centered column, 80% of the page wide
_, main, _ = st.columns([1, 8, 1])
with main:
    st.title("Prior Bahn", anchor=False)
    st.caption(
        f"Train-only routes on the timetable from {DATA_DAYS[0]:%-d} to "
        f"{DATA_DAYS[1]:%-d %B %Y}, ranked by earlier arrival, then fewer transfers. "
        "Delays are predicted with TabPFN from what was known at the chosen time."
    )

    with st.form("query"):
        col_from, col_to = st.columns(2)
        col_day, col_time = st.columns(2)
        day = col_day.date_input(
            "Date",
            value=date(2026, 9, 24),
            min_value=DATA_DAYS[0],
            max_value=DATA_DAYS[1],
        )
        stations = get_timetable(day).stations
        src_name = col_from.selectbox(
            "From", stations, index=stations.index("Freiburg (Breisgau) Hbf")
        )
        dst_name = col_to.selectbox(
            "To", stations, index=stations.index("Berlin Hauptbahnhof")
        )
        now_time = col_time.time_input(
            "Now (earliest departure)", value=time(8, 0), step=timedelta(minutes=5)
        )
        model_name = st.radio("Model", list(MODELS), horizontal=True)
        submitted = st.form_submit_button("Find routes", type="primary")

    if submitted:
        st.session_state["search"] = (day, src_name, dst_name, now_time, model_name)

    if "search" in st.session_state:
        day, src_name, dst_name, now_time, model_name = st.session_state["search"]
        if src_name == dst_name:
            st.warning("Pick two different stations.")
            st.stop()
        with st.spinner("Predicting delays with TabPFN..."):
            try:
                routes, seconds = plan(
                    day, src_name, dst_name, now_time, MODELS[model_name]
                )
            except (
                RuntimeError
            ) as error:  # tabpfn_client: the API is busy or unreachable
                st.error(
                    f"TabPFN is not reachable right now, please try again. ({error})"
                )
                st.stop()
        if not routes:
            st.info("No route found for that time.")
            st.stop()
        st.caption(f"{model_name} predicted the delays in {seconds:.1f} s.")
        st.html(render.page(routes))
