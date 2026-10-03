"""Streamlit UI for the offline router: plan a train journey on the planned timetable.

Run from the repo root: uv run --no-sync streamlit run app/streamlit_app.py
"""

import sys
from datetime import date, time, timedelta
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import router_core as rc

DATA_MONTH = (date(2026, 9, 1), date(2026, 9, 30))


@st.cache_resource(show_spinner="Building timetable for the day...")
def get_timetable(day: date) -> rc.Timetable:
    return rc.load_timetable(day)


def render_option(journey: list[rc.Leg]) -> None:
    """Show the planned journey only. Actual delays are deliberately left out."""
    first, last = journey[0], journey[-1]
    transfers = len(journey) - 1
    with st.container(border=True):
        st.markdown(
            f"**{first.dep:%H:%M} → {last.arr:%H:%M}** · "
            + ("direct" if transfers == 0 else f"{transfers} transfer(s)")
        )
        for leg in journey:
            t = leg.transfer_before
            if t is not None:
                st.markdown(
                    f"↳ transfer at **{t.station}**, planned {t.planned_min} min"
                )
            st.markdown(
                f"**{leg.label}** · {leg.from_station} {leg.dep:%H:%M} "
                f"→ {leg.to_station} {leg.arr:%H:%M}"
            )
            with st.expander(f"{len(leg.stops)} stops"):
                st.write(" → ".join(leg.stops))


st.set_page_config(page_title="Offline router", layout="centered")
st.title("Offline router")
st.caption("Train-only routes on the planned September 2026 timetable.")

with st.form("query"):
    day = st.date_input(
        "Date",
        value=date(2026, 9, 24),
        min_value=DATA_MONTH[0],
        max_value=DATA_MONTH[1],
    )
    stations = sorted(get_timetable(day).station_eva)
    col_from, col_to = st.columns(2)
    src_name = col_from.selectbox(
        "From", stations, index=stations.index("Heidelberg Hbf")
    )
    dst_name = col_to.selectbox("To", stations, index=stations.index("Lübeck Hbf"))
    col_time, col_n = st.columns(2)
    depart_time = col_time.time_input("Earliest departure", value=time(8, 0))
    n_options = col_n.number_input("Options", min_value=1, max_value=5, value=3)
    submitted = st.form_submit_button("Find routes", type="primary")

if submitted:
    tt = get_timetable(day)
    if src_name == dst_name:
        st.warning("Pick two different stations.")
        st.stop()
    src, dst = tt.station_eva[src_name], tt.station_eva[dst_name]
    depart = tt.day_start + timedelta(
        hours=depart_time.hour, minutes=depart_time.minute
    )
    found = 0
    for _ in range(n_options):
        legs = rc.earliest_arrival(tt, src, dst, depart)
        if legs is None:
            break
        journey = rc.replay(tt, legs)
        render_option(journey)
        found += 1
        depart = journey[0].dep + timedelta(minutes=1)
    if found == 0:
        st.info("No route found for that departure time.")
