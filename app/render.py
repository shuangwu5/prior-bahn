"""HTML for the route cards of the Streamlit app.

Each route is a dict made by `streamlit_app.plan`. The page shows one card per route, in
rank order: a dot in the color of the weakest transfer, the planned departure and arrival
as large as in the DB app, the arrival reached with 80% and 95% certainty, a timeline of
legs and transfers, and a vertical list of stops like the DB app. A switch in the page shows the actual delays and transfer
outcomes. It works in the browser alone, so opened stop lists stay open.
"""

from html import escape

from dbdelay.risk import LEVELS

CSS = """
<style>
.rr { --card:#ffffff; --ink:#1d1d1b; --muted:#5f5e58; --line:#e3e1da; --rail:#9a978e;
  --train:#2f4a6d; --train-ink:#ffffff; --badge:#1d1d1b; --badge-ink:#ffffff;
  --l1:#1f7a4a; --l2:#5c9e3a; --l3:#c08a00; --l4:#c0441f; --late:#c0441f; --ontime:#1f7a4a;
  color: var(--ink); font-size: 15px; line-height: 1.45; }
@media (prefers-color-scheme: dark) {
  .rr { --card:#22221f; --ink:#f1efe8; --muted:#b0ada3; --line:#3a3934; --rail:#6d6a62;
    --train:#8fb0dc; --train-ink:#111111; --badge:#f1efe8; --badge-ink:#161614;
    --l1:#6fd29a; --l2:#b0d977; --l3:#f0c04f; --l4:#f08a6a; --late:#f08a6a; --ontime:#6fd29a; }
}
.rr * { box-sizing: border-box; }
.rr .legend { display:flex; flex-wrap:wrap; gap:16px; font-size:14px; margin: 4px 0 12px; }
.rr .legend span { display:inline-flex; align-items:center; gap:6px; }
.rr .dot { display:inline-block; width:12px; height:12px; border-radius:50%; flex:none; background:var(--c); }
.rr .l1 { --c:var(--l1); } .rr .l2 { --c:var(--l2); } .rr .l3 { --c:var(--l3); } .rr .l4 { --c:var(--l4); }
.rr .l0 { --c:var(--rail); }
.rr .route { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px 20px; margin-bottom:14px; }
.rr .head { display:flex; flex-wrap:wrap; align-items:center; gap:4px 12px; }
.rr .head .dot { width:16px; height:16px; }
.rr .times { font-size:24px; font-weight:800; font-variant-numeric:tabular-nums; }
.rr .meta { color:var(--muted); font-size:16px; }
.rr .meta span + span::before { content:"|"; margin:0 10px; color:var(--line); }
.rr .forecast { color:var(--muted); font-size:14px; margin-top:2px; }
.rr .forecast b { color:var(--ink); font-variant-numeric:tabular-nums; }
.rr .track { display:flex; align-items:center; height:30px; margin-top:16px; }
.rr .leg { height:24px; border-radius:6px; background:var(--train); color:var(--train-ink); font-size:12px; font-weight:700;
  display:flex; align-items:center; justify-content:center; overflow:hidden; white-space:nowrap; padding:0 6px; min-width:28px; }
.rr .wait { height:4px; position:relative; min-width:24px;
  background:repeating-linear-gradient(90deg, var(--rail) 0 5px, transparent 5px 9px); }
.rr .wait .dot { position:absolute; left:50%; top:50%; width:16px; height:16px; margin:-8px 0 0 -8px; box-shadow:0 0 0 3px var(--card); }
.rr .ends { display:flex; justify-content:space-between; font-size:13px; color:var(--muted); margin-top:6px; }
.rr details.stops { margin-top:12px; border-top:1px solid var(--line); padding-top:6px; }
.rr summary { cursor:pointer; color:var(--muted); font-size:14px; padding:6px 0; }
.rr .row, .rr .xfer { display:grid; grid-template-columns:84px 28px 1fr; min-height:28px; }
.rr .t { text-align:right; font-size:14px; padding-top:3px; font-variant-numeric:tabular-nums; }
.rr .t > span { display:block; }
.rr .rail { position:relative; }
.rr .row .rail::before { content:""; position:absolute; left:13px; top:0; bottom:0; width:3px; background:var(--rail); }
.rr .row.first .rail::before { top:14px; }
.rr .row.last .rail::before { bottom:calc(100% - 14px); }
.rr .row .rail i { position:absolute; left:8px; top:8px; width:13px; height:13px; border-radius:50%;
  background:var(--card); border:2.5px solid var(--rail); }
.rr .row.big .rail i { left:6px; top:6px; width:17px; height:17px; border-width:3px; border-color:var(--ink); }
.rr .n { padding:3px 0 10px 6px; }
.rr .row.big .n, .rr .row.big .t { font-weight:700; }
.rr .badge { display:inline-block; background:var(--badge); color:var(--badge-ink); font-weight:700; font-size:14px;
  padding:3px 9px; border-radius:6px; margin-top:2px; }
.rr .to, .rr .pred, .rr .known { color:var(--muted); font-size:14px; font-weight:400; }
.rr .pred b { color:var(--ink); }
.rr .d { font-weight:700; margin-left:4px; display:none; }
.rr .d.ok { color:var(--ontime); } .rr .d.bad { color:var(--late); }
.rr .xfer .rail::before { content:""; position:absolute; left:13px; top:0; bottom:0; width:3px;
  background:repeating-linear-gradient(180deg, var(--rail) 0 3px, transparent 3px 7px); }
.rr .xfer .rail .dot { position:absolute; left:7px; top:12px; width:15px; height:15px; }
.rr .xfer .n { padding:9px 0 12px 6px; color:var(--muted); font-size:14px; }
.rr .held { color:var(--ontime); font-weight:700; margin-left:6px; display:none; }
.rr .missed { color:var(--late); font-weight:700; margin-left:6px; display:none; }
.rr:has(#rr-reveal:checked) .d, .rr:has(#rr-reveal:checked) .held, .rr:has(#rr-reveal:checked) .missed { display:inline; }
.rr .top { display:flex; flex-wrap:wrap; gap:12px 24px; align-items:center; justify-content:space-between; margin:4px 0 14px; }
.rr .top .legend { margin:0; }
.rr .reveal { display:inline-flex; align-items:center; gap:8px; font-weight:600; cursor:pointer; min-height:44px; }
.rr .reveal input { width:20px; height:20px; accent-color:var(--train); }
.rr details.mid { position:relative; }
.rr details.mid::before { content:""; position:absolute; left:97px; top:0; bottom:0; width:3px; background:var(--rail); }
.rr details.mid > summary { padding:2px 0 8px 118px; }
.rr .foot { color:var(--muted); font-size:13px; margin-top:16px; }
</style>
"""


def _top() -> str:
    """The legend of the transfer levels and the switch for what actually happened."""
    items = "".join(
        f'<span><i class="dot l{k}"></i>{label.capitalize()} · {chance}</span>'
        for k, (label, chance) in LEVELS.items()
    )
    return (
        f'<div class="top"><div class="legend">{items}</div>'
        '<label class="reveal"><input type="checkbox" id="rr-reveal"> '
        "Show what actually happened</label></div>"
    )


def _hm(t) -> str:
    return "" if t is None else f"{t:%H:%M}"


def _delay(d, canceled: bool) -> str:
    if canceled:
        return '<span class="d bad">canceled</span>'
    if d is None:
        return ""
    return f'<span class="d {"bad" if d > 5 else "ok"}">{d:+d}</span>'


def _stop(
    s: dict, cls: str, extra: str = "", arr: bool = True, dep: bool = True
) -> str:
    """One stop of the list. Where you board only the departure is shown (`arr` False),
    where you leave only the arrival (`dep` False), as in the DB app."""
    times = []
    if arr and s["planned_arr"] is not None:
        times.append(
            f"<span>{_hm(s['planned_arr'])}"
            f"{_delay(s['arr_delay'], s['arr_canceled'])}</span>"
        )
    if dep and s["planned_dep"] is not None:
        times.append(
            f"<span>{_hm(s['planned_dep'])}"
            f"{_delay(s['dep_delay'], s['dep_canceled'])}</span>"
        )
    return (
        f'<div class="row {cls}"><div class="t">{"".join(times)}</div>'
        f'<div class="rail"><i></i></div>'
        f'<div class="n">{escape(s["station"])}{extra}</div></div>'
    )


def _leg(leg: dict) -> str:
    stops = leg["stops"]
    first, last, mid = stops[0], stops[-1], stops[1:-1]
    known = (
        f'<div class="known">running, last known delay {leg["last_known_delay"]:+.0f} min</div>'
        if leg["last_known_delay"] is not None
        else ""
    )
    html = _stop(first, "big first", arr=False)
    html += (
        '<div class="row"><div class="t"></div><div class="rail"></div><div class="n">'
        f'<span class="badge">{escape(leg["label"])}</span>'
        f'<div class="to">to {escape(leg["destination"])}</div>{known}'
    )
    if mid:
        rows = "".join(_stop(s, "") for s in mid)
        html += (
            f'</div></div><details class="mid"><summary>{len(mid)} stop'
            f"{'s' if len(mid) > 1 else ''}</summary>{rows}</details>"
        )
    else:
        html += "</div></div>"
    pred = (
        f'<div class="pred">expected <b>{leg["q50"]:+.0f}</b> · 80%: {leg["q80"]:+.0f}</div>'
        if leg["q50"] is not None
        else ""
    )
    html += _stop(last, "big last", pred, dep=False)
    t = leg["transfer_after"]
    if t is not None:
        outcome = ""
        if t["held"] is not None:
            outcome = (
                '<span class="held">✓ held</span>'
                if t["held"]
                else '<span class="missed">✗ missed</span>'
            )
        html += (
            f'<div class="xfer"><div class="t"></div><div class="rail">'
            f'<i class="dot l{t["level"] or 0}"></i></div>'
            f'<div class="n">Transfer, {t["planned_min"]} min{outcome}</div></div>'
        )
    return html


def card(route: dict) -> str:
    legs = route["legs"]
    bar = []
    for leg in legs:
        bar.append(
            f'<div class="leg" style="flex:{leg["minutes"]}" '
            f'title="{escape(leg["label"])}">{escape(leg["short"])}</div>'
        )
        t = leg["transfer_after"]
        if t is not None:
            bar.append(
                f'<div class="wait" style="flex:{max(t["planned_min"], 1)}">'
                f'<i class="dot l{t["level"] or 0}"></i></div>'
            )
    n = len(legs) - 1
    caveat = "" if route["weakest"] == 1 else ", if all transfers work"
    minutes = int(
        (route["planned_arrival"] - route["planned_departure"]).total_seconds() // 60
    )
    transfers = "Direct" if n == 0 else f"{n} transfer" + ("s" if n > 1 else "")
    return (
        '<div class="route"><div class="head">'
        f'<i class="dot l{route["weakest"]}"></i>'
        f'<span class="times">{_hm(route["planned_departure"])} – '
        f"{_hm(route['planned_arrival'])}</span>"
        f'<span class="meta"><span>{minutes // 60}h {minutes % 60:02d}min</span>'
        f"<span>{transfers}</span></span></div>"
        f'<div class="forecast">Arrives by <b>{_hm(route["arrival_q80"])}</b> with 80% '
        f"chance, by <b>{_hm(route['arrival_q95'])}</b> with 95%{caveat}</div>"
        f'<div class="track">{"".join(bar)}</div>'
        f'<div class="ends"><span>{escape(legs[0]["stops"][0]["station"])}</span>'
        f"<span>{escape(legs[-1]['stops'][-1]['station'])}</span></div>"
        '<details class="stops"><summary>Stops and times</summary>'
        f"{''.join(_leg(leg) for leg in legs)}</details></div>"
    )


def page(routes: list[dict]) -> str:
    cards = "".join(card(r) for r in routes)
    foot = (
        '<p class="foot">A transfer\'s color: TabPFN predicts how late the incoming train '
        "will be. <b>Very likely</b> means you still have 2 minutes to change trains even if "
        "the train is later than in 95% of predicted cases, <b>likely</b> at 80%, "
        "<b>uncertain</b> at 50%, <b>unlikely</b> otherwise. The connecting train is "
        "assumed to leave on time. A card's large dot is its weakest transfer.</p>"
    )
    return f'<div class="rr">{CSS}{_top()}{cards}{foot}</div>'
