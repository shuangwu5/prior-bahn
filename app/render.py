"""HTML for the route cards of the Streamlit app.

Each route is a dict made by `streamlit_app.plan`. The page shows one card per route, in
rank order: a dot in the color of the weakest transfer, the planned departure and arrival
as large as in the DB app with the expected (median) arrival delay, a timeline of legs and
transfers, and a vertical list of stops like the DB app. In the stop list, a curve shows the
predicted arrival delay at each transfer, against the time left to change, and at the
destination. A switch in the page shows the actual delays and transfer outcomes. It works in
the browser alone, so opened stop lists stay open.
"""

from html import escape
from itertools import pairwise

from dbdelay.risk import CHANGE_MIN, LEVELS

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
.rr .legend { display:flex; flex-wrap:wrap; gap:8px 14px; font-size:14px; margin: 4px 0 12px; }
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
.rr .exp { color:var(--muted); font-size:16px; margin-left:-4px; }
.rr .exp b { color:var(--ink); }
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
/* pinned to the bottom-right corner, so it can be switched without scrolling up */
.rr .reveal { position:fixed; right:24px; bottom:24px; z-index:100; display:inline-flex; align-items:center; gap:8px;
  font-weight:600; cursor:pointer; min-height:44px; padding:0 16px; background:var(--card); border:1px solid var(--line);
  border-radius:22px; box-shadow:0 4px 14px rgba(0,0,0,.15); }
.rr .reveal input { width:20px; height:20px; accent-color:var(--train); }
.rr details.mid { position:relative; }
.rr details.mid::before { content:""; position:absolute; left:97px; top:0; bottom:0; width:3px; background:var(--rail); }
.rr details.mid > summary { padding:2px 0 8px 118px; }
.rr .legend .lead { font-weight:700; gap:4px; position:relative; }
.rr .info { display:inline-flex; align-items:center; justify-content:center; width:18px; height:18px;
  border-radius:50%; border:1.5px solid var(--muted); color:var(--muted); font-size:12px; font-weight:700;
  font-style:normal; cursor:help; }
.rr .info .tip { display:none; position:absolute; left:0; top:26px; z-index:10; width:min(420px, 80vw);
  background:var(--card); color:var(--ink); border:1px solid var(--line); border-radius:10px; padding:10px 12px;
  font-size:13px; font-weight:400; line-height:1.45; box-shadow:0 6px 20px rgba(0,0,0,.15); }
.rr .info:hover .tip, .rr .info:focus .tip { display:block; }
.rr .tip .lv { display:block; margin-top:6px; }
.rr .tip .lv .dot { width:10px; height:10px; margin-right:6px; }
.rr .xdist { margin-top:6px; max-width:420px; position:relative; font-weight:400; }
.rr .xdist > .actual.line { display:none; position:absolute; top:16px; bottom:38px; width:0; margin-left:-1px;
  border-left:2px solid var(--c); z-index:1; }
.rr .ticks span.r { transform:translateX(calc(-100% + 4px)); text-align:right; }
.rr .ticks span.l { transform:translateX(-4px); text-align:left; }
.rr .plot { position:relative; height:56px; overflow:hidden; border-bottom:1px solid var(--line); }
.rr .plot i { position:absolute; left:0; right:0; top:0; bottom:0; }
.rr .plot .edge { background:var(--train); }
.rr .plot .ok { top:2px; bottom:-2px; background:color-mix(in srgb, var(--l1) 35%, var(--card)); }
.rr .plot .all { top:2px; bottom:-2px; background:color-mix(in srgb, var(--train) 25%, var(--card)); }
.rr .plot .bad { top:2px; bottom:-2px; background:color-mix(in srgb, var(--l4) 45%, var(--card)); }
.rr .plot b { position:absolute; top:0; bottom:0; width:0; margin-left:-1px; }
.rr .plot .zero { border-left:2px solid var(--ink); }
.rr .plot .room { border-left:2px dashed var(--l4); }
.rr .actual.ok { --c:var(--ontime); } .rr .actual.bad { --c:var(--late); }
.rr .ticks { position:relative; height:34px; margin-top:4px; font-size:12px; }
.rr .ticks.above { height:30px; margin:0 0 2px; }
.rr .ticks.above span { color:var(--muted); background:var(--card); padding:0 2px; z-index:2; }
.rr .plot .q { border-left:1.5px dotted var(--muted); }
.rr .ticks span { position:absolute; transform:translateX(-50%); text-align:center; line-height:1.25; white-space:nowrap; }
.rr .ticks b { display:block; color:var(--ink); font-variant-numeric:tabular-nums; }
.rr .ticks.actual { visibility:hidden; height:16px; margin:0; }
.rr .ticks.actual span { color:var(--c); }
.rr .ticks.actual b { display:inline; color:var(--c); }
.rr:has(#rr-reveal:checked) .xdist > .actual.line { display:block; }
.rr:has(#rr-reveal:checked) .ticks.actual { visibility:visible; }
</style>
"""


def _top() -> str:
    """The legend of the transfer levels and the switch for what actually happened.
    Hovering or tapping the info icon explains the levels."""
    # TabPFN's own chance from the quantile the transfer still works at, not the
    # measured share of LEVELS
    sure = {
        1: "95% sure or more",
        2: "80 to 95% sure",
        3: "50 to 80% sure",
        4: "less than 50% sure",
    }
    levels = "".join(
        f'<span class="lv"><i class="dot l{k}"></i><b>{label.capitalize()}</b>: '
        f"{sure[k]}</span>"
        for k, (label, _) in LEVELS.items()
    )
    tip = (
        "<b>How sure TabPFN is that you catch your next train</b>"
        f"{levels}"
        f'<span class="lv">We assume you need at least {CHANGE_MIN} minutes to change '
        "trains, and that the next train leaves on time.</span>"
        '<span class="lv">The big dot on a card shows its weakest transfer.</span>'
    )
    lead = (
        '<span class="lead">Transfer reliability'
        f'<i class="info" tabindex="0" aria-label="About transfer reliability">i'
        f'<span class="tip" role="tooltip">{tip}</span></i></span>'
    )
    items = "".join(
        f'<span><i class="dot l{k}"></i>{label.capitalize()}</span>'
        for k, (label, _) in LEVELS.items()
    )
    return (
        f'<div class="top"><div class="legend">{lead}{items}</div>'
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


def _density(qs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points (delay, density) of the curve from the quantiles (level, delay), on a grid of
    1/4 minute. The share of trains between two quantiles is spread evenly between them,
    over at least the minute after the lower one, because a delay of 0 means 0 to 59
    seconds late. The 5% above the last quantile is spread over 2 such steps, the 5%
    below the first one too, but not below 0 unless TabPFN predicts early arrivals. A
    moving average over 1 minute smooths the steps; nothing is drawn below that floor."""
    first, last = qs[0][1], qs[-1][1]
    floor = min(first, 0.0)
    edge_lo = 2 * max(qs[1][1] - first, 1.0)
    edge_hi = 2 * max(last - qs[-2][1], 1.0)
    steps = [(lo, hi, b - a) for (a, lo), (b, hi) in pairwise(qs)]
    steps += [
        (max(first - edge_lo, floor), first, qs[0][0]),
        (last, last + edge_hi, 1 - qs[-1][0]),
    ]
    dx = 0.25
    x0 = floor - 1
    xs = [x0 + i * dx for i in range(int((last + edge_hi + 1 - x0) / dx) + 1)]
    ys = [0.0] * len(xs)
    for lo, hi, share in steps:
        w = max(hi - lo, 1.0)
        for i, x in enumerate(xs):
            if lo <= x < lo + w:
                ys[i] += share / w
    k = 2  # 2 points on each side: 1 minute
    smooth = [
        sum(ys[max(i - k, 0) : i + k + 1]) / (2 * k + 1) if x >= floor else 0.0
        for i, x in enumerate(xs)
    ]
    return list(zip(xs, smooth))


def _at(pts: list[tuple[float, float]], x: float) -> float:
    for (x0, y0), (x1, y1) in pairwise(pts):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0 or 1)
    return 0.0


def _curve(leg: dict, room: float | None, actual) -> str:
    """The predicted arrival delay of `leg` as a curve, with the 50%, 80% and 95%
    quantiles marked. Before a transfer, `room` is its time left to change: the curve is
    green where the transfer still works and red where it fails. At the destination
    (`room` None) it has one color. `actual` (the real delay, None if unknown) is marked
    only when the switch is on. Drawn with clip-path, because st.html removes SVG."""
    # an early arrival is drawn as on time: it makes the transfer just the same, and
    # TabPFN's smooth curve spreads the many on-time arrivals below 0 (15% early where
    # 2.4% of all arrivals in the data are early)
    qs = [(q, max(d, 0.0)) for q, d in leg["quantiles"]]
    pts = _density(qs)
    # from just before the 5% quantile to just after the 95% one; the long tail of big
    # delays is cut so that the part around the transfer stays readable
    # with some room left of 0 for the labels of quantiles near 0
    xmax = max(qs[-1][1] + 2, (room or 0) + 3)
    xmin = min(qs[0][1] - 1, (room or 0) - 1, -0.15 * xmax)
    ymax = max(y for _, y in pts) * 1.08

    def sx(x):  # % of the width
        return (min(max(x, xmin), xmax) - xmin) / (xmax - xmin) * 100

    def area(a, b):
        """clip-path polygon of the curve between delays a and b, in %."""
        xs = [a, *(x for x, _ in pts if a < x < b), b]
        top = ", ".join(
            f"{sx(x):.1f}% {100 - _at(pts, x) / ymax * 100:.1f}%" for x in xs
        )
        return f"polygon({sx(a):.1f}% 100%, {top}, {sx(b):.1f}% 100%)"

    # the curve in the train color, and on top the same shape 2px lower in green and
    # red, so a 2px line stays visible along the top
    layers = f'<i class="edge" style="clip-path:{area(xmin, xmax)}"></i>'
    if room is None:
        layers += f'<i class="all" style="clip-path:{area(xmin, xmax)}"></i>'
    else:
        if room > xmin:
            layers += f'<i class="ok" style="clip-path:{area(xmin, room)}"></i>'
        if room < xmax:
            layers += f'<i class="bad" style="clip-path:{area(room, xmax)}"></i>'
        layers += f'<b class="room" style="left:{sx(room):.1f}%"></b>'
    layers += f'<b class="zero" style="left:{sx(0):.1f}%"></b>'
    # the 50%, 80% and 95% quantiles as dotted lines, labeled above the curve. Labels
    # closer than 10% of the width are merged; labels closer than 25% point away from
    # each other (the left one ends at its line, the right one starts at it)
    marks = [(q, d) for q, d in qs if q in (0.5, 0.8, 0.95)]
    groups = []
    for q, d in marks:
        layers += f'<b class="q" style="left:{sx(d):.1f}%"></b>'
        if groups and sx(d) - sx(groups[-1][0][1]) < 10:
            groups[-1].append((q, d))
        else:
            groups.append([(q, d)])
    xs = [sx(g[0][1]) for g in groups]
    above = ""
    for i, g in enumerate(groups):
        values = dict.fromkeys(f"{d:+.0f}" for _, d in g)  # unique, in order
        side = ""
        if i + 1 < len(xs) and xs[i + 1] - xs[i] < 25:
            side = "r"
        elif i > 0 and xs[i] - xs[i - 1] < 25:
            side = "l"
        above += (
            f'<span class="{side}" style="left:{xs[i]:.1f}%">'
            f"{'/'.join('median' if q == 0.5 else f'{q:.0%}' for q, _ in g)}"
            f"<b>{'/'.join(values)}</b></span>"
        )
    top = ""
    if actual is not None:
        # at the destination, late is more than 5 minutes, as in the stop list
        cls = "ok" if actual <= (5 if room is None else room) else "bad"
        # the line runs from the label at the top down through the curve
        line = f'<b class="actual line {cls}" style="left:{sx(actual):.1f}%"></b>'
        top = line + (
            f'<div class="ticks actual {cls}"><span style="left:{sx(actual):.1f}%">'
            f"actual <b>{actual:+d}</b></span></div>"
        )

    ticks = f'<span style="left:{sx(0):.1f}%"><b>0</b></span>'
    if room is not None:
        last_chance = (
            f'<span style="left:{sx(room):.1f}%"><b>{room:+.0f}</b>last chance</span>'
        )
        # without the 0 where the two labels would overlap
        ticks = last_chance if abs(sx(room) - sx(0)) <= 5 else ticks + last_chance
    return (
        f'<div class="xdist">{top}<div class="ticks above">{above}</div>'
        f'<div class="plot">{layers}</div><div class="ticks">{ticks}</div></div>'
    )


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
    t = leg["transfer_after"]
    pred = (
        f'<div class="pred">expected <b>{leg["q50"]:+.0f}</b></div>'
        if leg["q50"] is not None
        else ""
    )
    actual = None if last["arr_canceled"] else last["arr_delay"]
    has_curve = len(leg["quantiles"]) > 1
    if t is None and has_curve:  # the destination
        pred += _curve(leg, None, actual)
    html += _stop(last, "big last", pred, dep=False)
    if t is not None:
        outcome = ""
        if t["held"] is not None:
            outcome = (
                '<span class="held">✓ held</span>'
                if t["held"]
                else '<span class="missed">✗ missed</span>'
            )
        curve = _curve(leg, t["room_min"], actual) if has_curve else ""
        html += (
            f'<div class="xfer"><div class="t"></div><div class="rail">'
            f'<i class="dot l{t["level"] or 0}"></i></div>'
            f'<div class="n">Transfer, {t["planned_min"]} min{outcome}{curve}'
            "</div></div>"
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
    minutes = int(
        (route["planned_arrival"] - route["planned_departure"]).total_seconds() // 60
    )
    transfers = "Direct" if n == 0 else f"{n} transfer" + ("s" if n > 1 else "")
    q50 = legs[-1]["q50"]
    expected = (
        f'<span class="exp">(expected <b>{q50:+.0f}</b>)</span>'
        if q50 is not None
        else ""
    )
    return (
        '<div class="route"><div class="head">'
        f'<i class="dot l{route["weakest"]}"></i>'
        f'<span class="times">{_hm(route["planned_departure"])} – '
        f"{_hm(route['planned_arrival'])}</span>{expected}"
        f'<span class="meta"><span>{minutes // 60}h {minutes % 60:02d}min</span>'
        f"<span>{transfers}</span></span></div>"
        f'<div class="track">{"".join(bar)}</div>'
        f'<div class="ends"><span>{escape(legs[0]["stops"][0]["station"])}</span>'
        f"<span>{escape(legs[-1]['stops'][-1]['station'])}</span></div>"
        '<details class="stops"><summary>Stops and times</summary>'
        f"{''.join(_leg(leg) for leg in legs)}</details></div>"
    )


def page(routes: list[dict]) -> str:
    cards = "".join(card(r) for r in routes)
    return f'<div class="rr">{CSS}{_top()}{cards}</div>'
