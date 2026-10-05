"""
HTML for the route cards of the Streamlit app.
"""

from datetime import datetime
from html import escape
from itertools import pairwise
from pathlib import Path

from priorbahn.risk import CHANGE_MIN, LEVELS

CSS = f"<style>\n{Path(__file__).with_name('route_cards.css').read_text()}</style>"


def _top() -> str:
    """
    The legend of the transfer levels and the switch for what actually happened.
    Hovering or tapping the info icon explains the levels.
    """
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


def _time(t: datetime | None) -> str:
    return "" if t is None else f"{t:%H:%M}"


def _delay(d: int | None, canceled: bool) -> str:
    if canceled:
        return '<span class="d bad">canceled</span>'
    if d is None:
        return ""
    return f'<span class="d {"bad" if d > 5 else "ok"}">{d:+d}</span>'


def _density(qs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Points (delay, density) of the curve from the quantiles (level, delay), on a grid of
    1/4 minute. The share of trains between two quantiles is spread evenly between them,
    over at least the minute after the lower one, because a delay of 0 means 0 to 59
    seconds late. The 5% above the last quantile is spread over 2 such steps, the 5%
    below the first one too, but not below 0 unless TabPFN predicts early arrivals. A
    moving average over 1 minute smooths the steps; nothing is drawn below that floor.
    """
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


def _curve(leg: dict, room: float | None, actual: int | None) -> str:
    """
    The predicted arrival delay of `leg` as a curve, with the 50%, 80% and 95%
    quantiles marked. Before a transfer, `room` is its time left to change: the curve is
    green where the transfer still works and red where it fails. At the destination
    (`room` None) it has one color. `actual` (the real delay, None if unknown) is marked
    only when the switch is on. Drawn with clip-path, because st.html removes SVG.
    """
    # an early arrival is drawn as on time (and "expected" is shown as at least +0): it makes the transfer just the same, and
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

    def sx(x: float) -> float:  # % of the width
        return (min(max(x, xmin), xmax) - xmin) / (xmax - xmin) * 100

    def area(a: float, b: float) -> str:
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
        values = dict.fromkeys(f"{round(d):+d}" for _, d in g)  # unique, in order
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
        last_chance = f'<span style="left:{sx(room):.1f}%"><b>{round(room):+d}</b>last chance</span>'
        # without the 0 where the two labels would overlap
        ticks = last_chance if abs(sx(room) - sx(0)) <= 5 else ticks + last_chance
    return (
        f'<div class="xdist">{top}<div class="ticks above">{above}</div>'
        f'<div class="plot">{layers}</div><div class="ticks">{ticks}</div></div>'
    )


def _stop(
    s: dict, cls: str, extra: str = "", arr: bool = True, dep: bool = True
) -> str:
    """
    One stop of the list. Where you board only the departure is shown (`arr` False),
    where you leave only the arrival (`dep` False), as in the DB app.
    """
    times = []
    if arr and s["planned_arr"] is not None:
        times.append(
            f"<span>{_time(s['planned_arr'])}"
            f"{_delay(s['arr_delay'], s['arr_canceled'])}</span>"
        )
    if dep and s["planned_dep"] is not None:
        times.append(
            f"<span>{_time(s['planned_dep'])}"
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
        f'<div class="known">running, last known delay {round(leg["last_known_delay"]):+d} min</div>'
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
        f'<div class="pred">expected <b>{round(max(leg["q50"], 0)):+d}</b></div>'
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
        f'<span class="exp">(expected <b>{round(max(q50, 0)):+d}</b>)</span>'
        if q50 is not None
        else ""
    )
    return (
        '<div class="route"><div class="head">'
        f'<i class="dot l{route["weakest"]}"></i>'
        f'<span class="times">{_time(route["planned_departure"])} – '
        f"{_time(route['planned_arrival'])}</span>{expected}"
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
