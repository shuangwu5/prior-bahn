"""Score the methods on router requests, with "now" set to each request's departure time.

Needs the rows from `python -m dbdelay.eval.requests <split>`. Each request is predicted
like in the app: one shared context for all its stops, built with what was known at
"now" (`build_context`). Per leg, the departure at the boarding stop and the arrival at
the alighting stop are scored. TabPFN predictions are cached per request, so an
interrupted run resumes where it stopped.

Scores are saved under the subset "requests" (or "requests_<N>" with --limit N).

Run from the repo root:
uv run --no-sync python -m dbdelay.eval.run_requests validation <method> [--limit N]
uv run --no-sync python -m dbdelay.eval.run_requests validation report [--subset requests]
"""

import argparse
import time
from datetime import datetime
from datetime import time as clock

import polars as pl

from dbdelay.eval import baselines, metrics, paths
from dbdelay.model.context import build_context
from dbdelay.model.features import with_days_ago
from dbdelay.model.predict import predict_delays

STOPS = "data/processed/stops.parquet"
MODELS = ["arr", "dep"]
# TabPFN variants: whether the context uses "now" or only days before the request day.
# Both leave S-Bahn out of the context for other trains and use "days_ago".
TABPFN = {"tabpfn_now": {"use_now": True}, "tabpfn_midnight": {"use_now": False}}
BASELINES = {
    "global": baselines.global_quantiles,
    "train_station": baselines.train_station_quantiles,
    "carry_forward": baselines.carry_forward,
}
METHODS = [*BASELINES, *TABPFN]
QCOLS = ["q50", "q80", "q95"]
KEYS = ["request_id", "run_id", "stop_num"]


def tabpfn(stops: pl.LazyFrame, rows: pl.DataFrame, split: str, variant: str):
    """Predictions for every row of `rows`, both models, with columns KEYS, model and
    QCOLS."""
    path = paths.request_cache(split, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = pl.read_parquet(path) if path.exists() else None
    done = set() if cache is None else set(cache["request_id"])

    requests = rows.partition_by("request_id", maintain_order=True)
    for i, req in enumerate(requests):
        request_id = req["request_id"][0]
        if request_id in done:
            continue
        t = time.time()
        now = req["now"][0]
        day = now.date()
        # each stop once, even when both of its events are scored
        query = req.unique(["run_id", "stop_num"], maintain_order=True)
        cutoff = now if TABPFN[variant]["use_now"] else datetime.combine(day, clock())
        context = build_context(stops, query, cutoff)
        context, query = with_days_ago(context, day), with_days_ago(query, day)
        pred = pl.concat(
            query.select(KEYS)
            .hstack(predict_delays(context, query, m, local=True, extra=("days_ago",)))
            .with_columns(model=pl.lit(m))
            for m in MODELS
        )
        cache = pred if cache is None else pl.concat([cache, pred])
        # write to a temporary file first, so an interruption cannot corrupt the cache
        tmp = path.with_suffix(".tmp")
        cache.write_parquet(tmp)
        tmp.replace(path)
        print(
            f"{variant} request {i + 1}/{len(requests)}: {len(query)} stops, "
            f"{len(context)} context rows, {time.time() - t:.0f}s",
            flush=True,
        )
    return cache


def predict(method: str, stops: pl.LazyFrame, rows: pl.DataFrame, split: str):
    """Predictions of `method` aligned with `rows` (QCOLS columns)."""
    if method in TABPFN:
        cache = tabpfn(stops, rows, split, method)
        return rows.select(*KEYS, "model").join(
            cache, on=[*KEYS, "model"], how="left", maintain_order="left"
        )
    fn = BASELINES[method]
    rows = rows.with_row_index("row")
    out = []
    for m in MODELS:
        part = rows.filter(pl.col("model") == m)
        pred = fn(stops, part.drop("row"), m)
        out.append(pred.with_columns(part["row"]))
    return pl.concat(out).sort("row")


def score_method(split: str, method: str, limit: int | None) -> None:
    stops = pl.scan_parquet(STOPS)
    rows = pl.read_parquet(paths.request_rows(split))
    if limit is not None:
        keep = rows["request_id"].unique().sort().head(limit).implode()
        rows = rows.filter(pl.col("request_id").is_in(keep))

    pred = predict(method, stops, rows, split).select(QCOLS)
    parts = []
    for m in MODELS:
        is_m = rows["model"] == m
        part = metrics.scored_rows(
            rows.filter(is_m),
            pred.filter(is_m),
            m,
            extra=("request_id", "seen_delay", "minutes_ahead"),
        )
        parts.append(part.with_columns(method=pl.lit(method), model=pl.lit(m)))
    scored = pl.concat(parts)
    out = paths.scores(split, subset_name(limit), method)
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(out)
    print(f"wrote {out}")


def subset_name(limit: int | None) -> str:
    return "requests" if limit is None else f"requests_{limit}"


def case() -> pl.Expr:
    """Not seen yet, or seen, by minutes from now to the event."""
    ahead = pl.col("minutes_ahead")
    return (
        pl.when(pl.col("seen_delay").is_null())
        .then(pl.lit("not seen"))
        .when(ahead <= 15)
        .then(pl.lit("seen, up to 15 min"))
        .when(ahead <= 60)
        .then(pl.lit("seen, 15 to 60 min"))
        .otherwise(pl.lit("seen, over 60 min"))
        .alias("case")
    )


def report(split: str, subset: str, methods: list[str] | None) -> None:
    """Scores of the saved methods of one subset, on the requests they all scored."""
    folder = paths.scores(split, subset, "x").parent
    methods = methods or sorted(p.stem for p in folder.glob("*.parquet"))
    scored = pl.concat(
        pl.read_parquet(paths.scores(split, subset, m)) for m in methods
    ).with_columns(case())
    n_methods = scored["method"].n_unique()
    shared = (
        scored.group_by("request_id")
        .agg(pl.col("method").n_unique())
        .filter(pl.col("method") == n_methods)
        .select("request_id")
    )
    scored = scored.join(shared, on="request_id")
    print(f"{split}, {subset}: {len(shared)} requests scored by all of {methods}")

    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_rows(-1)
    for by in (
        ["model", "method"],
        ["model", "case", "method"],
        ["model", "group", "method"],
    ):
        print(metrics.score(scored, by=by))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["validation", "test"])
    parser.add_argument("method", choices=[*METHODS, "report"])
    parser.add_argument("--limit", type=int, help="score only the first N requests")
    parser.add_argument("--subset", default="requests", help="subset to report")
    parser.add_argument("--methods", nargs="+", help="methods to report")
    args = parser.parse_args()
    if args.method == "report":
        report(args.split, args.subset, args.methods)
    else:
        score_method(args.split, args.method, args.limit)


if __name__ == "__main__":
    main()
