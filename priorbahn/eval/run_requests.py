"""Score the methods on router requests, with "now" set to each request's departure time.

Needs the rows from `python -m priorbahn.eval.requests <split>`. Each request is predicted
like in the app: one shared context for all its stops, built with what was known at
"now" (`build_context`). Per leg, the departure at the boarding stop and the arrival at
the alighting stop are scored. TabPFN predictions are cached per request, so an
interrupted run resumes where it stopped.

Scores are saved under the subset "requests" (or "requests_<N>" with --limit N).

Run from the repo root:
uv run --no-sync python -m priorbahn.eval.run_requests validation <method> [--limit N]
    [--models arr] [--workers 4] [--local]
uv run --no-sync python -m priorbahn.eval.run_requests validation report [--subset requests]
    [--reference carry_forward]
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import polars as pl

from priorbahn.eval import baselines, metrics, paths
from priorbahn.model import last_known
from priorbahn.model.context import build_context
from priorbahn.model.features import with_days_ago
from priorbahn.model.predict import load_token, predict_delays

STOPS = "data/processed/stops.parquet"
MODELS = ["arr", "dep"]
# TabPFN variants: whether the last known delay is a feature and TabPFN predicts the change
# from it (docs/data-prep-plan.md, section 10), the context size, and how many days back
# the same train's rides go (the other context groups use 7 days). All use the context of
# priorbahn.model.context (known at now) and "days_ago". The cached results of the older
# variants "tabpfn_now" and "tabpfn_midnight" (2,000 rows, all earlier days) stay in
# data/eval, but their code is gone.
TABPFN = {
    "tabpfn_7d": {"last_known": False, "size": 10_000, "same_train_days": 7},
    "tabpfn_7d_last_known": {"last_known": True, "size": 10_000, "same_train_days": 7},
    "tabpfn_14d_5k_last_known": {
        "last_known": True,
        "size": 5_000,
        "same_train_days": 14,
    },
}
BASELINES = {
    "global": baselines.global_quantiles,
    "train_station": baselines.train_station_quantiles,
    "carry_forward": baselines.carry_forward,
}
METHODS = [*BASELINES, *TABPFN]
QCOLS = ["q50", "q80", "q95"]
KEYS = ["request_id", "run_id", "stop_num"]


def tabpfn(
    stops: pl.LazyFrame,
    rows: pl.DataFrame,
    split: str,
    variant: str,
    models: list[str],
    local: bool,
    workers: int,
):
    """Predictions for every row of `rows` and each of `models`, with columns KEYS, model
    and QCOLS. `local` runs TabPFN on this machine instead of the Prior Labs API.
    `workers` requests run at the same time (the API does the work, we mostly wait)."""
    if not local:
        load_token()
    config = TABPFN[variant]
    path = paths.request_cache(split, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = pl.read_parquet(path) if path.exists() else None
    done = set() if cache is None else set(cache.select("request_id", "model").rows())

    def predict_request(req: pl.DataFrame) -> pl.DataFrame:
        request_id, now = req["request_id"][0], req["now"][0]
        todo = [m for m in models if (request_id, m) not in done]
        # each stop once, even when both of its events are scored
        query = req.unique(["run_id", "stop_num"], maintain_order=True)
        context = build_context(
            stops,
            query,
            now,
            size=config["size"],
            same_train_days=config["same_train_days"],
        )
        context = with_days_ago(context, now.date())
        query = with_days_ago(query, now.date())
        parts = []
        for m in todo:
            ctx_m, query_m, extra = context, query, ("days_ago",)
            if config["last_known"]:
                ctx_m = last_known.with_last_known(context, stops, now, m, replay=True)
                query_m = last_known.with_last_known(query, stops, now, m, replay=False)
                extra = ("days_ago", *last_known.COLUMNS)
            pred = predict_delays(
                ctx_m,
                query_m,
                m,
                local=local,
                extra=extra,
                change_from="last_known_delay" if config["last_known"] else None,
            )
            parts.append(query.select(KEYS).hstack(pred).with_columns(model=pl.lit(m)))
        return pl.concat(parts)

    requests = [
        req
        for req in rows.partition_by("request_id", maintain_order=True)
        if any((req["request_id"][0], m) not in done for m in models)
    ]
    t = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(predict_request, req) for req in requests]
        for i, future in enumerate(as_completed(futures)):
            pred = future.result()
            cache = pred if cache is None else pl.concat([cache, pred])
            # write to a temporary file first, so an interruption cannot corrupt the cache
            tmp = path.with_suffix(".tmp")
            cache.write_parquet(tmp)
            tmp.replace(path)
            print(
                f"{variant}: {i + 1}/{len(requests)} requests done, "
                f"{time.time() - t:.0f}s so far",
                flush=True,
            )
    return cache


def predict(
    method: str,
    stops: pl.LazyFrame,
    rows: pl.DataFrame,
    split: str,
    models: list[str],
    local: bool,
    workers: int,
):
    """Predictions of `method` aligned with `rows` (QCOLS columns)."""
    if method in TABPFN:
        cache = tabpfn(stops, rows, split, method, models, local, workers)
        return rows.select(*KEYS, "model").join(
            cache, on=[*KEYS, "model"], how="left", maintain_order="left"
        )
    fn = BASELINES[method]
    rows = rows.with_row_index("row")
    out = []
    for m in models:
        part = rows.filter(pl.col("model") == m)
        pred = fn(stops, part.drop("row"), m)
        out.append(pred.with_columns(part["row"]))
    return pl.concat(out).sort("row")


def score_method(
    split: str,
    method: str,
    limit: int | None,
    models: list[str],
    local: bool,
    workers: int,
) -> None:
    stops = pl.scan_parquet(STOPS)
    rows = pl.read_parquet(paths.request_rows(split))
    rows = rows.filter(pl.col("model").is_in(models))
    if limit is not None:
        keep = rows["request_id"].unique().sort().head(limit).implode()
        rows = rows.filter(pl.col("request_id").is_in(keep))

    pred = predict(method, stops, rows, split, models, local, workers).select(QCOLS)
    parts = []
    for m in models:
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


def report(split: str, subset: str, methods: list[str] | None, reference: str) -> None:
    """Scores of the saved methods of one subset, on the requests they all scored, and
    each method's difference to `reference` with a 95% range (see `differences`)."""
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
    if reference in methods:
        print(differences(scored, reference))


def differences(scored: pl.DataFrame, reference: str, n: int = 2000) -> pl.DataFrame:
    """Mean pinball loss of each method minus that of `reference`, on the same events,
    per model and "seen" or "not seen". The 95% range comes from drawing the requests
    again with replacement `n` times: a range that includes 0 means no clear difference.
    Delays have a long tail, so a few events can decide a mean."""
    loss = pl.mean_horizontal(metrics.pinball(q) for q in metrics.QCOLS)
    wide = scored.with_columns(
        loss=loss,
        seen=pl.when(pl.col("seen_delay").is_null())
        .then(pl.lit("not seen"))
        .otherwise(pl.lit("seen")),
    ).pivot(
        on="method",
        index=["request_id", "run_id", "stop_num", "model", "seen"],
        values="loss",
    )
    rng = np.random.default_rng(0)
    out = []
    for (model, seen), part in wide.group_by("model", "seen", maintain_order=True):
        per_request = part.group_by("request_id").agg(
            pl.len(), *(pl.col(m).sum() for m in wide.columns[5:])
        )
        counts = per_request["len"].to_numpy()
        draws = rng.integers(0, len(per_request), (n, len(per_request)))
        for m in wide.columns[5:]:
            if m == reference:
                continue
            diff = (per_request[m] - per_request[reference]).to_numpy()
            means = diff[draws].sum(axis=1) / counts[draws].sum(axis=1)
            low, high = np.percentile(means, [2.5, 97.5])
            out.append((model, seen, m, diff.sum() / counts.sum(), low, high))
    return pl.DataFrame(
        out,
        schema=["model", "seen", "method", f"minus_{reference}", "low", "high"],
        orient="row",
    ).sort("model", "seen", "method")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["validation", "test"])
    parser.add_argument("method", choices=[*METHODS, "report"])
    parser.add_argument("--limit", type=int, help="score only the first N requests")
    parser.add_argument(
        "--local", action="store_true", help="run TabPFN here instead of the API"
    )
    parser.add_argument(
        "--models", nargs="+", default=MODELS, choices=MODELS, help="events to score"
    )
    parser.add_argument("--workers", type=int, default=4, help="requests at a time")
    parser.add_argument(
        "--reference", default="carry_forward", help="method the others are compared to"
    )
    parser.add_argument("--subset", default="requests", help="subset to report")
    parser.add_argument("--methods", nargs="+", help="methods to report")
    args = parser.parse_args()
    if args.method == "report":
        report(args.split, args.subset, args.methods, args.reference)
    else:
        score_method(
            args.split, args.method, args.limit, args.models, args.local, args.workers
        )


if __name__ == "__main__":
    main()
