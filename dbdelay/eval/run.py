"""Score TabPFN and the baselines on the sampled runs of one week (per-leg evaluation).

Needs the sampled rows from `python -m dbdelay.eval.runs <split>`. TabPFN gets one shared
context per run, built from the run's own stop rows as the query, like a request in the
app. Its predictions are cached in one file per variant, saved after every run, so an
interrupted run resumes where it stopped.

Each call scores one method on a subset of the sampled runs (all of them by default) and
saves its scored rows under that subset's name, see `paths.subset_name`. `report`
compares the saved methods of one subset on the runs they all share. data/eval/README.md
lists the files and the experiments behind them.

Run from the repo root:
uv run --no-sync python -m dbdelay.eval.run validation tabpfn [--per-group N]
    [--train-type ICE]
uv run --no-sync python -m dbdelay.eval.run validation report [--subset all]
    [--methods ...]
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
# TabPFN variants: whether S-Bahn rows are left out of the context for other trains
# (see dbdelay.model.context) and whether "days_ago" is a feature. The cached results in
# data/eval were made with an older context (2,000 rows, fixed shares, all earlier days).
TABPFN = {
    "tabpfn": {"skip_s_bahn": False, "days_ago": False},
    "tabpfn_no_sbahn": {"skip_s_bahn": True, "days_ago": False},
    "tabpfn_no_sbahn_days_ago": {"skip_s_bahn": True, "days_ago": True},
}
METHODS = ["global", "train_station", *TABPFN]


def tabpfn(
    stops: pl.LazyFrame, rows: pl.DataFrame, split: str, variant: str
) -> dict[str, pl.DataFrame]:
    config = TABPFN[variant]
    extra = ("days_ago",) if config["days_ago"] else ()
    # one cache file per variant and split, rewritten after every run
    path = paths.tabpfn_cache(split, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = pl.read_parquet(path) if path.exists() else None
    done = set() if cache is None else set(cache["run_id"])

    runs = rows.partition_by("run_id", maintain_order=True)
    for i, run in enumerate(runs):
        if run["run_id"][0] in done:
            continue
        t = time.time()
        day = run["run_day"][0].date()
        # "now" is the start of the run's day: whole runs are scored, so nothing of
        # that day may be known yet
        context = build_context(
            stops,
            run,
            datetime.combine(day, clock()),
            skip_s_bahn=config["skip_s_bahn"],
        )
        context, query = with_days_ago(context, day), with_days_ago(run, day)
        pred = pl.concat(
            [run.select("run_id", "stop_num")]
            + [
                predict_delays(context, query, m, local=True, extra=extra).rename(
                    lambda c, m=m: f"{m}_{c}"
                )
                for m in MODELS
            ],
            how="horizontal",
        )
        cache = pred if cache is None else pl.concat([cache, pred])
        # write to a temporary file first, so an interruption cannot corrupt the cache
        tmp = path.with_suffix(".tmp")
        cache.write_parquet(tmp)
        tmp.replace(path)
        print(
            f"{variant} run {i + 1}/{len(runs)}: {len(run)} rows, "
            f"{len(context)} context rows, {time.time() - t:.0f}s",
            flush=True,
        )

    pred = rows.select("run_id", "stop_num").join(
        cache, on=["run_id", "stop_num"], how="left", maintain_order="left"
    )
    return {
        m: pred.select(pl.col(f"^{m}_.*$").name.map(lambda c: c.split("_", 1)[1]))
        for m in MODELS
    }


def predict(method: str, stops: pl.LazyFrame, rows: pl.DataFrame, split: str):
    """Predictions of `method` for both models, aligned with `rows`."""
    if method in TABPFN:
        return tabpfn(stops, rows, split, method)
    fn = {
        "global": baselines.global_quantiles,
        "train_station": baselines.train_station_quantiles,
    }[method]
    return {m: fn(stops, rows, m) for m in MODELS}


def score_method(
    split: str, method: str, per_group: int | None, train_type: str | None
) -> None:
    stops = pl.scan_parquet(STOPS)
    rows = pl.read_parquet(paths.sample_runs(split))
    if train_type is not None:
        rows = rows.filter(pl.col("train_type") == train_type)
    if per_group is not None:
        # the first runs of each group, so a smaller trial is a subset of a larger one
        keep = (
            rows.select("group", "run_id")
            .unique()
            .sort("run_id")
            .group_by("group")
            .head(per_group)
        )
        rows = rows.join(keep.select("run_id"), on="run_id", maintain_order="left")

    scored = pl.concat(
        metrics.scored_rows(rows, pred, model).with_columns(
            method=pl.lit(method), model=pl.lit(model)
        )
        for model, pred in predict(method, stops, rows, split).items()
    )
    out = paths.scores(split, paths.subset_name(per_group, train_type), method)
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(out)
    print(f"wrote {out}")


def report(split: str, subset: str, methods: list[str] | None) -> None:
    """Scores of the saved `methods` of one subset (all saved ones by default), on the
    runs that every one of them has scored."""
    folder = paths.scores(split, subset, "x").parent
    methods = methods or sorted(p.stem for p in folder.glob("*.parquet"))
    scored = pl.concat(pl.read_parquet(paths.scores(split, subset, m)) for m in methods)
    n_methods = scored["method"].n_unique()
    shared = (
        scored.group_by("run_id")
        .agg(pl.col("method").n_unique())
        .filter(pl.col("method") == n_methods)
        .select("run_id")
    )
    scored = scored.join(shared, on="run_id")
    print(f"{split}, {subset}: {len(shared)} runs scored by all of {methods}")

    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_rows(-1)
    print(metrics.score(scored, by=["model", "method"]))
    print(metrics.score(scored, by=["model", "group", "method"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["validation", "test"])
    parser.add_argument("method", choices=[*METHODS, "report"])
    parser.add_argument("--per-group", type=int, help="score only N runs per group")
    parser.add_argument("--train-type", help="score only runs of this train type")
    parser.add_argument("--subset", default="all", help="subset to report")
    parser.add_argument("--methods", nargs="+", help="methods to report")
    args = parser.parse_args()
    if args.method == "report":
        report(args.split, args.subset, args.methods)
    else:
        score_method(args.split, args.method, args.per_group, args.train_type)


if __name__ == "__main__":
    main()
