"""Where the evaluation files live. data/eval/README.md describes each of them."""

from pathlib import Path

EVAL = Path("data/eval")


def sample_runs(split: str) -> Path:
    """Stop rows of the sampled runs that every method is scored on."""
    return EVAL / split / "sample_runs.parquet"


def router_requests(split: str) -> Path:
    return EVAL / split / "router_requests.parquet"


def router_legs(split: str) -> Path:
    return EVAL / split / "router_legs.parquet"


def tabpfn_cache(split: str, variant: str) -> Path:
    """TabPFN predictions of one variant, one row per stop of every run predicted so far."""
    return EVAL / split / "tabpfn_cache" / f"{variant}.parquet"


def scores(split: str, subset: str, method: str) -> Path:
    """Actual delays next to one method's predictions, on one subset of the sampled runs."""
    return EVAL / split / "scores" / subset / f"{method}.parquet"


def subset_name(per_group: int | None, train_type: str | None) -> str:
    """Name of a subset of the sampled runs, e.g. "all", "2-per-group", "ice_10-per-group"."""
    parts = []
    if train_type is not None:
        parts.append(train_type.lower())
    if per_group is not None:
        parts.append(f"{per_group}-per-group")
    return "_".join(parts) or "all"
