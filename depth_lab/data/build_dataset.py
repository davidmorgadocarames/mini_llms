"""Builds the Fase B dataset: a mixed-depth training split (depths 0-5,
matching the paper's "train on recursion at most five") plus a validation
split at the same depths, and separate out-of-distribution test splits for
each depth 6-12 -- kept as independent files so accuracy can be measured per
depth (that per-depth breakdown is the whole point of the experiment).

Train/test seeds never overlap and train/test *depths* never overlap either
(0-5 vs 6-12), so every test example is genuinely out-of-distribution.

Usage:
    python -m depth_lab.data.build_dataset
    python -m depth_lab.data.build_dataset --domain arith
"""

import argparse
import json
from pathlib import Path

from depth_lab.data.generator import Example, generate_dataset

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

TRAIN_DEPTHS = range(0, 6)   # 0-5, in-distribution
TEST_DEPTHS = range(6, 13)   # 6-12, out-of-distribution
DEFAULT_MAX_SHALLOW = 2      # keeps expression length roughly linear in depth


def _write_jsonl(examples: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"expr": ex.expr, "value": ex.value, "depth": ex.depth}) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build(domain: str, n_train_per_depth: int, n_val_per_depth: int,
          n_test_per_depth: int, max_shallow: int | None) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    train = generate_dataset(domain, TRAIN_DEPTHS, n_train_per_depth, seed=0, max_shallow=max_shallow)
    val = generate_dataset(domain, TRAIN_DEPTHS, n_val_per_depth, seed=1, max_shallow=max_shallow)
    paths["train"] = ARTIFACTS_DIR / f"{domain}_train.jsonl"
    paths["val"] = ARTIFACTS_DIR / f"{domain}_val.jsonl"
    _write_jsonl(train, paths["train"])
    _write_jsonl(val, paths["val"])

    for d in TEST_DEPTHS:
        test = generate_dataset(domain, [d], n_test_per_depth, seed=1000 + d, max_shallow=max_shallow)
        key = f"test_depth{d}"
        paths[key] = ARTIFACTS_DIR / f"{domain}_{key}.jsonl"
        _write_jsonl(test, paths[key])

    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", default="bool", choices=["bool", "arith"])
    p.add_argument("--n-train-per-depth", type=int, default=1500)
    p.add_argument("--n-val-per-depth", type=int, default=200)
    p.add_argument("--n-test-per-depth", type=int, default=500)
    p.add_argument("--max-shallow", type=int, default=DEFAULT_MAX_SHALLOW)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = build(args.domain, args.n_train_per_depth, args.n_val_per_depth,
                   args.n_test_per_depth, args.max_shallow)
    for name, path in paths.items():
        n = sum(1 for _ in path.open(encoding="utf-8"))
        print(f"{name:16s} {n:6d} examples -> {path}")


if __name__ == "__main__":
    main()
