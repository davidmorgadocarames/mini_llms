from depth_lab.data.build_dataset import (
    TEST_DEPTHS,
    TRAIN_DEPTHS,
    build,
    load_jsonl,
)


def test_build_produces_expected_file_sizes(tmp_path, monkeypatch):
    monkeypatch.setattr("depth_lab.data.build_dataset.ARTIFACTS_DIR", tmp_path)
    paths = build("bool", n_train_per_depth=5, n_val_per_depth=3, n_test_per_depth=4, max_shallow=2)

    train = load_jsonl(paths["train"])
    assert len(train) == 5 * len(TRAIN_DEPTHS)

    val = load_jsonl(paths["val"])
    assert len(val) == 3 * len(TRAIN_DEPTHS)

    for d in TEST_DEPTHS:
        test = load_jsonl(paths[f"test_depth{d}"])
        assert len(test) == 4
        assert all(ex["depth"] == d for ex in test)


def test_train_and_test_depths_never_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr("depth_lab.data.build_dataset.ARTIFACTS_DIR", tmp_path)
    paths = build("bool", n_train_per_depth=5, n_val_per_depth=3, n_test_per_depth=4, max_shallow=2)

    train_depths = {ex["depth"] for ex in load_jsonl(paths["train"])}
    val_depths = {ex["depth"] for ex in load_jsonl(paths["val"])}
    test_depths = set()
    for d in TEST_DEPTHS:
        test_depths |= {ex["depth"] for ex in load_jsonl(paths[f"test_depth{d}"])}

    assert train_depths <= set(TRAIN_DEPTHS)
    assert val_depths <= set(TRAIN_DEPTHS)
    assert test_depths <= set(TEST_DEPTHS)
    assert train_depths.isdisjoint(test_depths)


def test_build_is_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr("depth_lab.data.build_dataset.ARTIFACTS_DIR", tmp_path / "a")
    paths_a = build("bool", n_train_per_depth=5, n_val_per_depth=3, n_test_per_depth=4, max_shallow=2)

    monkeypatch.setattr("depth_lab.data.build_dataset.ARTIFACTS_DIR", tmp_path / "b")
    paths_b = build("bool", n_train_per_depth=5, n_val_per_depth=3, n_test_per_depth=4, max_shallow=2)

    assert load_jsonl(paths_a["train"]) == load_jsonl(paths_b["train"])
    assert load_jsonl(paths_a["test_depth6"]) == load_jsonl(paths_b["test_depth6"])


def test_dataset_values_are_correct_booleans(tmp_path, monkeypatch):
    monkeypatch.setattr("depth_lab.data.build_dataset.ARTIFACTS_DIR", tmp_path)
    paths = build("bool", n_train_per_depth=5, n_val_per_depth=1, n_test_per_depth=1, max_shallow=2)
    for ex in load_jsonl(paths["train"]):
        assert eval(ex["expr"]) == ex["value"]  # noqa: S307 (safe: our own generated text)
