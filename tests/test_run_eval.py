from depth_lab.eval.run_eval import plot_accuracy_vs_depth


def test_plot_accuracy_vs_depth_writes_a_file(tmp_path):
    results = {
        "baseline": {d: 1.0 / d for d in range(6, 13)},
        "encoder-decoder": {d: 0.5 for d in range(6, 13)},
        "llr": {d: 0.9 for d in range(6, 13)},
    }
    out_path = tmp_path / "accuracy_vs_depth.png"
    plot_accuracy_vs_depth(results, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
