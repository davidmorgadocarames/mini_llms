"""Measure the resident memory of loading the Fase D model(s), for the Streamlit
Community Cloud deploy check (plan section 9). Fase C never implemented a memory
measurement (its measure_efficiency only records params/latency), so this is new.

    python -m compare_lab.measure_memory --local <ckpt> <tokenizer_dir>
    python -m compare_lab.measure_memory --hf         # downloads from the Hub

Reports process RSS before and after loading. To estimate the full deployed app
(all pages' models resident at once), load this alongside the other phases'
checkpoints -- the app keeps every visited page's st.cache_resource models in
memory simultaneously.
"""

import argparse

import psutil


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf", action="store_true")
    p.add_argument("--local", nargs=2, metavar=("CKPT", "TOKENIZER_DIR"))
    args = p.parse_args()

    base = rss_mb()
    print(f"RSS before load: {base:.1f} MB")

    from compare_lab.demo import load_from_local, load_from_hf, chat

    if args.local:
        model, tok = load_from_local(args.local[0], args.local[1], device="cpu")
        device = "cpu"
    else:
        model, tok, device = load_from_hf(device="cpu")

    after = rss_mb()
    print(f"RSS after load:  {after:.1f} MB  (+{after - base:.1f} MB for the model)")
    print(f"params: {model.num_parameters():,}")

    reply = chat(model, tok, [{"role": "user", "content": "Hello, who are you?"}],
                 device=device, max_new_tokens=40)
    peak = rss_mb()
    print(f"RSS after one generation: {peak:.1f} MB")
    print(f"sample reply: {reply[:200]!r}")


if __name__ == "__main__":
    main()
