"""Tokens-per-word of the new Fase D tokenizer vs Fase A's (plan section 2).

This is the only evidence that training a new tokenizer was worth it: Fase A's
BPE was trained on WikiText-103, so on educational web text and on conversations
it should need more tokens per word than one trained on the actual Fase D mix.
Measured on both text types the plan asks for (pretraining text and chat), and
written as .json + .md like every other table.

Usage:
    python -m compare_lab.eval.tokenizer_report
    python -m compare_lab.eval.tokenizer_report --docs 300   # smaller sample
"""

import argparse
import json
from pathlib import Path

from compare_lab.data.tokenizer import Tokenizer, DEFAULT_TOKENIZER_DIR

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FASE_A_TOKENIZER = Path(__file__).resolve().parent.parent.parent / "mini_llm" / "data" / "artifacts" / "tokenizer"


def tokens_per_word(encode, texts: list[str]) -> dict:
    n_tokens = n_words = n_chars = 0
    for t in texts:
        n_tokens += len(encode(t))
        n_words += len(t.split())
        n_chars += len(t)
    return {"tokens": n_tokens, "words": n_words, "chars": n_chars,
            "tokens_per_word": round(n_tokens / max(1, n_words), 4),
            "chars_per_token": round(n_chars / max(1, n_tokens), 4)}


def collect_samples(docs: int, seed: int = 1337) -> dict[str, list[str]]:
    """Pretraining text and conversation text, both streamed (no full download)."""
    from compare_lab.data.streaming import sample_texts_for_tokenizer
    from compare_lab.data.prepare_smoltalk import sample_smoltalk_texts

    pretrain_texts = []
    for i, t in enumerate(sample_texts_for_tokenizer(max_docs_per_subset=max(1, docs // 2), seed=seed)):
        pretrain_texts.append(t)
        if len(pretrain_texts) >= docs:
            break
    chat_texts = list(sample_smoltalk_texts(max_docs=docs, seed=seed))
    return {"pretraining": pretrain_texts, "conversations": chat_texts}


def build(docs: int = 500, out_dir: Path = RESULTS_DIR,
          tokenizer_dir: Path = DEFAULT_TOKENIZER_DIR) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    new_tok = Tokenizer.from_dir(tokenizer_dir)
    encoders = {"fase_d": new_tok.encode}

    if (FASE_A_TOKENIZER / "vocab.json").exists():
        from mini_llm.tokenizer.bpe import BPETokenizer
        encoders["fase_a"] = BPETokenizer.from_dir(FASE_A_TOKENIZER).encode
    else:
        print(f"(Fase A tokenizer not found at {FASE_A_TOKENIZER} -- reporting Fase D only)")

    samples = collect_samples(docs)
    results = {}
    for text_kind, texts in samples.items():
        results[text_kind] = {name: tokens_per_word(enc, texts) for name, enc in encoders.items()}
        if "fase_a" in encoders:
            a = results[text_kind]["fase_a"]["tokens_per_word"]
            d = results[text_kind]["fase_d"]["tokens_per_word"]
            results[text_kind]["improvement_pct"] = round(100.0 * (a - d) / a, 2)

    payload = {"docs_per_kind": docs, "results": results,
               "vocab_size": {"fase_d": new_tok.vocab_size}}
    (out_dir / "tokenizer_report.json").write_text(json.dumps(payload, indent=2))

    lines = ["# Tokenizer de Fase D frente al de Fase A", "",
             "Tokens por palabra (menos es mejor: el mismo texto cabe en menos tokens).", "",
             "| Texto | Tokenizer | Tokens/palabra | Chars/token | Tokens | Palabras |",
             "| --- | --- | --- | --- | --- | --- |"]
    for text_kind, per_tok in results.items():
        for name in ("fase_a", "fase_d"):
            if name not in per_tok:
                continue
            m = per_tok[name]
            lines.append(f"| {text_kind} | {name} | {m['tokens_per_word']} | {m['chars_per_token']} | "
                         f"{m['tokens']:,} | {m['words']:,} |")
    lines.append("")
    for text_kind, per_tok in results.items():
        if "improvement_pct" in per_tok:
            lines.append(f"- **{text_kind}**: el tokenizer de Fase D necesita "
                         f"{per_tok['improvement_pct']:+.2f}% menos tokens por palabra que el de Fase A.")
    lines += ["", f"Muestra: {docs} documentos por tipo de texto, obtenidos por streaming.",
              "El tokenizer de Fase A se entreno sobre WikiText-103; el de Fase D sobre la propia",
              "mezcla de preentrenamiento (~90%) y conversaciones (~10%).", ""]
    (out_dir / "tokenizer_report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--docs", type=int, default=500)
    p.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    args = p.parse_args()
    payload = build(args.docs, tokenizer_dir=Path(args.tokenizer_dir))
    print(json.dumps(payload["results"], indent=2))


if __name__ == "__main__":
    main()
