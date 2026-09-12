"""Configuration and fairness tables for Fase D (plan sections 3 and 14).

Writes each table as a .json (raw data) plus a .md (rendered), following the
project convention that a figure/table always ships with the data that feeds it.

The fairness table deliberately reports BOTH parameters and compute, because at
equal parameters an encoder-decoder and a decoder-only do NOT do equal work per
token -- and that trade-off is exactly what the architecture-comparison
literature tabulates:

- T5 (Raffel et al. 2020, sec. 3.2.2) lists an encoder-decoder with L+L layers
  as 2P parameters at M FLOPs, and an encoder-decoder with L/2+L/2 layers as
  P parameters at M/2 FLOPs, against a decoder-only prefix LM of L layers at
  P parameters and M FLOPs. Our pair sits on the P-parameter rows: Cracked-D is
  the decoder-only prefix LM (L=8, P, M) and Sliced-D is the L/2+L/2-style
  encoder-decoder (P, ~M/2).
- Both T5 and Wang et al. 2022 (ICML, "What Language Model Architecture and
  Pretraining Objective Work Best for Zero-Shot Generalization?") keep d_model,
  head count and FFN width IDENTICAL across architectures and vary only the
  number of layers; Wang et al. match the compute budget and let the
  encoder-decoder carry ~2x the parameters. Neither shrinks d_model to force a
  parameter match, which is why Sliced-D keeps d_model=512 here and differs in
  layer allocation instead.

So "matched parameters" is one axis, not the whole story; the table makes the
other axis (compute, and layers traversed per token) explicit instead of leaving
a reader to assume they are equal too.

Usage:
    python -m compare_lab.eval.tables
"""

import json
from pathlib import Path

from compare_lab import config
from compare_lab.models.cracked_d import build_model as build_cracked
from compare_lab.models.sliced_d import SlicedD

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _swiglu_hidden(d_model: int, mult: float, multiple_of: int) -> int:
    h = int(mult * d_model)
    return ((h + multiple_of - 1) // multiple_of) * multiple_of


def architecture_rows() -> list[dict]:
    cracked = build_cracked(config.cracked_config())
    sliced = SlicedD(config.sliced_config())
    cc, sc = config.cracked_config(), config.sliced_config()

    ffn_c = _swiglu_hidden(cc.n_embd, cc.ffn_mult, cc.ffn_multiple_of)
    ffn_s = _swiglu_hidden(sc.d_model, sc.ffn_mult, sc.ffn_multiple_of)

    rows = [
        {
            "model": "Cracked-D", "topology": "decoder-only (causal)",
            "layers": f"{cc.n_layer}", "d_model": cc.n_embd, "n_head": cc.n_head,
            "n_kv_head": cc.n_kv_head, "ffn_dim": ffn_c, "context": cc.block_size,
            "params": cracked.num_parameters(),
            "embedding_params": cracked.tok_emb.weight.numel(),
            "layers_per_token": cc.n_layer,
            "t5_row": "decoder-only prefix LM (L layers): P params, M FLOPs",
        },
        {
            "model": "Cracked-D-full", "topology": "decoder-only (causal), control",
            "layers": f"{cc.n_layer}", "d_model": cc.n_embd, "n_head": cc.n_head,
            "n_kv_head": cc.n_kv_head, "ffn_dim": ffn_c, "context": cc.block_size,
            "params": cracked.num_parameters(),
            "embedding_params": cracked.tok_emb.weight.numel(),
            "layers_per_token": cc.n_layer,
            "t5_row": "same architecture as Cracked-D; only the pretraining loss differs",
        },
        {
            "model": "Sliced-D", "topology": "encoder-decoder (bidirectional enc + causal dec + cross-attn)",
            "layers": f"{sc.n_enc_layer} enc + {sc.n_dec_layer} dec",
            "d_model": sc.d_model, "n_head": sc.n_head, "n_kv_head": sc.n_kv_head,
            "ffn_dim": ffn_s, "context": sc.max_src_len,
            "params": sliced.num_parameters(),
            "embedding_params": sliced.tok_emb.weight.numel(),
            # a continuation token passes through the decoder stack only; the
            # prefix->prediction path is enc + dec layers
            "layers_per_token": sc.n_dec_layer,
            "prefix_to_prediction_path": sc.n_enc_layer + sc.n_dec_layer,
            "t5_row": "encoder-decoder L/2+L/2-style: P params, ~M/2 FLOPs",
        },
    ]
    target = cracked.num_parameters()
    for r in rows:
        r["params_vs_cracked_pct"] = round(100.0 * (r["params"] - target) / target, 2)
    return rows


def _md_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for r in rows:
        cells = []
        for _, key in columns:
            v = r.get(key, "")
            cells.append(f"{v:,}" if isinstance(v, int) and key not in ("n_head", "n_kv_head",
                                                                       "context", "d_model",
                                                                       "layers_per_token") else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_architecture_table(out_dir: Path = RESULTS_DIR) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = architecture_rows()
    payload = {"rows": rows, "tolerance_pct": config.PARAM_TOLERANCE * 100,
               "note": "d_model/heads/FFN are held identical across architectures and only the "
                       "layer allocation differs, following T5 (Raffel et al. 2020) and Wang et al. "
                       "(2022), which keep width fixed and vary depth rather than shrinking width "
                       "to force a parameter match."}
    (out_dir / "architecture_config.json").write_text(json.dumps(payload, indent=2))

    cols = [("Modelo", "model"), ("Topologia", "topology"), ("Capas", "layers"),
            ("d_model", "d_model"), ("Cabezas", "n_head"), ("Cabezas KV", "n_kv_head"),
            ("FFN", "ffn_dim"), ("Contexto", "context"), ("Parametros", "params"),
            ("Params embedding", "embedding_params"), ("vs Cracked-D", "params_vs_cracked_pct")]
    md = ["# Configuracion de las tres arquitecturas (Fase D)", "",
          _md_table(rows, cols), "",
          f"Tolerancia de igualdad de parametros: +-{config.PARAM_TOLERANCE * 100:.0f}%.", "",
          "## Por que no se iguala tambien la profundidad", "",
          "No se pueden igualar a la vez parametros, profundidad por token y anchura: las dos",
          "topologias tienen formas estructuralmente distintas. T5 (Raffel et al. 2020, sec. 3.2.2)",
          "tabula justamente ese compromiso: un encoder-decoder de L+L capas son 2P parametros a",
          "M FLOPs, y uno de L/2+L/2 capas son P parametros a M/2 FLOPs, frente a un decoder-only",
          "prefix LM de L capas con P parametros y M FLOPs. Wang et al. (2022, ICML) mantienen",
          "identicos d_model, cabezas y FFN entre arquitecturas y varian solo el numero de capas,",
          "igualando el **computo** y dejando que el encoder-decoder tenga ~2x parametros.", "",
          "Esta fase iguala **parametros** (la restriccion del plan) preservando d_model=512 en",
          "ambos modelos, que es el eje que la literatura mantiene fijo. La consecuencia, explicita",
          "en la tabla, es que un token de continuacion atraviesa 4 capas en Sliced-D frente a 8 en",
          "Cracked-D (aunque el camino prefijo->prediccion sea de 3+4=7 capas), y que Sliced-D hace",
          "menos FLOPs por fragmento. Cualquier conclusion debe leerse con ese matiz.", ""]
    (out_dir / "architecture_config.md").write_text("\n".join(md), encoding="utf-8")
    return payload


def main() -> None:
    payload = write_architecture_table()
    for r in payload["rows"]:
        print(f"{r['model']:16} {r['layers']:>14}  d={r['d_model']}  "
              f"{r['params']:>12,} params ({r['params_vs_cracked_pct']:+.2f}%)  "
              f"capas/token={r['layers_per_token']}")
    print(f"\nwritten to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
