"""Fase D -- controlled decoder-only vs encoder-decoder comparison.

Same tokenizer, data, parameter budget and training budget for two topologies
built from the same modern blocks (RoPE, RMSNorm, SwiGLU, GQA):

- Cracked-D: decoder-only (reuses mini_llm.model.GPT).
- Sliced-D: bidirectional encoder + causal decoder with cross-attention.
- Cracked-D-full: control, identical to Cracked-D but pretrained with loss on
  every token instead of the prefix-LM continuation-only loss.

Nothing here modifies Fases A/B/C: shared building blocks are imported, never
edited. See AUDITORIA_FASE_C.md for the Fase C findings this phase corrects.
"""
