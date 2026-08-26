# proyecto_llm_mini

Proyecto de portfolio: construir un mini-LLM desde cero en dos fases, subido a GitHub.

## Roadmap

- **[EN CURSO] Fase 0 — Andamiaje**: estructura de repo, docs, dependencias.
- **Fase A — Mini-LLM de propósito general**: decoder-only con arquitectura moderna
  (RoPE, RMSNorm, SwiGLU, Grouped Query Attention), entrenado sobre texto real.
  Inspirado en [Ashx098/Mini-LLM](https://github.com/Ashx098/Mini-LLM), recortado a
  nuestro hardware (RTX 4060, 8GB). Deliberadamente "manos en la masa": se espera
  depurar errores reales de forma/dtype/memoria como parte del aprendizaje.
- **Fase B — Laboratorio de generalización de profundidad**: reproduce y extiende
  "Exploring Depth Generalization in Large Language Models for Solving Recursive Logic
  Tasks" (Zhiyuan He, AAAI 2026, `docs/papers/`). Compara tres arquitecturas —
  decoder-only, encoder-decoder clásico (`Attention Is All You Need`), y el pipeline
  "Looped Locate-and-Replace" del paper — sobre expresiones de código Python anidadas
  generadas sintéticamente (`(3 + (2 * (5 - 1)))`, `not (True and (False or True))`).
  Objetivo: reproducir el fallo de accuracy en profundidades OOD y demostrar cómo cada
  arquitectura lo mitiga (o no). Reutiliza bloques de `mini_llm/model/`.

## Entorno

- Windows, Python 3.13, PyTorch 2.7.1+cu118.
- GPU: NVIDIA RTX 4060, 8GB VRAM. Implicación práctica: usar mixed precision
  (bf16/fp16) y gradient accumulation cuando el batch no quepa en memoria; no asumir
  la VRAM del repo de referencia (A100 80GB).
- Shell: Git Bash (bash tool) o PowerShell.

## Mapa del repo

```
docs/
  karpathy/       Transcripts y resumen de la serie "Neural Networks: Zero to Hero"
                  (docs/karpathy/resumen.md) — base conceptual del proyecto.
  papers/         Attention_is_all_you_need.pdf,
                  Exploring_Depth_Generalization_..._Recursive_Logic_Tasks.pdf
mini_llm/         Fase A: tokenizer/, data/, model/, train/, inference/
depth_lab/        Fase B: data/ (generador sintético), models/ (decoder-only,
                  encoder-decoder, locator/replacer), eval/ (arnés de evaluación por
                  profundidad), demo/ (visualizador interactivo)
scripts/          Utilidades sueltas (p.ej. clean_vtt.py para los transcripts)
tests/            pytest
```

## Convenciones

- No reinventar utilidades de bajo nivel ya resueltas por librerías maduras: BPE vía
  `tokenizers` (HuggingFace), no un BPE propio.
- Sí reinventar a mano los bloques de arquitectura (atención, RoPE, RMSNorm, SwiGLU,
  bucle de entrenamiento) siguiendo el estilo educativo de la serie de Karpathy — es el
  punto del proyecto.
- Tests con `pytest` en `tests/`, sobre todo para el generador de datos de Fase B
  (verificar que las etiquetas de las expresiones anidadas son correctas).
- Scripts de entrenamiento ejecutables por CLI (no solo notebooks).
- Antes de un entrenamiento completo, verificar overfit en un batch pequeño para
  confirmar que la arquitectura es correcta.

## Referencias clave

- `docs/karpathy/resumen.md` — fundamentos (bigramas → MLP → BatchNorm → backprop
  manual → WaveNet → Transformer decoder-only completo).
- `docs/papers/Attention_is_all_you_need.pdf` — arquitectura encoder-decoder original.
- `docs/papers/Exploring_Depth_Generalization_..._Recursive_Logic_Tasks.pdf` — paper de
  Fase B: por qué los Transformers fallan en recursión profunda y el pipeline Looped
  Locate-and-Replace propuesto como mitigación.
