# proyecto_llm_mini

Proyecto de portfolio: construir un mini-LLM desde cero en dos fases, subido a GitHub.

## Roadmap

- **[HECHO] Fase 0 — Andamiaje**: estructura de repo, docs, dependencias.
- **[HECHO] Fase A — Mini-LLM de propósito general**: decoder-only con arquitectura
  moderna (RoPE, RMSNorm, SwiGLU, Grouped Query Attention), entrenado sobre WikiText-103
  real. Inspirado en [Ashx098/Mini-LLM](https://github.com/Ashx098/Mini-LLM), recortado a
  nuestro hardware (RTX 4060, 8GB). 26.4M parámetros, val_loss final ≈ 2.95 tras 20.000
  pasos (~1.6h). Ver README para detalles y ejemplo de generación. Nota de proceso: en
  el camino apareció un error real de infraestructura (descarga de HuggingFace colgada
  por el backend Xet en esta red; arreglado con `HF_HUB_DISABLE_XET=1`), justo el tipo
  de aprendizaje "manos en la masa" que se buscaba en esta fase.
- **[EN CURSO] Fase B — Laboratorio de generalización de profundidad**: reproduce y
  extiende "Exploring Depth Generalization in Large Language Models for Solving
  Recursive Logic Tasks" (Zhiyuan He, AAAI 2026, `docs/papers/`). Compara tres
  arquitecturas — decoder-only, encoder-decoder clásico (`Attention Is All You Need`), y
  el pipeline "Looped Locate-and-Replace" del paper — sobre expresiones de código Python
  anidadas generadas sintéticamente (`(3 + (2 * (5 - 1)))`, `not (True and (False or
  True))`). Objetivo: reproducir el fallo de accuracy en profundidades OOD y demostrar
  cómo cada arquitectura lo mitiga (o no). Reutiliza bloques de `mini_llm/model/`.
  Generador de datos ya implementado y probado (`depth_lab/data/generator.py`);
  pendiente: modelos, arnés de evaluación y demo interactiva.
- **[HECHO] Coconut TUI**: interfaz de agente de terminal estilo Claude Code
  (`coconut_tui/`) sobre el modelo de Fase A — panel de conversación con streaming,
  panel de actividad plegable, diffs coloreados, arquitectura basada en eventos
  Pydantic desacoplada de Textual y del proveedor del LLM. Ver README para el detalle
  y `coconut_tui/agent.py` para las dos decisiones de diseño clave: los comandos `/`
  disparan herramientas reales (el modelo no tiene function-calling), y la UI nunca
  muestra razonamiento interno del modelo, solo acciones observables.

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
coconut_tui/      TUI de agente (Textual) sobre el modelo de Fase A: events.py/bus.py
                  (contrato Pydantic, sin Textual), providers/ (LLM desacoplado),
                  tools/ (read/test/diff reales), agent.py (orquestador, sin Textual),
                  widgets/ + app.py (la única capa que sí importa Textual)
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
