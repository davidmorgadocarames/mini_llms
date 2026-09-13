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
- **[HECHO] Fase B — Laboratorio de generalización de profundidad**: reproduce y
  extiende "Exploring Depth Generalization in Large Language Models for Solving
  Recursive Logic Tasks" (Zhiyuan He, AAAI 2026, `docs/papers/`). Compara tres
  arquitecturas sobre expresiones booleanas anidadas generadas sintéticamente
  (`(True and (False or not (True)))`) con profundidad y longitud desacopladas
  (`max_shallow`, para que la profundidad sea la única variable que cambia en el
  experimento): decoder-only (reutiliza `mini_llm.model.GPT`), encoder-decoder clásico
  hecho desde cero (`Attention Is All You Need`: sinusoidal PE, MHA estándar, LayerNorm,
  MLP con ReLU), y el pipeline "Looped Locate-and-Replace" del paper (locator con ALiBi +
  clasificación por token, replacer con NoPE + generación, bucle de reducción iterativo).
  Entrenadas solo en profundidad ≤5, evaluadas en profundidad 6-12 (out-of-distribution,
  nunca tocado hasta la evaluación final). Resultado (`depth_lab/eval/results/`,
  enlazado desde el README): LLR domina en profundidades OOD moderadas (6-9,
  hasta 98.8% vs 78.6% del baseline) pero pierde su ventaja en las más extremas
  (10-12) por composición de error a través de la cadena de pasos, que crece con
  la profundidad — un hallazgo real y verificado, no la historia simple de "LLR
  siempre gana". Demo interactiva en `pages/1_Fase_B_Depth_Lab.py` (segunda página
  de la misma app de Streamlit de Fase A), con la traza de reducción de LLR animada
  paso a paso. Nota de proceso: al reproducir a escala real, un checkpoint de
  ~400 pasos daba accuracy OOD casi nula en LLR; antes de reportarlo como fallo de
  arquitectura, se re-entrenó más a fondo (3000 pasos) y el patrón desapareció —
  era infra-entrenamiento, no un bug. Mismo criterio de verificar antes de concluir
  que en el error de Xet de Fase A.
- **[HECHO] Coconut TUI**: interfaz de agente de terminal estilo Claude Code
  (`coconut_tui/`) sobre el modelo de Fase A — panel de conversación con streaming,
  panel de actividad plegable, diffs coloreados, arquitectura basada en eventos
  Pydantic desacoplada de Textual y del proveedor del LLM. Ver README para el detalle
  y `coconut_tui/agent.py` para las dos decisiones de diseño clave: los comandos `/`
  disparan herramientas reales (el modelo no tiene function-calling), y la UI nunca
  muestra razonamiento interno del modelo, solo acciones observables.
- **[HECHO] Fase C — Coconut interactivo + 3 arquitecturas en razonamiento real**:
  fine-tuning de instrucciones/chat (Alpaca + oasst1) sobre Coconut (**Cracked**,
  decoder-only), más la misma comparación de 3 arquitecturas de Fase B pero sobre
  GSM8K (razonamiento matemático en lenguaje natural) en vez de una gramática
  sintética: **Sliced** (encoder-decoder desde cero) y **Pressed** (Looped
  Locate-and-Replace adaptado: un drafter redacta un borrador, locator+replacer
  reutilizados literalmente de Fase B corrigen su aritmética). Comparación en 5
  piezas (`coconut_lab/eval/`, resultados y tablas enlazados desde el README):
  accuracy en GSM8K test por nº de pasos, set propio de 141 ejemplos con 3 semillas
  (Cracked gana con 21.5%±3.5%, margen pequeño), `lm-evaluation-harness`
  (`lambada_openai`+`piqa`), eficiencia, y k-fold=5 de estabilidad de entrenamiento
  (25 entrenamientos individuales, checkpointeados y resumibles automáticamente ante
  un corte — ver `mini_llm/train/checkpoint.py`). Resultado honesto: a esta escala de
  parámetros (8-26M) GSM8K real es muchísimo más difícil que el dominio sintético de
  Fase B — el accuracy exact-match se queda casi en el suelo (0-11%) incluso tras
  bajar el loss de entrenamiento con claridad, muy distinto del patrón de Fase B
  donde el dominio sintético se aprendía del todo en pocos miles de pasos. Demo
  interactiva en `pages/2_Fase_C_Coconut_Interactivo.py` (tercera página de la app de
  Streamlit), con selector de modelo en popup y chat multi-turno que conserva el
  historial al cambiar de arquitectura. Nota de proceso: un test de equivalencia
  durante la integración con `lm-evaluation-harness` destapó un bug real en
  `coconut_lab/models/sliced.py` (el decoder enmascaraba por error su propio token
  BOS como si fuera padding, ya que BOS y pad comparten id en el `BPETokenizer` de
  este proyecto — a diferencia del `CharTokenizer` de Fase B, que los tiene
  separados — dejando la primera posición sin ninguna clave válida a la que
  atender). Afectaba a todos los checkpoints de Sliced entrenados hasta ese momento;
  arreglado, Sliced reentrenado desde cero, y los puntos de C.6 afectados
  re-ejecutados con el checkpoint corregido — mismo criterio de verificar antes de
  concluir que en Fases A y B.
- **[EN CURSO] Fase D — por qué los chatbots responden mal**: investigación abierta
  sobre cómo mejorar las respuestas de Cracked/Sliced/Pressed (déficit de datos vs.
  Chinchilla, dominio de WikiText-103 vs. conversación, falta de preentrenamiento en
  Sliced/drafter de Pressed). Se concretó en `compare_lab/`: comparación
  **controlada** decoder-only vs encoder-decoder igualando datos, tokenizer,
  parámetros y presupuesto — las dos cosas que la Fase C no controlaba.

  **Etapa 1 [HECHA, luego descartada y rehecha en Etapa 2]**: solo **Cracked-D**
  (decoder-only, 26.354.176 params), preentrenado con objetivo prefix-LM sobre
  SmolLM-Corpus (1200M tokens, 1 época) y afinado sobre smol-smoltalk, val_loss
  2.7888 preentrenamiento / 1.5621 fine-tuning. Probado interactivamente y
  descartado: sin coherencia básica de chat (p. ej. no respondía bien a "¿quién
  es Isaac Newton?"). Diagnóstico respaldado en literatura (no una corazonada):
  capacidad de conocimiento factual limitada a ~2 bits/parámetro (Allen-Zhu & Li,
  *Physics of Language Models 3.3*, ICLR 2025) y ratio tokens/parámetro muy por
  debajo de modelos pequeños de referencia que sí funcionan como chat
  (TinyLlama-1.1B/3T tokens, SmolLM2-135M/2T — Muennighoff et al., *Scaling
  Data-Constrained LM*, NeurIPS 2023).

  **Etapa 2 [EN CURSO, con OK explícito]**: Cracked-D y Sliced-D redimensionados a
  ~80M parámetros (Cracked-D 80.628.480; Sliced-D 80.335.872, dentro de la
  tolerancia del 5%), `PRETRAIN_TOKENS` subido a 1.6B (suelo de Chinchilla, ~20
  tokens/parámetro — Hoffmann et al. 2022). Un ensayo corto en hardware real
  (`nvidia-smi` sondeado durante pasos reales) reveló que la VRAM no escala
  igual de limpio que el cómputo: `MICRO_BATCH=16` original llegaba al 97% y el
  throughput se desplomaba por thrashing del allocator; `MICRO_BATCH=4` /
  `GRAD_ACCUM=64` (mismo batch efectivo) baja a ~57-59% y además va más rápido
  (~24-26k tok/s). `finetune.py` no tiene grad-accum y sufría el mismo problema;
  `FINETUNE_BATCH_SIZE` bajado de 16 a 4 por el mismo motivo. Todo documentado
  con números reales en el docstring de `compare_lab/config.py`.

  Cracked-D-80M **entrenado**: val_loss 2.5014 preentrenamiento (mejora real
  sobre el 2.7888 de 26M) / 1.6878 fine-tuning (empeora ligeramente respecto al
  1.5621 de 26M, posiblemente por el batch de fine-tuning más pequeño; sin
  concluir todavía, pendiente de probar el chat de verdad), ~19.2h en una RTX
  4060. Sliced-D-80M: pendiente de lanzar. Cracked-D-full (control): pospuesto,
  fuera de esta tanda.

  Demo separada de la app multipágina de Fases A/B/C: `streamlit_app_fase_d.py`
  (antes `pages/3_Fase_D_Comparacion_Controlada.py`), desplegada como app propia
  de Streamlit Community Cloud. El checkpoint slim pasó de ~100MB a ~300MB al
  redimensionar a 80M, y compartir el límite de 1GB con las otras fases
  (`st.cache_resource` mantiene cada modelo visitado en memoria durante toda la
  vida del contenedor) arriesgaba tirar la app entera si alguien visitaba varias
  fases en la misma sesión — de ahí la app independiente.

  Configuración **congelada** en `compare_lab/config.py` (y duplicada,
  deliberadamente, en `compare_lab/models/cracked_d.py` y `sliced_d.py`, que son
  los que de verdad usa el entrenamiento — `config.py` es solo lo que comparan
  `eval/tables.py` y `tests/test_compare_lab_freeze.py`), verificada contra los
  checkpoints realmente entrenados por ese test: si algo cambia, Cracked-D se
  reentrena antes que los demás.

  Arnés de evaluación, k-fold y sección de README con los resultados finales:
  pendientes hasta que Sliced-D-80M termine.

  Notas de proceso de la Etapa 1, en la misma línea de verificar antes de concluir
  de las fases anteriores: (a) un ensayo de pausa/reanudación **con datos y GPU
  reales** destapó un bug fatal que ningún test de CPU podía ver — `map_location`
  movía el estado del RNG a GPU y *toda* reanudación reventaba; (b) el
  entrenamiento se hizo tolerante a fallos duros (checkpoint atómico con `fsync` +
  verificación + copia `.prev`, fichero `STOP` como única parada limpia, ver
  `compare_lab/PAUSAR_Y_REANUDAR.md`); (c) `pytest` estaba **destruyendo un
  deliverable** en silencio (la curva de loss del preentrenamiento) porque
  `_save_curves` escribía antes de comprobar si había datos y dos tests no
  redirigían `RESULTS_DIR` — de ahí el fixture `autouse` en `tests/conftest.py` y
  `compare_lab/train/regen_curves.py`, que reconstruye la curva desde el historial
  que llevan dentro los checkpoints.

  Existe además un `CONTEXTO_SESION.md` con el estado detallado de una sesión previa
  y el setup local del modelo de referencia
  [Ashx098/Mini-LLM](https://github.com/Ashx098/Mini-LLM) para comparar, pero es un
  fichero **local y no versionado** (está en `.gitignore`): no forma parte de la
  documentación del proyecto y no estará en un clon del repositorio.

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
coconut_lab/      Fase C: data/ (Alpaca, oasst1, GSM8K), models/ (cracked.py, sliced.py,
                  pressed.py + pressed_loop.py), eval/ (run_eval.py, run_domain_eval.py,
                  run_lm_eval.py, run_kfold.py, lm_eval_adapter.py, resultados en
                  eval/results/), logos.py (ASCII-art de Cracked/Sliced/Pressed)
compare_lab/      Fase D: config.py (configuración CONGELADA), data/ (tokenizer con ids
                  especiales distintos, prefix_lm.py con cortes y orden compartidos,
                  streaming.py con mezcla 50/50 por tokens), models/ (cracked_d,
                  sliced_d, attention GQA parametrizable), train/ (checkpoint.py
                  duradero, pretrain, finetune, run_pipeline, regen_curves),
                  verify/ (ensayo de pausa/reanudación con datos reales),
                  eval/results/, demo.py, export_for_demo.py, upload_to_hf.py
scripts/          Utilidades sueltas (p.ej. clean_vtt.py para los transcripts,
                  render_logo.py para rasterizar los logos ASCII-art a PNG)
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
- Cualquier gráfico generado (matplotlib u otro) guarda también los datos crudos que
  lo alimentan en un `.json` junto al `.png`, con el mismo nombre base (p.ej.
  `depth_lab/eval/results/accuracy_vs_depth.png` + `.json`) — así se puede regenerar o
  modificar el gráfico en el futuro sin tener que re-ejecutar todo el entrenamiento/
  evaluación que lo produjo.

## Referencias clave

- `docs/karpathy/resumen.md` — fundamentos (bigramas → MLP → BatchNorm → backprop
  manual → WaveNet → Transformer decoder-only completo).
- `docs/papers/Attention_is_all_you_need.pdf` — arquitectura encoder-decoder original.
- `docs/papers/Exploring_Depth_Generalization_..._Recursive_Logic_Tasks.pdf` — paper de
  Fase B: por qué los Transformers fallan en recursión profunda y el pipeline Looped
  Locate-and-Replace propuesto como mitigación.
