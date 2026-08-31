# Contexto de sesión — 2026-08-31

> Nota: este archivo es un volcado de contexto para retomar la conversación,
> no documentación del proyecto. Si no lo quieres en el repo de portfolio,
> añádelo a `.gitignore` o bórralo cuando ya no lo necesites.

## Estado actual, en una frase

Fase A/B/C, TUI y demo de Streamlit están terminadas, documentadas y
desplegadas. Lo que queda abierto es una investigación sobre por qué los
chatbots (Cracked/Sliced/Pressed) responden mal, con un plan "Fase D" **en
pausa, sin aprobar**, y una comparación pendiente contra el modelo de
referencia `Ashx098/Mini-LLM` corriendo en local.

## 1. Trabajo ya hecho y desplegado esta sesión

- Redeploy de Streamlit arreglado (había 9 commits sin `push`).
- Chat rediseñado con bocadillos estilo ChatGPT/Gemini
  (`pages/2_Fase_C_Coconut_Interactivo.py`).
- Banner de cada modelo: GIF (con transparencia arreglada frame a frame,
  `scripts/make_gifs_transparent.py`) + wordmark ASCII al lado
  (`coconut_lab/logos.py`, `scripts/render_logo.py`).
- Layout móvil arreglado: hueco muerto entre el chat y la barra de prompt
  (bug de `flex-basis` en `stLayoutWrapper` de Streamlit) + botón "Enviar".
- Todo commiteado y pusheado a `origin/master`.

## 2. Valoración honesta de portfolio (ya dada, no pendiente)

Los modelos en sí son mediocres (26M-62M params, datasets pequeños) pero el
**proceso de ingeniería es genuinamente fuerte**: arquitecturas construidas
desde cero, comparación de 3 diseños con eval real, bugs reales encontrados y
verificados (Xet de HF, infra-entrenamiento de LLR, BOS/pad compartido en
Sliced), decisiones documentadas con números. Consejo dado: enmarcar el
portfolio en torno al proceso/metodología, no prometer que el chatbot "habla
bien".

## 3. Investigación de por qué responden mal (ya hecha, resumen)

- **Chinchilla** (Hoffmann et al. 2022, ~20 tok/param): Cracked necesitaría
  ~528M tokens de entrenamiento óptimo; solo vio 328M (139M únicos, ~2.35
  épocas) → déficit real de ~4x, medido, no estimado.
- **WikiText-103 es prosa enciclopédica**: el modelo nunca vio una
  conversación en preentrenamiento, así que "no sabe conversar" también es
  un problema de dominio de datos, no solo de escala.
- **Sliced y el drafter de Pressed no tienen preentrenamiento en absoluto**
  (`coconut_lab/models/sliced.py:13-16`, `pressed.py:11`) — se entrenan desde
  cero solo sobre instrucciones. Eso explica su incoherencia peor que la de
  Cracked.
- **TinyStories** (Eldan & Li, 2023): prueba que 10-30M params bastan para
  inglés fluido *dentro de un dominio restringido* — fluidez ≠ inteligencia,
  no resuelve por sí solo el objetivo de "hablar con un humano".
- **LoRA/QLoRA**: técnicas de fine-tuning eficiente sobre una base grande ya
  preentrenada, NO una forma de añadir capacidad a un modelo de 26M propio.
  Solo tienen sentido si se parte de un modelo base real (Qwen2.5-0.5B,
  SmolLM2-360M, o 7B vía QLoRA).

## 4. Plan "Fase D" — EN PAUSA, no aprobado

Archivo completo: `C:\Users\Gatrix\.claude\plans\vamos-a-hacer-un-soft-finch.md`

Contenido (por si se retoma): D.0 probar el modelo de referencia antes de
gastar cómputo, D.1 arreglar sampling/repetition penalty (gratis, sin
reentrenar), D.2 reentrenar a escala de la referencia (80M params, 2B
tokens, ~30-40h estimadas sin medir), D.3 instruction-tuning encima, D.4
comparar contra una base preentrenada real con LoRA.

**Importante**: este plan se escribió por una mala lectura de scope mía — el
usuario dijo "planificalo" tras una lista de opciones y yo planifiqué todo
el Fase D completo cuando la pregunta real, dos veces, era mucho más
estrecha ("cómo pruebo el modelo de referencia en mi PC" / "solo quería un
resumen"). **No retomar este plan a gran escala sin que se pida
explícitamente por nombre.**

## 5. Modelo de referencia `Ashx098/Mini-LLM` — setup local

- El repo de HuggingFace (`huggingface.co/Ashx098/Mini-LLM`) **no es cargable
  con `transformers` estándar** (archivos en rutas no estándar, sin
  `modeling_*.py` pese a declarar `"architectures": ["MiniLLM"]`).
- El código real que funciona está en `github.com/Ashx098/Mini-LLM` (GitHub,
  no HF). Ya clonado en: `C:\Users\Gatrix\pc\proyectos ia\reference_mini_llm\`
  (fuera del repo `proyecto_llm_mini`, es un directorio hermano).
- Venv propio ahí (`--system-site-packages` para heredar el `torch` global),
  con `transformers`+`safetensors`+`sentencepiece`+`einops` instalados
  **solo dentro del venv** (limpiado del Python global a petición tuya).
- Checkpoint descargado en `out/ckpt.pt` (601MB).
- **Trampa real en el script**: `run_inference.py` tiene
  `--checkpoint` con default `out_production/ckpt.pt`, que NO es donde está
  el checkpoint. Hay que lanzarlo así:
  ```
  .\.venv\Scripts\python.exe run_inference.py --checkpoint out/ckpt.pt --prompt "Once upon a time" --max_tokens 60 --temp 0.8
  ```
- **Yo no puedo ejecutar este script**: el clasificador de seguridad de Auto
  Mode lo bloqueó 3 veces en la sesión (cargarlo con `trust_remote_code`,
  `git clone`, y `run_inference.py`) — bloqueo de entorno, no una decisión
  mía, no hay forma de rodearlo. **Tienes que lanzarlo tú** y pasarme la
  salida si quieres que la compare con Cracked/Sliced/Pressed.
- Config real verificada en los pesos (no en la tarjeta del modelo, que
  exagera): **62.3M params** (la tarjeta dice 80M), **contexto real de 128
  tokens** (la tarjeta dice 2048 — `rope.freqs_cis` es `[128, 32]`),
  `n_kv_heads == n_heads` (sin GQA, a diferencia de Coconut).

## 6. Pendiente sin confirmar

Ofrecí preparar un script en `proyecto_llm_mini` que genere con
Cracked/Sliced/Pressed sobre el mismo set de prompts, para poner sus
salidas al lado de las del modelo de referencia una vez tú lo lances. **No
lo he construido** — pídemelo explícitamente si lo quieres.

## 7. Auditoría de arquitectura (hecha hoy, sin cambios necesarios)

Verificado en las tres arquitecturas: optimizador AdamW unificado (con
`lr`/`weight_decay` configurables y weight decay solo en params `dim()>=2`),
QKV separado en heads, conexiones residuales, LayerNorm/RMSNorm etiquetada
con nombre propio, y tests dedicados de dimensiones de tensores de salida
(`tests/test_model.py`, `test_encoder_decoder_model.py`,
`test_locator_model.py`, `test_replacer_model.py`). Todo correcto, ningún
cambio pendiente de esta auditoría.
