# proyecto_llm_mini

Mini-LLM construido desde cero, en dos fases, como proyecto de portfolio.

## Fase A — Mini-LLM de propósito general

Un modelo de lenguaje decoder-only con arquitectura moderna (RoPE, RMSNorm, SwiGLU,
Grouped Query Attention), entrenado sobre texto real (WikiText-103). ~26M parámetros,
tokenizer BPE propio.

```bash
pip install -r requirements.txt

# 1. Descargar WikiText-103, entrenar el tokenizer BPE y binarizar los tokens
python -m mini_llm.data.prepare_data --dataset wikitext-103-raw-v1 --vocab-size 8192

# 2. Entrenar
python -m mini_llm.train.train --max-steps 20000 --batch-size 32

# 3. Generar texto
python -m mini_llm.inference.generate --prompt "The history of" --max-new-tokens 200
```

**Resultado**: 26.4M parámetros, entrenado 20.000 pasos (~3.3 épocas sobre 139M tokens)
en ~1.6h en una RTX 4060, `val_loss` final ≈ 2.95 (perplejidad ≈ 19). Ejemplo de
generación (`--prompt "In 1943, the"`):

> In 1943, the US and Germany would not be part of the new German states .
> = = = Final days = = =
> Following the death of Obersturm in July 1943 , the government decided to transfer
> the remainder of the Army and the military to the United States. [...]

El modelo aprendió gramática, puntuación, y hasta convenciones propias del formato
WikiText (como `@-@` para guiones) sin que se le indicara explícitamente.

### Coconut — TUI de agente (estilo Claude Code)

```bash
python coconut.py
```

Interfaz de terminal construida con [Textual](https://textual.textualize.io/): panel
de conversación (streaming token a token, Markdown vía Rich) + panel de actividad
con eventos plegables (`⏺ Leyendo README.md` → `✓`), diffs coloreados, indicador de
"trabajando" e input inferior — igual que Claude Code, pero para nuestro propio modelo.

Como Coconut es un **base model sin fine-tuning de instrucciones**, la propia UI lo
deja claro y sugiere prompts que sí funcionan (continuaciones de prosa, no preguntas).

Comandos disponibles:
- `/read <ruta>` — lee un fichero real del repo (con resaltado)
- `/test [ruta]` — ejecuta pytest de verdad y muestra el resultado
- `/diff <ruta>` — muestra el `git diff` real de un fichero, coloreado
- `/temp <valor>`, `/tokens <n>` — ajustan la generación
- `/help` — ayuda

**Arquitectura**: basada en eventos (Pydantic) para desacoplar el `Agent` de Textual —
`coconut_tui/agent.py`, `events.py`, `bus.py`, `providers/` y `tools/` no importan
Textual y se testean con pytest normal, sin necesitar una terminal real. El proveedor
del LLM (`coconut_tui/providers/`) es una interfaz intercambiable — cambiar de modelo
es implementarla de nuevo, sin tocar el resto.

### Demos en el navegador

- 🌐 **[Réplica estática (GitHub Pages)](https://davidmorgadocarames.github.io/mini_llms/)**
  — reproduce con animación de escritura generaciones reales pregrabadas del modelo.
  No hay servidor detrás, así que solo funciona con los prompts de ejemplo.
- 🚀 **[Inferencia real en vivo (Streamlit Community Cloud)](https://minillms-p2qhjk4tkphgwcw4yqfqks.streamlit.app)**
  — el modelo corriendo de verdad en un servidor, cargado desde el checkpoint publicado en
  [HuggingFace](https://huggingface.co/davidmorgado/coconut-mini-llm). Código en
  [`streamlit_app.py`](streamlit_app.py). La misma app tiene dos páginas más (menú lateral):
  **Fase B — Depth Lab**, descrita más abajo, y **Fase C — Coconut Interactivo**, con el
  chat multi-modelo también descrito más abajo.

## Fase B — ¿Por qué los Transformers fallan en razonamiento recursivo profundo?

Un Transformer estándar entrenado para evaluar expresiones anidadas como
`(3 + (2 * (5 - 1)))` aprende perfectamente durante el entrenamiento... y luego falla
sistemáticamente en cuanto la expresión tiene más niveles de anidación de los que vio
en entrenamiento — incluso si es más corta en longitud total. Esto es un hallazgo real,
publicado en AAAI 2026 ("Exploring Depth Generalization in Large Language Models for
Solving Recursive Logic Tasks", Zhiyuan He).

Este proyecto reproduce ese fallo, lo visualiza, y compara tres arquitecturas para ver
cuál lo mitiga mejor: un Transformer decoder-only estándar, un encoder-decoder clásico
(el de "Attention Is All You Need"), y el pipeline "Looped Locate-and-Replace" propuesto
en el paper.

Las tres arquitecturas se entrenan **solo** con profundidades 0-5 (`depth_lab/data/artifacts/bool_train.jsonl`)
y se evalúan con exact-match sobre profundidades 6-12, completamente fuera de distribución
(otro rango de profundidad, otras semillas) — nunca vistas ni usadas para elegir
hiperparámetros durante el entrenamiento. El split de validación (mismas profundidades que
train, semilla distinta) solo se usa para monitorizar el entrenamiento, nunca para la
comparación final.

![Accuracy vs. profundidad](depth_lab/eval/results/accuracy_vs_depth.png)

Resultado, con las tres arquitecturas ya entrenadas (`python -m depth_lab.eval.run_eval`):

| profundidad | decoder-only | encoder-decoder | LLR |
|---:|---:|---:|---:|
| 6  | 0.786 | 0.736 | **0.988** |
| 7  | 0.762 | 0.648 | **0.946** |
| 8  | 0.740 | 0.654 | **0.854** |
| 9  | 0.750 | 0.640 | **0.824** |
| 10 | **0.714** | 0.582 | 0.688 |
| 11 | **0.706** | 0.628 | 0.620 |
| 12 | **0.678** | 0.620 | 0.590 |

LLR domina claramente en profundidades moderadamente OOD (6-9), pero pierde su ventaja
frente al baseline en las más extremas (10-12) — un patrón más matizado que "LLR siempre
gana". La razón, verificada directamente (no es una suposición): LLR encadena una
predicción del locator por cada paréntesis a reducir, y ese número de pasos crece con la
profundidad (9.7 pasos de media en profundidad 6, frente a 21.8 en profundidad 12). Incluso
con una precisión por paso altísima, esa precisión también degrada ligeramente con la
profundidad (99.9% → 95.4%), y ambos efectos se componen multiplicativamente: una cadena de
~22 pasos al 95.4% cada uno ya no llega ni de lejos al 95.4% global. El baseline, en cambio,
produce la respuesta en una sola pasada — no tiene ese efecto de composición de errores. Es
un trade-off real del pipeline "locate-and-replace", no un fallo de implementación.

Reproducible con:

```bash
python -m depth_lab.data.build_dataset          # genera el dataset (train/val/test por profundidad)
python -m depth_lab.eval.run_eval               # entrena las 3 arquitecturas y genera el gráfico
```

### Demo interactiva — Depth Lab

La página **Fase B — Depth Lab** de la [app de Streamlit](https://minillms-p2qhjk4tkphgwcw4yqfqks.streamlit.app)
(`pages/1_Fase_B_Depth_Lab.py`) deja escribir o generar una expresión anidada a la
profundidad que se quiera y evaluarla en vivo con las tres arquitecturas a la vez,
incluyendo la traza paso a paso de la reducción de LLR animada (qué sub-expresión
localiza el locator en cada paso, y qué valor le asigna el replacer).

## Fase C — Coconut interactivo: 3 arquitecturas en razonamiento real

Fase A dejó a Coconut siendo un modelo *base*: solo completa texto al estilo Wikipedia,
no responde a instrucciones. Fase C lo lleva a terreno real en dos frentes: (1) fine-tuning
de instrucciones y chat (Alpaca + OpenAssistant/oasst1) para que responda de verdad, y
(2) repetir la comparación de 3 arquitecturas de Fase B — decoder-only, encoder-decoder,
y el pipeline Looped Locate-and-Replace — pero sobre razonamiento matemático en lenguaje
natural (GSM8K) en vez de una gramática sintética controlada.

Las 3 arquitecturas tienen nombre propio dentro del universo Coconut:

- **Cracked** — decoder-only, el propio checkpoint de Fase A afinado sobre instrucciones. El baseline.
- **Sliced** — encoder-decoder clásico, entrenado desde cero (sin el preentrenamiento de Fase A).
- **Pressed** — LLR adaptado: un *drafter* redacta un borrador (razonamiento o respuesta),
  y un locator+replacer (reutilizados literalmente de Fase B, cero cambios de arquitectura)
  recorren el borrador corrigiendo su aritmética paso a paso.

En GSM8K, la analogía con Fase B es directa: cada anotación `<<expr=resultado>>` que el
propio dataset ya incluye es, en la práctica, un "paso reducible" pre-etiquetado — el mismo
patrón locate-and-replace de las expresiones anidadas de Fase B, ahora en problemas de
matemáticas escritos en inglés.

### Resultado: comparación honesta en 5 piezas (C.6)

Un solo número de accuracy no basta para comparar 3 arquitecturas de forma justa,
especialmente en modelos tan pequeños (8-26M parámetros), donde diferencias de 1-2 puntos
suelen ser ruido. C.6 combina 5 piezas complementarias:

**1. Accuracy en GSM8K test por número de pasos de razonamiento** (test split oficial,
held-out, nunca visto en entrenamiento):

| pasos | Cracked | Sliced | Pressed |
|---:|---:|---:|---:|
| 0 | 0.0% | 0.0% | 0.0% |
| 1 | 0.0% | 0.0% | 0.0% |
| 2 | 2.5% | 0.0% | 0.0% |
| 3 | 2.5% | 2.5% | 0.0% |
| 4 | 0.0% | 2.5% | 0.0% |
| 5 | 2.5% | 0.0% | 0.0% |
| 6 | 0.0% | 0.0% | 5.0% |
| 7 | 0.0% | 0.0% | 0.0% |
| 8 | 0.0% | 0.0% | 11.1% |

Las 3 casi en el suelo. Esto **no** es un fallo del arnés de evaluación: se probó con dos
presupuestos de entrenamiento distintos (un primer pase más corto, y un segundo ~4x más
largo que bajó el loss de forma clara, de plateaus de 2.2-3.5 a 1.1-2.1) y el accuracy
exact-match apenas se movió entre ambos. A esta escala de parámetros, bajar el loss de
next-token-prediction no se traduce en acertar la respuesta numérica final — GSM8K real es
muchísimo más difícil que el dominio sintético de Fase B, donde el mismo patrón se aprendía
del todo en pocos miles de pasos.

**2. Set propio de dominio** (141 ejemplos curados a mano, 7 categorías — aritmética,
conversión de unidades, preguntas factuales, seguir instrucciones con palabra clave,
clasificación sí/no, seguir un formato pedido, explicaciones breves — cada uno con un
criterio de éxito programático explícito, no "parece bien"). Prompt y temperatura fijos,
**3 semillas**, media ± desviación típica:

| | Cracked | Sliced | Pressed |
|---|---:|---:|---:|
| **Accuracy media** | **21.5% ± 3.5%** | 18.2% ± 0.7% | 16.8% ± 1.7% |

Cracked lidera, pero el margen frente a Sliced/Pressed está dentro de 1-2 desviaciones
típicas — no un resultado aplastante. Por categoría no hay un ganador limpio: Sliced de
hecho lidera en "seguir instrucciones con palabra clave"; aritmética y conversión de
unidades siguen en el suelo para las 3, coherente con el punto 1.

**3. `lm-evaluation-harness`** (EleutherAI, el estándar de facto), vía un adaptador propio
(`coconut_lab/eval/lm_eval_adapter.py`) ya que ninguna de las 3 arquitecturas es compatible
con `transformers`. Dos tareas apropiadas para este tamaño de modelo: `lambada_openai`
(predicción de última palabra, sin necesitar conocimiento del mundo) y `piqa` (sentido
común físico, 2 opciones, azar = 50%):

| tarea | Cracked | Sliced | Pressed |
|---|---:|---:|---:|
| lambada_openai (acc) | **7.4%** | 0.0% | 4.4% |
| piqa (acc, azar=50%) | 54.8% | 53.6% | **55.5%** |

Las 3 superan el azar en piqa, muy cerca entre sí. Sliced falla del todo en
lambada_openai — tiene que comprimir todo el pasaje previo en una ventana de encoder fija
(384 tokens) en vez de atender directamente como un decoder-only.

**4. Eficiencia** (misma GPU, mismas condiciones):

| | parámetros | latencia (una generación) |
|---|---:|---:|
| Cracked | 26.4M | 0.13s |
| Sliced | 7.6M | 0.08s |
| Pressed (drafter+locator+replacer) | 15.9M | 0.03s |

**5. k-fold=5 de estabilidad de entrenamiento**: no mide generalización (eso ya lo hace el
punto 1), sino si el resultado depende de qué partición de GSM8K train le tocó ver al
modelo. 25 entrenamientos individuales (5 folds × 5 componentes), checkpointeados y
resumibles automáticamente ante un corte:

| | Cracked | Sliced | Pressed |
|---|---:|---:|---:|
| Accuracy media ± desv. típica entre folds | 1.6% ± 1.0% | 1.2% ± 1.2% | 1.8% ± 1.0% |

Las 3 casi en el suelo otra vez (mismo presupuesto de pasos más ligero que el punto 1). Las
desviaciones típicas son pequeñas, pero hay que reconocerlo con honestidad: con un accuracy
medio ya tan cerca de 0 hay poco margen real para que los números varíen — "estable" aquí
es una conclusión más débil que la que daría un checkpoint más entrenado.

### Un bug real encontrado por el camino

Durante el punto 3, un test de equivalencia detectó que Sliced generaba texto roto en
posición 0 (predecía la *segunda* palabra de la respuesta en vez de la primera,
consistentemente). Causa raíz: `coconut_lab/models/sliced.py` enmascaraba por error el
token BOS del decoder como si fuera padding — BOS y pad comparten id en el `BPETokenizer`
de este proyecto (a diferencia del `CharTokenizer` de Fase B, que los tiene separados), y
con máscara causal esa posición se quedaba sin ninguna clave válida a la que atender.
Afectaba a *todos* los checkpoints de Sliced entrenados hasta ese momento. Arreglado,
Sliced reentrenado desde cero, y los 3 puntos de C.6 que usaban su checkpoint re-ejecutados
con el checkpoint corregido — las cifras de arriba ya son las corregidas.

### Demo interactiva — Coconut Interactivo

La página **Fase C — Coconut Interactivo** de la
[app de Streamlit](https://minillms-p2qhjk4tkphgwcw4yqfqks.streamlit.app)
(`pages/2_Fase_C_Coconut_Interactivo.py`) deja chatear con cualquiera de los 3 modelos
(selector en popup) y cambiar de modelo a media conversación sin perder el historial — los
3 se entrenaron sobre el mismo formato de conversación, así que es una forma directa de
comparar cómo sigue cada arquitectura la misma charla.

Reproducible con:

```bash
python -m coconut_lab.data.prepare_instructions   # Alpaca
python -m coconut_lab.data.prepare_conversations   # oasst1
python -m coconut_lab.data.prepare_reasoning       # GSM8K
python -m coconut_lab.eval.run_eval                # entrena las 3 arquitecturas "final" + evalúa GSM8K + eficiencia
python -m coconut_lab.eval.build_domain_eval_set   # genera el set propio de 141 ejemplos
python -m coconut_lab.eval.run_domain_eval         # evalúa el set propio (3 semillas)
python -m coconut_lab.eval.run_lm_eval             # lm-evaluation-harness (lambada_openai + piqa)
python -m coconut_lab.eval.run_kfold               # k-fold=5 de estabilidad
```

## Contexto y fundamentos

La base conceptual de este proyecto — desde bigramas hasta un Transformer completo —
está documentada en [`docs/karpathy/resumen.md`](docs/karpathy/resumen.md), a partir de
la serie "Neural Networks: Zero to Hero" de Andrej Karpathy.

## Estructura

Ver [`CLAUDE.md`](CLAUDE.md) para el mapa completo del repo y las convenciones del
proyecto.
