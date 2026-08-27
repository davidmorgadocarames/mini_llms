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

🌐 **[Prueba la demo interactiva en el navegador](https://davidmorgadocarames.github.io/mini_llms/)** (sin instalar nada).

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
en el paper. *(En construcción.)*

## Contexto y fundamentos

La base conceptual de este proyecto — desde bigramas hasta un Transformer completo —
está documentada en [`docs/karpathy/resumen.md`](docs/karpathy/resumen.md), a partir de
la serie "Neural Networks: Zero to Hero" de Andrej Karpathy.

## Estructura

Ver [`CLAUDE.md`](CLAUDE.md) para el mapa completo del repo y las convenciones del
proyecto.
