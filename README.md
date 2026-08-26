# proyecto_llm_mini

Mini-LLM construido desde cero, en dos fases, como proyecto de portfolio.

## Fase A — Mini-LLM de propósito general

Un modelo de lenguaje decoder-only con arquitectura moderna (RoPE, RMSNorm, SwiGLU,
Grouped Query Attention), entrenado sobre texto real. *(En construcción.)*

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
