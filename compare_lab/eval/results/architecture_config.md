# Configuracion de las tres arquitecturas (Fase D)

| Modelo | Topologia | Capas | d_model | Cabezas | Cabezas KV | FFN | Contexto | Parametros | Params embedding | vs Cracked-D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cracked-D | decoder-only (causal) | 12 | 768 | 12 | 3 | 2,048 | 1024 | 80,628,480 | 6,291,456 | 0.0 |
| Cracked-D-full | decoder-only (causal), control | 12 | 768 | 12 | 3 | 2,048 | 1024 | 80,628,480 | 6,291,456 | 0.0 |
| Sliced-D | encoder-decoder (bidirectional enc + causal dec + cross-attn) | 7 enc + 4 dec | 768 | 12 | 3 | 2,048 | 1024 | 80,335,872 | 6,291,456 | -0.36 |

Tolerancia de igualdad de parametros: +-5%.

## Por que no se iguala tambien la profundidad

No se pueden igualar a la vez parametros, profundidad por token y anchura: las dos
topologias tienen formas estructuralmente distintas. T5 (Raffel et al. 2020, sec. 3.2.2)
tabula justamente ese compromiso: un encoder-decoder de L+L capas son 2P parametros a
M FLOPs, y uno de L/2+L/2 capas son P parametros a M/2 FLOPs, frente a un decoder-only
prefix LM de L capas con P parametros y M FLOPs. Wang et al. (2022, ICML) mantienen
identicos d_model, cabezas y FFN entre arquitecturas y varian solo el numero de capas,
igualando el **computo** y dejando que el encoder-decoder tenga ~2x parametros.

Esta fase iguala **parametros** (la restriccion del plan) preservando d_model=512 en
ambos modelos, que es el eje que la literatura mantiene fijo. La consecuencia, explicita
en la tabla, es que un token de continuacion atraviesa 4 capas en Sliced-D frente a 8 en
Cracked-D (aunque el camino prefijo->prediccion sea de 3+4=7 capas), y que Sliced-D hace
menos FLOPs por fragmento. Cualquier conclusion debe leerse con ese matiz.
