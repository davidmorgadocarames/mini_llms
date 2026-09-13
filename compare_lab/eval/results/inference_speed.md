# Velocidad de inferencia: Fase A vs Fase D (Cracked-D)

Medido en NVIDIA GeForce RTX 4060, 3 prompts x 3 semillas, 200 tokens generados por corrida (tras 32 tokens de warmup).

| Modelo | Contexto | Parametros | tok/s (KV cache) | tok/s (sin cache) |
| --- | --- | --- | --- | --- |
| Fase A | 512 | 26,354,176 | 321.0 +/- 3.6 | 299.3 +/- 4.3 |
| Fase D - Cracked-D | 1024 | 80,628,480 | 215.1 +/- 1.8 | 172.9 +/- 0.8 |

Ambos modelos comparten literalmente la misma clase `GPT` y el mismo `generate_stream` (mini_llm/model/transformer.py); la unica diferencia de codigo entre las dos filas es la config (block_size, vocab) y los pesos entrenados, no una ruta de inferencia distinta.