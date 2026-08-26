# Resumen: serie "Neural Networks: Zero to Hero" (Andrej Karpathy)

Resumen técnico de los 6 vídeos, generado a partir de sus transcripts (`docs/karpathy/transcripts/`), como base conceptual para el proyecto `proyecto_llm_mini`.

## 1. The spelled-out intro to language modeling: building makemore

Introduce **makemore**, un modelo de lenguaje a nivel de carácter que aprende de un dataset de nombres (`names.txt`, ~32 000 nombres) a generar nuevos nombres plausibles.

- **Bigramas por conteo**: se cuenta cuántas veces cada carácter sigue a otro (matriz 27×27, 26 letras + token especial `.` para inicio/fin). Se normalizan las filas para obtener una distribución de probabilidad y se muestrea con `torch.multinomial`.
- **Evaluación**: la calidad del modelo se mide con **negative log-likelihood (NLL)** promedio — el objetivo de entrenamiento estándar en modelado de lenguaje. Probabilidad 0 (bigramas no vistos) da NLL infinito, lo que motiva el **model smoothing** (sumar un conteo falso a todas las celdas).
- **Reformulación como red neuronal**: el mismo bigrama se implementa como una única capa lineal sin sesgo ni no-linealidad. Los caracteres de entrada se codifican con **one-hot encoding**, se multiplican por una matriz de pesos `W` (`logits`), se exponencian (`counts = logits.exp()`) y se normalizan (`softmax`) para obtener probabilidades. Minimizar el NLL vía **descenso de gradiente** converge a (aproximadamente) el mismo resultado que el conteo explícito — esta equivalencia es la idea clave que conecta conteo estadístico clásico con redes neuronales.
- Introduce practicas de PyTorch que se usan en toda la serie: broadcasting, `torch.tensor`, indexado avanzado, `generator` para reproducibilidad.

## 2. Building makemore Part 2: MLP

Sustituye el bigrama (contexto de 1 carácter) por un **Multi-Layer Perceptron** siguiendo Bengio et al. 2003, porque el conteo explícito escala exponencialmente con el tamaño de contexto (27^n filas).

- **Embeddings**: cada carácter se mapea a un vector denso de baja dimensión (tabla de lookup `C`, aprendida). Indexar en la tabla de embeddings es matemáticamente equivalente a multiplicar un one-hot por una capa lineal, pero mucho más eficiente.
- **Arquitectura**: concatenar los embeddings de los N caracteres de contexto → capa oculta con `tanh` → capa de salida (logits) → `softmax` + NLL. Se usa `F.cross_entropy` en vez de implementar softmax+NLL a mano (más eficiente y numéricamente estable, resta el máximo de los logits para evitar overflow del `exp`).
- **Entrenamiento práctico**: minibatches (en vez de todo el dataset por paso) para acelerar cada iteración a costa de gradientes más ruidosos pero suficientes; búsqueda de **learning rate** óptimo barriendo exponencialmente y observando la curva pérdida-vs-lr; **learning rate decay** al final del entrenamiento.
- **Train/dev/test split** (80/10/10): entrenar parámetros en train, elegir hiperparámetros (tamaño de embedding, tamaño de capa oculta, regularización) en dev, evaluar una única vez en test. Permite diagnosticar **underfitting** (train ≈ dev, modelo demasiado pequeño) vs **overfitting** (train ≪ dev).

## 3. Building makemore Part 3: Activations & Gradients, BatchNorm

Analiza cómo la **inicialización** y la distribución de activaciones/gradientes afecta la entrenabilidad de redes más profundas — crítico antes de escalar a arquitecturas grandes como un Transformer.

- **Loss esperado en la inicialización**: con logits ~0, la salida debe ser aprox. uniforme, así que el NLL inicial esperado es `-log(1/vocab_size)`. Si el loss inicial real es mucho mayor, la red está "confidently wrong" y desperdicia las primeras iteraciones simplemente aplastando pesos demasiado grandes (aspecto de "hockey stick" en la curva de pérdida). Solución: escalar los pesos/bias de la última capa a valores pequeños al inicializar.
- **Saturación de `tanh`**: si las pre-activaciones son demasiado grandes, `tanh` satura en ±1, y su derivada local `1 - t²` se anula → gradientes muertos (**vanishing gradients**). Un histograma de activaciones cercanas a ±1 en toda una columna indica una **neurona muerta** que nunca aprenderá. Mismo problema aplica a sigmoid; ReLU sufre el análogo con la región plana negativa.
- **Inicialización de Kaiming/He**: para mantener activaciones con varianza ~1 a través de las capas, los pesos deben escalarse por `gain / sqrt(fan_in)`, donde `gain` depende de la no-linealidad (√2 para ReLU, 5/3 para tanh). Implementado en PyTorch como `nn.init.kaiming_normal_`.
- **Batch Normalization** (Ioffe & Szegedy, 2015): normaliza las pre-activaciones a media 0 / varianza 1 por batch, luego aplica una escala y desplazamiento aprendibles (`gamma`, `beta`). Mantiene medias/varianzas "running" (running mean/std) para usarlas en inferencia sobre ejemplos individuales. Efecto colateral: acopla los ejemplos de un mismo batch entre sí (actúa como regularizador vía "jitter", pero también es fuente de bugs sutiles). Alternativas modernas menos acopladas: LayerNorm, GroupNorm — LayerNorm es la que usa el Transformer.
- Las innovaciones modernas (residual connections, normalization layers, optimizadores como Adam) hacen que la inicialización precisa importe menos que hace ~10 años, pero entender el mecanismo sigue siendo clave para depurar redes profundas.

## 4. Building makemore Part 4: Becoming a Backprop Ninja

Ejercicio de **backpropagation manual** (sin `loss.backward()`) sobre el MLP + BatchNorm del vídeo anterior, para entender backprop como una "leaky abstraction" que puede fallar silenciosamente si no se comprende.

- Deriva a mano el gradiente de cada operación del forward pass: `cross_entropy`/softmax, multiplicación matricial (`dL/dA = dL/dD @ B.T`, `dL/dB = A.T @ dL/dD`), suma con broadcasting (implica sumar gradientes en la dimensión broadcasteada), `tanh`, y BatchNorm completo.
- Regla general reiterada: **broadcasting en forward ⇒ suma en backward** (y viceversa); una variable reusada varias veces acumula (suma) los gradientes de cada uso.
- Deriva la fórmula simplificada y numéricamente estable del gradiente de `cross_entropy` respecto a los logits directamente (evitando pasar por softmax paso a paso), y el gradiente cerrado de la capa BatchNorm — mucho más eficiente que retropropagar por cada sub-operación.
- Valor pedagógico: entender esto ayuda a depurar arquitecturas nuevas y a reconocer productivamente errores de forma/broadcasting.

## 5. Building makemore Part 5: Building a WaveNet

Sustituye el MLP "plano" (que aplasta todo el contexto en una sola capa) por una arquitectura **jerárquica tipo árbol**, inspirada en WaveNet (van den Oord et al., 2016), que fusiona el contexto progresivamente (2 caracteres → 4 → 8...) en vez de todos a la vez.

- **Refactor a estilo PyTorch**: introduce módulos reutilizables (`Linear`, `BatchNorm1d`, `Tanh`, `Embedding`, `Flatten`, `Sequential`) que imitan la API real de `torch.nn`, sentando las bases conceptuales de cómo se construyen redes complejas por capas componibles.
- `FlattenConsecutive(n)`: en vez de aplanar todo el contexto de golpe, agrupa de `n` en `n` caracteres consecutivos, permitiendo un tensor 3D `(batch, grupos, canales)` — la matmul de PyTorch opera sobre la última dimensión y trata las anteriores como dimensiones de batch, lo cual permite procesar todos los grupos en paralelo.
- **Bug real de BatchNorm con tensores 3D**: al recibir entradas `(B, T, C)`, hay que reducir sobre las dimensiones `(0, 1)` (no solo `0`), o si no se calculan estadísticas por posición temporal en vez de por canal — bug sutil corregido en el vídeo.
- Conecta directamente con **convoluciones causales dilatadas**: el mismo cálculo jerárquico puede implementarse de forma mucho más eficiente como convolución 1D deslizante en vez de llamadas independientes al modelo, reusando cómputo compartido entre posiciones — anticipa por qué las CNN son eficientes para secuencias.
- Mensaje meta: el flujo real de trabajo en deep learning es iterativo (leer documentación de PyTorch, hacer que las formas de los tensores encajen, prototipar en notebook) y **falta un harness experimental** (esto se resuelve en el vídeo siguiente con el script de entrenamiento del GPT).

## 6. Let's build GPT: from scratch, in code, spelled out (★ vídeo más relevante)

Construye un **Transformer decoder-only** (arquitectura equivalente a GPT-2) desde cero, entrenado a nivel de carácter sobre el dataset "tiny Shakespeare" (~1M caracteres). Es la culminación directa de toda la serie makemore.

**Preparación de datos**
- Tokenización a nivel de carácter (vocabulario de 65 símbolos) vs. tokenización por sub-palabras (BPE, usada por GPT real con ~50k tokens) — trade-off entre tamaño de vocabulario y longitud de secuencia.
- `block_size` (longitud de contexto): un chunk de `block_size+1` caracteres contiene `block_size` ejemplos de entrenamiento simultáneos (contexto de 1 hasta `block_size` caracteres), lo cual acostumbra al modelo a predecir con cualquier longitud de contexto hasta el máximo.
- Batching: tensores `(B, T)` de índices enteros, procesados en paralelo pero de forma independiente entre sí.

**Self-attention (el mecanismo central)**
- Truco matemático clave: promediar/agregar información del pasado de forma eficiente usando **multiplicación matricial con una matriz triangular inferior** (`torch.tril`) en vez de bucles explícitos — y aplicar `softmax` sobre una matriz de afinidades con `-inf` en las posiciones futuras para enmascarar (**causal masking**), en vez de promedio uniforme.
- Cada token emite tres vectores vía proyecciones lineales aprendidas: **query** ("qué busco"), **key** ("qué contengo"), **value** ("qué comunico si me encuentran interesante"). Las afinidades son `Q @ K.T`, escaladas por `1/sqrt(head_size)` (**scaled attention**) para evitar que el softmax se sature hacia vectores one-hot cuando la dimensión de la cabeza es grande.
- **Multi-head attention**: varias cabezas de atención en paralelo (cada una con `head_size` menor), concatenadas — permite capturar distintos tipos de relación entre tokens simultáneamente.
- Distinción **self-attention** (Q, K, V provienen todos de la misma secuencia) vs. **cross-attention** (Q de la secuencia actual, K/V de una fuente externa, p. ej. el encoder en traducción) — el GPT que se construye es *decoder-only*, sin cross-attention ni encoder, porque no hay nada que condicionar: solo se genera texto libremente.
- La atención no tiene noción de posición por sí sola (opera sobre un conjunto de vectores) — de ahí la necesidad de **positional embeddings** sumados a los token embeddings.

**Bloque Transformer completo**
- Cada bloque intercala **comunicación** (multi-head self-attention) y **cómputo** (feed-forward MLP por token, con expansión ×4 y no-linealidad ReLU/GELU).
- **Residual/skip connections** (`x = x + sublayer(x)`): crean una "autopista" de gradiente directa entrada→salida, crítica para poder entrenar redes profundas (muchos bloques apilados) sin que el gradiente se degrade.
- **LayerNorm** (pre-norm, aplicado antes de cada sub-capa en vez de después, a diferencia del paper original): normaliza cada token independientemente (no acopla ejemplos del batch como BatchNorm), evitando los problemas de acoplamiento vistos en el vídeo 3.
- **Dropout**: apaga aleatoriamente una fracción de activaciones en cada forward/backward para regularizar, añadido al escalar el modelo para evitar overfitting.

**Escalado y resultado**
- Partiendo de un bigrama simple (loss ~2.4) se añade atención de una cabeza (2.4→2.28), multi-head (→2.28... mejor), feed-forward (→2.24), bloques + residuales (→2.08), LayerNorm (→2.06), y finalmente escalar hiperparámetros (context length 256, embedding 384, 6 cabezas, 6 capas, dropout 0.2) baja el loss de validación hasta **1.48**, generando texto mucho más reconociblemente shakespeariano (aunque sin sentido semántico).
- Comparación directa con GPT-3 real: mismo diseño arquitectónico, pero ~10M parámetros / ~300K tokens de entrenamiento en el ejemplo del vídeo frente a 175B parámetros / 300B tokens en GPT-3 (~un millón de veces más grande).
- Explica que el resultado de este entrenamiento (pretraining) es solo un "completador de documentos", no un asistente: para llegar a algo tipo ChatGPT hace falta una etapa posterior de **fine-tuning** (SFT sobre pares pregunta-respuesta, luego un modelo de recompensa entrenado con preferencias humanas, y RLHF vía PPO) — etapa que el vídeo no implementa pero explica conceptualmente.

## Aplicación al proyecto mini-LLM

- **Base de datos y tokenizador**: seguir el enfoque de tokenización a nivel de carácter (vídeos 1 y 6) para simplicidad inicial; documentar como posible mejora futura pasar a BPE/subpalabras.
- **Arquitectura núcleo**: implementar directamente el Transformer decoder-only del vídeo 6 — embeddings de token + posición, bloques de (self-attention multi-head causal + feed-forward), residual connections, pre-LayerNorm, dropout, capa lineal final a logits.
- **Self-attention**: reutilizar el truco de `tril` + `masked_fill(-inf)` + `softmax` con escalado `1/sqrt(head_size)` para la atención causal; implementar primero una sola cabeza para verificar correctitud antes de generalizar a multi-head.
- **Inicialización**: aplicar los principios del vídeo 3 (Kaiming/He, evitar logits extremos al inicio) para que el loss inicial sea el esperado (`-log(1/vocab_size)`) y el entrenamiento no desperdicie iteraciones iniciales.
- **Entrenamiento**: usar `F.cross_entropy` directamente (no reinventar softmax+NLL), minibatches, `AdamW`, búsqueda de learning rate por barrido exponencial, y split train/val para monitorizar overfitting.
- **Depuración**: si se necesita optimizar rendimiento o entender bugs de forma/broadcasting, el ejercicio de backprop manual (vídeo 4) da el marco mental para razonar sobre gradientes en capas custom.
- **Generación**: implementar `generate()` de forma autoregresiva (muestreo con `torch.multinomial` sobre la distribución softmax del último token), recortando el contexto a `block_size` como en el vídeo 6.
- **Escalado incremental**: seguir la progresión validada empíricamente en el vídeo 6 (bigrama → atención → multi-head → feed-forward → bloques+residual → LayerNorm → escalar hiperparámetros) como plan de desarrollo iterativo con checkpoints de validación en cada paso.
- **Fuera de alcance inicial pero documentable como "next steps"**: fine-tuning tipo instrucción (SFT) y RLHF, mencionados solo conceptualmente en el vídeo 6, relevantes si el proyecto evoluciona hacia un asistente conversacional en vez de un generador de texto libre.
