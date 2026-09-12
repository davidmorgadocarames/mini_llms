# Fase D — Comparación controlada decoder-only vs encoder-decoder con datos de calidad

## Contexto

Lee primero `CLAUDE.md` y `README.md`. El repo ya tiene la Fase A (mini-LLM decoder-only entrenado con WikiText-103), la Fase B (`depth_lab/`) y la Fase C (`coconut_lab/`, con Cracked, Sliced y Pressed).

La Fase C tiene dos problemas que la Fase D resuelve:

1. **La comparación no es justa.** Cracked parte del preentrenamiento de la Fase A y Sliced se entrena desde cero, con tamaños muy distintos (26,4M frente a 7,6M parámetros).
2. **Las respuestas son malas.** WikiText-103 es un corpus pequeño y poco adecuado para un asistente, y Alpaca y oasst1 son datasets heterogéneos para un modelo tan pequeño.

## Objetivos

1. **Comparación justa:** con los mismos datos, tokenizer, número de parámetros y presupuesto de entrenamiento, ¿rinde mejor un Transformer decoder-only o un encoder-decoder a una escala de unos 26M de parámetros?
2. **Calidad de respuesta:** que ambos modelos respondan lo mejor posible a su escala, usando datos de preentrenamiento y de fine-tuning pensados para modelos pequeños.

## Plan por etapas

La Fase D se hace en **dos etapas**. Al terminar la etapa 1, **detente, preséntame un resumen y espera a que te diga que continúes**. No empieces la etapa 2 por tu cuenta.

### Etapa 1: Cracked-D en Streamlit

1. Auditoría del fine-tuning de la Fase C (sección 0).
2. Preparación de datos y tokenizer (secciones 1 y 2).
3. Diseño de **las tres configuraciones**: Cracked-D, Sliced-D y Cracked-D-full (sección 3). Aunque solo se entrene Cracked-D, la configuración de Sliced-D debe quedar fijada con sus parámetros igualados.
4. Preentrenamiento y fine-tuning de **Cracked-D** (secciones 4 y 5). Sin k-fold ni harness.
5. Página **Fase D** en Streamlit solo con Cracked-D, desplegada en la nube y con medición de memoria (sección 9).

### Etapa 2: Sliced-D, control, evaluación y README

1. Preentrenamiento y fine-tuning de **Sliced-D** y de **Cracked-D-full** (secciones 4 y 5).
2. Evaluación con el harness de los tres modelos (sección 6).
3. k-fold de Cracked-D y Sliced-D (sección 7).
4. Muestras de respuestas (sección 8).
5. Añadir Sliced-D a la página de Streamlit, con cambio de modelo (sección 9).
6. Sección Fase D en el README y actualización de `CLAUDE.md` (sección 14).

### Regla de congelación

Al terminar la etapa 1 se congelan en un fichero de configuración versionado: el tokenizer, la muestra de datos y su orden, los puntos de corte del prefix LM, el conjunto de fine-tuning, los hiperparámetros y las tres configuraciones de arquitectura. **Si antes o durante la etapa 2 cambio cualquiera de estos elementos, Cracked-D debe reentrenarse con la configuración nueva antes de entrenar Sliced-D y Cracked-D-full.** Si detectas un cambio así, avísame antes de seguir.

## Alcance

**Incluido:**
- Un tokenizer nuevo, compartido por todos los modelos.
- Dos arquitecturas: **Cracked-D** (decoder-only) y **Sliced-D** (encoder-decoder), más la variante de control **Cracked-D-full**.
- Preentrenamiento con SmolLM-Corpus (Cosmopedia v2 y FineWeb-Edu-Dedup).
- Fine-tuning con smol-smoltalk.
- Evaluación de capacidades con `lm-evaluation-harness`.
- Evaluación de estabilidad con k-fold=5.
- Una página nueva **Fase D** en la app de Streamlit.

**Fuera de alcance:**
- Pressed / Looped Locate-and-Replace.
- GSM8K y el set propio de 141 ejemplos de la Fase C.
- WikiText-103, Alpaca y oasst1 como datos de entrenamiento.
- DPO u otras técnicas de preferencias.
- Cracked-D-full en la demo de Streamlit.

**No modifiques** el código ni los resultados de las Fases A, B y C. Si reutilizas módulos o recursos, impórtalos o cópialos sin cambiar su comportamiento. Todos los tests existentes deben seguir pasando.

## Estructura

Crea un paquete nuevo `compare_lab/` siguiendo el patrón de `coconut_lab/`: `data/`, `models/`, `train/` y `eval/` (con `eval/results/`). Guarda los checkpoints en un directorio propio, sin sobrescribir los de fases anteriores. Los datos y checkpoints no se suben a GitHub: añádelos a `.gitignore`.

## 0. Auditoría previa del fine-tuning de la Fase C (etapa 1)

Antes de implementar nada, revisa el pipeline de fine-tuning y generación de `coconut_lab/` **sin modificarlo** y documenta lo que encuentres en `compare_lab/AUDITORIA_FASE_C.md`:

- ¿La loss se calcula solo sobre las respuestas del asistente o también sobre el prompt?
- ¿Se añade el token EOS al final de cada respuesta y se incluye en la loss?
- ¿BOS, EOS y padding tienen ids distintos? ¿Se enmascara EOS por error al tratarlo como padding?
- ¿El formato de chat es idéntico en entrenamiento y en inferencia?
- ¿Se truncan respuestas a mitad?
- ¿La generación se detiene en EOS? ¿Qué temperatura y demás parámetros de muestreo usa?

Aplica todas las lecciones de esta auditoría a la Fase D.

## 1. Datos (etapa 1)

### Preentrenamiento

- Dataset: `HuggingFaceTB/smollm-corpus`, subsets `cosmopedia-v2` (libros de texto e historias sintéticas) y `fineweb-edu-dedup` (web educativa filtrada). Mezcla **50/50 en tokens**. No uses `python-edu`.
- **Usa solo una parte pequeña del corpus, nunca el completo.** Los dos subsets suman unos 248.000M de tokens y ocupan cientos de GB; la Fase D solo necesita una muestra fija.
- **Tamaño de la muestra:** orientativamente **unos 2.000M de tokens en total** (1.000M de cada subset), menos del 1% del corpus. Es la **misma muestra para todos los modelos**, no una por modelo. Antes de fijarla, mide el rendimiento real en tokens por segundo en la RTX 4060. **El preentrenamiento de los tres modelos no debe superar unas 24 horas de GPU en total**; si la estimación lo supera, propón en el plan una muestra más pequeña que encaje en ese límite. Debe ser configurable.
- **Cómo obtenerla:** usa `streaming=True`, lee documentos solo hasta alcanzar el tamaño fijado, tokenízalos sobre la marcha y escribe directamente los binarios. No descargues los ficheros del dataset completo ni guardes en disco todo el texto crudo.
- La muestra se prepara **una sola vez** con una semilla fija y se reutiliza en todos los entrenamientos, que leen solo los binarios locales y nunca de internet.
- Cada modelo recorre la muestra en **una sola época, sin repetir datos**.
- Reserva un conjunto de validación de la misma mezcla, del orden de 5M de tokens, separado de la muestra de entrenamiento.
- Guarda los datos tokenizados en binarios `uint16` (el vocabulario de 8192 cabe) e incluye en el plan una estimación del espacio en disco.

### Fine-tuning

- Dataset: `HuggingFaceTB/smol-smoltalk`, el subconjunto de SmolTalk preparado por Hugging Face para modelos de menos de 1B de parámetros. Revisa su formato de mensajes antes de procesarlo.
- Tras tokenizar, **descarta** las conversaciones que no quepan en el contexto, en lugar de truncarlas. Informa de cuántas quedan.
- Reserva un **split de test fijo** (del orden del 2%) que nunca se use en entrenamiento ni en el k-fold.
- Si el conjunto resultante es demasiado grande para el tiempo disponible, propón en el plan un número máximo de conversaciones.

## 2. Tokenizer (etapa 1)

- Entrena un único tokenizer BPE con la librería `tokenizers`, con vocabulario de 8192, sobre una muestra representativa de los datos de preentrenamiento y de smol-smoltalk (orientativo: 90/10). Lo comparten todos los modelos. Se entrena **antes** de generar los binarios, con una muestra pequeña obtenida también por streaming.
- Define tokens especiales con **ids distintos**: `<pad>`, `<bos>`, `<eos>` y los tokens de rol del formato de chat (sistema, usuario y asistente). Documenta la plantilla de chat.
- Informa de cuántos tokens por palabra produce el tokenizer nuevo frente al de la Fase A, sobre texto de preentrenamiento y sobre conversaciones.
- Construye las máscaras a partir de longitudes o de `attention_mask`, **nunca** comparando con el id de padding. Añade tests que verifiquen que `<bos>` nunca queda enmascarado y que `<eos>` nunca se trata como padding.

## 3. Arquitecturas (configuración en la etapa 1)

Ambas arquitecturas usan los mismos bloques, para que la única diferencia sea la topología:

- RoPE, RMSNorm (pre-norm), SwiGLU y GQA en la self-attention. Reutiliza `mini_llm.model` siempre que sea posible.
- **Cracked-D:** decoder-only con máscara causal.
- **Cracked-D-full:** arquitectura idéntica a Cracked-D; solo cambia la loss del preentrenamiento (sección 4).
- **Sliced-D:** encoder bidireccional y decoder causal con cross-attention. Aplica RoPE solo en las self-attention; la cross-attention va sin codificación posicional. Documenta esta decisión.
- **Parámetros igualados** a unos 26M, con una diferencia máxima del 5%. Ajusta capas y dimensiones de Sliced-D para conseguirlo, y verifica el conteo instanciando el modelo ya en la etapa 1.
- **Contexto:** propón una longitud en el plan (orientativa: 1024 tokens). La ventana del encoder de Sliced-D no puede ser menor que el contexto del decoder-only; en la Fase C, la ventana de 384 tokens del encoder penalizó a Sliced en LAMBADA.
- Documenta la configuración final en una tabla: capas, `d_model`, cabezas de atención, cabezas KV, dimensión FFN, longitud de contexto, parámetros totales y parámetros de embedding.

## 4. Preentrenamiento

**Cracked-D y Sliced-D se entrenan con el mismo objetivo: prefix LM.** Cada fragmento de texto se corta en un punto aleatorio en un prefijo y una continuación, y ambos modelos deben predecir la continuación token a token.

- **Cracked-D (etapa 1):** recibe prefijo + continuación como una sola secuencia con máscara causal. La loss se calcula **solo** sobre los tokens de la continuación. Registra además, sin usarla en el backward, la loss sobre los tokens del prefijo como diagnóstico.
- **Sliced-D (etapa 2):** el encoder lee el prefijo y el decoder predice la continuación.
- **Cracked-D-full (etapa 2):** un decoder-only idéntico a Cracked-D, pero con la loss sobre **todos** los tokens, prefijo incluido. Mide cuánto le cuesta al decoder-only renunciar a la señal del prefijo. Se entrena por separado y después recibe exactamente el mismo fine-tuning que Cracked-D. Documéntalo como experimento de control, separado de la comparación principal.
- **Mismos cortes y mismo orden:** genera el punto de corte de cada fragmento una sola vez con una semilla fija y guárdalo en la etapa 1. Los tres modelos ven exactamente los mismos pares prefijo/continuación en el mismo orden. Durante el entrenamiento de Cracked-D, registra un hash de cada batch en un fichero; los entrenamientos de la etapa 2 deben verificar sus batches contra ese registro y detenerse si no coinciden.
- **Iguala también:** optimizador, learning rate y schedule (warmup + cosine), batch efectivo, número de pasos, precisión mixta y semilla.
- Este objetivo es coherente con el resto del experimento: el fine-tuning (prompt → respuesta) y el harness (contexto → continuación) tienen la misma estructura.
- Guarda las curvas de loss de entrenamiento y validación.
- Una semilla de preentrenamiento por modelo; deja preparado el script para repetir con 3 semillas si hay tiempo.

## 5. Fine-tuning con smol-smoltalk

Cracked-D en la etapa 1; Sliced-D y Cracked-D-full en la etapa 2.

- La plantilla de chat es **idéntica** en entrenamiento, evaluación y demo.
- **Loss solo sobre los tokens del asistente**, en todos sus turnos, incluyendo el `<eos>` al final de cada respuesta.
  - **Cracked-D y Cracked-D-full:** la conversación va en una sola secuencia y se enmascara todo lo que no sea respuesta del asistente.
  - **Sliced-D:** el encoder recibe el historial de la conversación hasta ese turno y el decoder genera la respuesta del asistente.
  - Garantiza que todos calculan la loss sobre exactamente los mismos tokens del asistente, y documenta cómo empaquetas las conversaciones de varios turnos en cada arquitectura.
- Mismos hiperparámetros, mismo orden de datos y mismo número de pasos para los tres modelos.
- La generación se detiene en `<eos>` y usa los mismos parámetros de muestreo en todos los modelos.

## 6. Evaluación de capacidades con lm-evaluation-harness (etapa 2)

- Reutiliza o adapta `coconut_lab/eval/lm_eval_adapter.py`. Para Sliced-D, el contexto va al encoder y la continuación se puntúa o genera en el decoder.
- Añade un test de equivalencia: las log-probabilidades del adaptador deben coincidir con un cálculo manual en ambas arquitecturas.
- **Tras el preentrenamiento y tras el fine-tuning:** `lambada_openai`, `piqa`, `arc_easy`, `hellaswag` y `blimp`. Guarda los checkpoints preentrenados de la etapa 1 para poder evaluarlos aquí.
- **Solo tras el fine-tuning:** `ifeval`, que mide si el modelo cumple instrucciones y requiere generación, así que el adaptador debe implementar `generate_until` en ambas arquitecturas. A esta escala se espera un resultado bajo; documéntalo.
- Evalúa Cracked-D, Sliced-D y Cracked-D-full.
- Verifica que las tareas existen en la versión instalada y documenta el nivel de azar de cada una.
- Reporta la accuracy con el error estándar que proporciona el harness.

## 7. Evaluación de estabilidad con k-fold=5 (etapa 2)

- Divide el conjunto de fine-tuning, **sin el split de test fijo**, en 5 particiones por conversación completa.
- Para cada partición y arquitectura: haz fine-tuning desde el mismo checkpoint preentrenado con 4 particiones y mide loss y perplejidad sobre los tokens del asistente de la partición excluida.
- Son 10 entrenamientos (5 particiones × 2 arquitecturas: Cracked-D y Sliced-D). Cracked-D-full queda fuera del k-fold.
- Reporta media ± desviación típica por arquitectura.
- El k-fold y el harness son evaluaciones separadas: no mezcles sus resultados.

## 8. Muestras de respuestas (etapa 2)

- Guarda en un `.json` las respuestas de los tres modelos a los prompts del split de test fijo, con los mismos parámetros de generación.
- No se puntúan: sirven para inspeccionar la calidad de las respuestas.

## 9. Streamlit: página Fase D

### Etapa 1: Cracked-D

- Crea `pages/3_Fase_D_Comparacion_Controlada.py` **copiando la estructura de interfaz de la página de la Fase C** (`pages/2_Fase_C_Coconut_Interactivo.py`), con el mismo estilo que `streamlit_app.py` y el resto de páginas.
- Solo aparece **Cracked-D**, con un chat y un botón para resetear la conversación, igual que en la Fase C.
- Reutiliza el **GIF animado de Cracked de la Fase C** para Cracked-D.
- En la interfaz, los modelos se llaman **Cracked-D** y **Sliced-D**, para no confundirlos con los de la Fase C.
- La página debe avisar de que es un modelo pequeño, de unos 26M de parámetros, y de que sus respuestas serán limitadas.
- La lógica de carga y generación va en `compare_lab/`, sin importar Streamlit, para poder testearla con pytest. La página solo contiene la interfaz.

### Etapa 1: despliegue y prueba de memoria

El objetivo es comprobar si Streamlit Community Cloud soporta el modelo nuevo junto a los de las otras fases.

1. **Prueba local:** ejecuta la app con `streamlit run` y mide la memoria del proceso con los modelos de todas las páginas cargados.
2. **Límites de la nube:** consulta en la documentación oficial los límites actuales de recursos de Streamlit Community Cloud y compáralos con tu medición.
3. **Subida del checkpoint:** súbelo a HuggingFace del mismo modo que el de la Fase A, **tras pedirme permiso**. Si no hay sesión iniciada en HuggingFace, pídeme que la inicie yo; **nunca me pidas que pegue tokens o contraseñas en la conversación**.
4. **Push a GitHub:** haz commit solo de los ficheros de la Fase D y push a la rama que usa el despliegue, **tras pedirme permiso** y enseñándome antes la lista de ficheros. Verifica qué rama usa la app desplegada.
5. **Carga eficiente:** carga el modelo con `st.cache_resource` y solo cuando se abra la página.
6. **Informe:** dime la memoria medida en local, los límites de la nube y si la app desplegada funciona. Si no cabe, propón opciones (pesos en float16, descargar modelos que no se estén usando, etc.) y **espera mi decisión antes de aplicarlas**.

### Etapa 2: Sliced-D

- Añade **Sliced-D** a la misma página, con su **GIF animado de Sliced de la Fase C**.
- Replica el comportamiento de la Fase C: selector de modelo, cambio de modelo a mitad de conversación conservando el historial y botón para resetear la conversación.
- Sube el checkpoint de Sliced-D y haz push siguiendo los mismos pasos de permiso, y repite la prueba de memoria con los dos modelos.

## 10. Tareas largas: lanzamiento, progreso y reanudación

Son tareas largas: preparar la muestra de datos, los preentrenamientos, los fine-tunings, el k-fold y el harness completo. Voy a controlar esta sesión a menudo desde el móvil con Remote Control, así que todo debe poder gestionarse desde la propia conversación.

### Lanzamiento

- Prepara y prueba cada script con ejecuciones cortas. Cuando funcione, muéstrame el comando exacto, qué hace, su duración estimada y dónde guardará checkpoints, logs y estado, y **pide mi permiso explícito** antes de lanzarlo.
- Con mi permiso, lánzalo **como proceso independiente de Claude Code**: en Windows, con `Start-Process` en una ventana nueva de terminal, de forma que siga ejecutándose aunque esta sesión termine o se desconecte.
- Tras lanzarlo, comprueba que ha arrancado (el fichero de estado existe y avanza) y **termina tu turno**. No te quedes esperando ni revisando la salida periódicamente.
- La primera vez, recuérdame desactivar la suspensión de Windows mientras dure la tarea.

### Progreso

- Cada tarea genera tres salidas:
  - **Barra de progreso** en su propia ventana de terminal, para cuando esté en el PC.
  - **Log completo** en fichero.
  - **Fichero de estado** `compare_lab/runs/<nombre_tarea>/status.json`, actualizado al menos cada 30 segundos con escritura atómica (fichero temporal y renombrado).
- Campos de `status.json`: nombre de la tarea, modelo, estado (`running`, `paused`, `finished` o `error`), PID, paso actual y total, porcentaje completado, tokens procesados y totales, loss de entrenamiento reciente, última loss de validación, tokens por segundo, tiempo transcurrido, tiempo restante estimado, hora estimada de fin, último checkpoint (paso y hora), hora de la última actualización y mensaje de error si lo hay.
- Crea `python -m compare_lab.status`, que resume todas las tareas a partir de sus `status.json`: estado, progreso, tiempo restante y problemas detectados.
- **Cuando te pregunte cómo va algo**, ejecuta ese comando, lee las últimas líneas del log si hace falta y respóndeme con cifras exactas.
- **Detecta tareas caídas:** si una tarea figura como `running` pero su última actualización tiene más de 5 minutos o su PID ya no existe, avísame de que el proceso ha muerto y enséñame el final del log.

### Parada y reanudación

- Toda tarea larga debe poder pararse en cualquier momento y reanudarse **sin afectar al resultado final**.
- **Parada limpia:** al pulsar Ctrl+C en su ventana o al crear un fichero `STOP` en la carpeta de la tarea, guarda un checkpoint y termina con estado `paused`. Así también puedo pedirte que la pares desde el móvil.
- **Parada brusca** (corte de luz, cierre de la ventana): guarda checkpoints periódicos cada 15 minutos aproximadamente, con escritura atómica, para que como máximo se pierda el trabajo desde el último.
- **Contenido de cada checkpoint:** pesos, estado del optimizador, del scheduler y del escalador de precisión mixta, estados de los generadores aleatorios (Python, NumPy, PyTorch CPU y CUDA), paso actual y posición exacta en los datos.
- **Reanudar es relanzar el mismo comando:** detecta el último checkpoint válido y continúa con los mismos datos, en el mismo orden y con el mismo schedule, sin repetir ni saltarse batches.
- **k-fold y harness** se reanudan por unidad completada: las particiones terminadas y las tareas del harness ya evaluadas no se repiten.
- **Test de reanudación:** en una ejecución corta, compara un entrenamiento interrumpido y reanudado con otro sin interrupciones. Los batches deben ser idénticos y las curvas de loss deben coincidir dentro de una tolerancia pequeña; algunos kernels de GPU no son deterministas, así que no se exige igualdad bit a bit.

## 11. Restricciones del entorno y convenciones

- Windows, Python 3.13, PyTorch 2.7.1+cu118 y una RTX 4060 de 8GB: usa precisión mixta y acumulación de gradientes.
- Todos los scripts ejecutables por CLI (`python -m compare_lab....`).
- Antes de cada entrenamiento completo, verifica que el modelo puede sobreajustar un batch pequeño.
- **Resultados separados de su presentación.** Cada gráfico `.png` y cada tabla se guardan junto a un `.json` con los datos crudos y el mismo nombre base. Las tablas se guardan además renderizadas en Markdown.
- Crea un script `python -m compare_lab.eval.build_figures` que regenere **todas** las gráficas y tablas leyendo solo los `.json`, sin reentrenar ni reevaluar nada.
- Tests con pytest en `tests/` para: tokens especiales del tokenizer, máscaras, conteo de parámetros, construcción de pares prefix LM, que Cracked-D y Sliced-D calculan la loss sobre los mismos tokens, que los entrenamientos de la etapa 2 verifican sus batches contra el registro de hashes de Cracked-D, enmascarado de la loss en el fine-tuning (solo tokens del asistente y `<eos>` incluido), que un entrenamiento interrumpido y reanudado recibe los mismos batches que uno sin interrupciones y partición k-fold sin fugas entre particiones.

## 12. Modelo de Claude Code por bloque

Uso distintos modelos de Claude según el tipo de trabajo. El modelo lo cambio yo con `/model`, también desde el móvil con Remote Control.

- **Al empezar cada bloque de la tabla**, comprueba si el modelo recomendado coincide con el que estás usando. Si no coincide, o no puedes saberlo, **detente antes de empezar el bloque** y dime el comando exacto (por ejemplo, `/model sonnet`).
- **No me pidas cambios dentro de un mismo bloque:** cada cambio de modelo invalida la caché de la conversación y obliga a reprocesarla.
- **No recomiendes Fable.** Si en mi plan Fable consume créditos de uso, la confirmación no se puede responder desde el móvil y el turno se cancela. Si un problema supera a Opus, dímelo y lo decido yo desde el PC.
- Esfuerzo recomendado: `high`, el valor por defecto.

| Bloque | Pasos (sección 13) | Modelo | Motivo |
| :--- | :--- | :--- | :--- |
| Auditoría y plan | Etapa 1, pasos 1 y 2 | Opus | Decisiones de diseño y detección de fallos sutiles |
| Implementación del núcleo | Etapa 1, paso 3 | Opus | Máscaras, prefix LM, tokenizer y reanudación: un error aquí cuesta horas de GPU |
| Tareas largas | Etapa 1, paso 4 | Sonnet | Preparar comandos, lanzarlos y leer `status.json` |
| Streamlit, despliegue y cierre | Etapa 1, pasos 5 a 7 | Sonnet | Interfaz copiada de la Fase C y pasos mecánicos |
| Congelación e implementación pendiente | Etapa 2, pasos 1 y 2 | Opus | Equivalencia de log-probabilidades y generación en el encoder-decoder |
| Tareas largas | Etapa 2, paso 3 | Sonnet | Preparar comandos, lanzarlos y leer `status.json` |
| Análisis de resultados | Etapa 2, paso 4 | Opus | Interpretación estadística y limitaciones honestas |
| Sliced-D en Streamlit | Etapa 2, paso 5 | Sonnet | Replicar el comportamiento de la Fase C |
| README y `CLAUDE.md` | Etapa 2, paso 6 | Opus | Redacción de resultados y limitaciones |
| Depuración | Cuando algo falle o un resultado sea sospechoso | Opus | Diagnóstico de causas no evidentes |

## 13. Forma de trabajar

### Etapa 1

1. **Auditoría** (sección 0).
2. **Plan.** Explora el repo y preséntame un plan con: qué reutilizas y qué creas, las tres configuraciones con su conteo de parámetros, el rendimiento medido en tokens por segundo, el tamaño de la muestra de preentrenamiento y si los tres preentrenamientos caben en 24 horas, el tamaño final del conjunto de fine-tuning, el espacio en disco necesario, una estimación del tiempo de cada entrenamiento y cómo adaptarás la página de la Fase C. **Espera mi aprobación antes de pasar a la implementación.**
3. **Implementación con tests** y una ejecución corta de todo el pipeline de principio a fin, incluidas las piezas que usará Sliced-D.
4. **Tareas largas** de Cracked-D según la sección 10.
5. **Página de Streamlit, despliegue y prueba de memoria** (sección 9).
6. **Congelación** de la configuración (regla de congelación).
7. **Resumen y parada:** dime qué se ha hecho, el enlace a la app, la memoria medida, las curvas de loss de Cracked-D y cualquier problema. **Espera a que te diga que continúes.**

### Etapa 2

1. **Comprueba la congelación:** verifica que la configuración no ha cambiado desde la etapa 1. Si ha cambiado, avísame antes de hacer nada.
2. **Implementación pendiente con tests:** adaptador del harness para ambas arquitecturas (incluido `generate_until`), k-fold y muestras de respuestas (secciones 6, 7 y 8).
3. **Tareas largas** según la sección 10: entrenamientos de Sliced-D y Cracked-D-full, harness, k-fold y muestras de respuestas.
4. **Análisis de resultados:** tablas, gráficas e interpretación.
5. **Sliced-D en Streamlit**, despliegue y prueba de memoria (sección 9).
6. **README y `CLAUDE.md`** (sección 14).

### En ambas etapas

- **Verifica antes de concluir.** Si un resultado es sospechoso, por ejemplo una accuracy casi nula o respuestas que no terminan, investiga posibles causas (falta de entrenamiento, errores de máscara, EOS) antes de reportarlo, como se hizo en las Fases B y C.
- Cuando terminen las tareas largas, analizas los resultados en esta misma sesión.

## 14. Entregables

### Etapa 1

- `compare_lab/AUDITORIA_FASE_C.md` con los hallazgos de la sección 0.
- Tokenizer, muestra de datos, conjunto de fine-tuning y fichero de configuración congelado.
- Checkpoints preentrenado y con fine-tuning de Cracked-D, y registro de hashes de batches.
- El sistema de seguimiento de tareas largas: `status.json` por tarea, logs en fichero, parada con `STOP`, reanudación y el comando `python -m compare_lab.status`.
- Página **Fase D** con Cracked-D, desplegada, con el checkpoint publicado en HuggingFace y el informe de memoria.

### Etapa 2

- Checkpoints de Sliced-D y Cracked-D-full.
- Tablas en `compare_lab/eval/results/`, cada una en `.json` y en Markdown:
  - Configuración y equidad: parámetros, tokens procesados y tokens con loss.
  - Tokenizer: tokens por palabra frente al de la Fase A.
  - Harness antes y después del fine-tuning, con error estándar, de los tres modelos.
  - k-fold: media ± desviación típica de Cracked-D y Sliced-D.
- Gráficas, cada una con su `.json`: curvas de loss de los tres modelos y barras del harness con barras de error.
- Muestras de respuestas de los tres modelos en `.json`.
- El script `build_figures` que regenera todo lo anterior a partir de los `.json`.
- Página **Fase D** con Cracked-D y Sliced-D, desplegada, con el checkpoint de Sliced-D publicado en HuggingFace.
- Una sección **Fase D** en `README.md` con comandos de reproducción, enlace a la demo, resultados y limitaciones explicadas con honestidad (por ejemplo, una sola semilla de preentrenamiento o que el decoder-only no aprovecha la loss sobre el prefijo).
- Actualiza el roadmap y el mapa del repo en `CLAUDE.md`.
