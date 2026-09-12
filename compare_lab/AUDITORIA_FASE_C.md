# Auditoría del fine-tuning y la generación de la Fase C

Revisión de solo lectura del pipeline de `coconut_lab/` (sección 0 del plan de Fase D),
**sin modificar nada**. El objetivo es entender qué hace realmente el fine-tuning de
instrucciones/chat de la Fase C y su generación, para aplicar las lecciones a la Fase D
(`compare_lab/`). Cada punto cita el archivo y las líneas donde se observa.

## Resumen

La Fase C es correcta en lo esencial (la loss se calcula solo sobre las respuestas del
asistente, con el token de fin de turno incluido, y la generación se detiene en ese
token), pero arrastra un **diseño de tokenizer heredado de la Fase A donde BOS, EOS y
padding son el mismo id** (el de `<|endoftext|>`), lo que ya causó un bug real en Sliced,
y una **inconsistencia de plantilla de chat entre entrenamiento e inferencia**. Además,
el checkpoint no guarda el estado de los generadores aleatorios ni la posición en los
datos, así que una reanudación no reproduce el mismo orden de batches. La Fase D corrige
las tres cosas por diseño.

## 1. ¿La loss se calcula solo sobre las respuestas del asistente?

**Sí.** Se usa una máscara multiplicativa `y_mask` (no `ignore_index=-100`), pero el
efecto es el mismo: prompt e input quedan fuera del backward.

- `coconut_lab/models/cracked.py:75-78` — `masked_loss()` multiplica la cross-entropy
  por `y_mask` y normaliza por `y_mask.sum()`.
- `coconut_lab/data/loader.py:43-57` — `InstructionDataset.__getitem__` construye
  `is_response = [0]*len(prompt_ids) + [1]*len(response_ids) + [0]*pad_len`.
- `coconut_lab/data/loader.py:90-100` — `ConversationDataset._encode_turns` marca
  `is_response=1` en **todos** los turnos del asistente, no solo el último.

**Lección Fase D:** mantener este criterio (loss solo sobre el asistente, en todos los
turnos). Con el tokenizer nuevo de ids distintos se puede usar tanto una máscara
multiplicativa como `ignore_index`; el plan usa máscaras por longitud.

## 2. ¿Se añade el token de fin al final de cada respuesta y se incluye en la loss?

**Sí, ambas cosas.** El proyecto no tiene un `<eos>` dedicado; usa el id de
`<|endoftext|>` (EOT) como marcador de fin de turno, y ese id cae dentro del tramo
marcado como respuesta, así que entra en la loss.

- `coconut_lab/data/loader.py:35` — `response_ids = tokenizer.encode(ex["response"]) + [self.pad_id]`.
- `coconut_lab/data/loader.py:96-97` — en `ConversationDataset`, `turn_ids + [self.pad_id]`
  solo para turnos `assistant` ("EOT marks the end of this assistant turn").

**Lección Fase D:** incluir `<eos>` al final de cada respuesta del asistente y contarlo
en la loss, para que el modelo aprenda a **parar**. En Fase D `<eos>` es un token con id
propio, no compartido con padding.

## 3. ¿BOS, EOS y padding tienen ids distintos?

**No: los tres comparten el id de `<|endoftext|>`** (el único token especial que se
entrenó en el BPE de la Fase A). Este es el hallazgo más importante.

- `mini_llm/tokenizer/bpe.py` — `EOT_TOKEN = "<|endoftext|>"` es el único token especial;
  `BPETokenizer` no expone `bos_id`/`eos_id`/`pad_id`.
- `coconut_lab/data/loader.py:10-14` (docstring) — "No dedicated pad token exists in Fase
  A's BPE vocab ... reusing EOT's id as padding".
- `coconut_lab/data/loader.py:28,74,140,176` — `self.pad_id = tokenizer.encode(EOT_TOKEN)[0]`
  en `InstructionDataset`, `ConversationDataset`, `Seq2SeqDataset`, `PressedLocatorDataset`.
- `coconut_lab/data/loader.py:131-134` (docstring de `Seq2SeqDataset`) — "EOT_TOKEN's id
  doubles as pad, bos, and eos here ... position encoding and context are what let the
  model tell them apart, not the token id alone".

**El bug histórico que provocó:** en `coconut_lab/models/sliced.py:121-130`, la máscara de
padding del decoder `tgt_pad_mask = tgt_in == pad_id` clasificaba la posición 0 (el BOS
real, prepended por `Seq2SeqDataset`) como padding, porque BOS y pad comparten id —
dejando la primera query del decoder sin ninguna clave válida bajo máscara causal +
padding. El parche es la línea `tgt_pad_mask[:, 0] = False`. Contraste: el `CharTokenizer`
de la Fase B (`depth_lab/tokenizer.py`) **sí** tiene `pad_id`/`bos_id`/`eos_id` separados,
por eso allí no ocurría.

**Lección Fase D (crítica):** el tokenizer nuevo define `<pad>`, `<bos>`, `<eos>` (y los
tokens de rol) con **ids distintos**, y las máscaras se construyen **por longitud /
`attention_mask`, nunca comparando con el id de padding**. Tests dedicados verifican que
`<bos>` nunca se enmascara y que `<eos>` nunca se trata como padding.

## 4. ¿El formato de chat es idéntico en entrenamiento y en inferencia?

**No.** Hay dos problemas.

1. **Dos plantillas conviven en el mismo modelo.** Cracked se entrena sobre un
   `ConcatDataset` de Alpaca + oasst1 (`coconut_lab/models/cracked.py:193`), con formatos
   distintos:
   - Alpaca: `"### Instruction:\n{...}\n\n### Response:\n"` sin tokens de rol
     (`coconut_lab/data/prepare_instructions.py:26-34`).
   - oasst1: marcadores `<|user|>\n` / `<|assistant|>\n`, sin rol de sistema
     (`coconut_lab/data/prepare_conversations.py:27-28,105-110`).
2. **La demo solo usa la plantilla oasst1** para los tres modelos
   (`pages/2_Fase_C_Coconut_Interactivo.py:233-246`, `build_prompt`), que coincide con
   cómo se entrenó `ConversationDataset` pero **no** con Alpaca.
3. **Mismatch extra en Sliced:** en entrenamiento, `conversations_to_seq2seq`
   (`coconut_lab/models/sliced.py:56-65`) pone en el encoder `format_turns(turns[:-1])`
   **sin** `<|assistant|>` final; pero la demo (`build_prompt`) **sí** añade
   `<|assistant|>\n` al `src` del encoder. El encoder ve en inferencia un token final que
   nunca vio en esa posición durante el entrenamiento.

**Lección Fase D:** una **única plantilla de chat**, idéntica en entrenamiento,
evaluación y demo, con tokens de rol dedicados (`<|system|>`, `<|user|>`, `<|assistant|>`)
y documentada en un solo sitio.

## 5. ¿Se truncan respuestas a mitad?

Depende del dataset; el criterio no es uniforme.

- `InstructionDataset` (Alpaca): si el prompt solo ya no cabe, **descarta** el ejemplo
  (`coconut_lab/data/loader.py:33-34`); si sobra respuesta, la **trunca por el final**
  (`loader.py:36-37`).
- `ConversationDataset` (oasst1): **trunca por el principio** (`ids[-block_size:]`,
  `loader.py:82-84`); si tras truncar no queda respuesta, descarta.
- `Seq2SeqDataset` (Sliced): **descarta** el ejemplo entero, nunca trunca
  (`loader.py:146-148`).

**Lección Fase D:** en el fine-tuning, **descartar** (no truncar) las conversaciones que
no quepan en el contexto e informar de cuántas quedan, para no entrenar sobre respuestas
cortadas a mitad. (En el preentrenamiento prefix-LM sí se trocea el texto continuo, que
es otra cosa.)

## 6. ¿La generación se detiene en el fin de secuencia? ¿Con qué muestreo?

**Sí, se detiene.** Parámetros:

- Cracked (`coconut_lab/models/cracked.py:140-153`): rompe en `next_id == eot_id`;
  defaults `temperature=0.8, top_k=50`.
- Sliced (`coconut_lab/models/sliced.py:152-170`): rompe en `next_id == pad_id` (mismo id
  que EOT); default `temperature=0.0` (greedy).
- **No hay top-p ni repetition penalty** en ningún generador del repo: `GPT._sample`
  (`mini_llm/model/transformer.py:99-106`) y `EncoderDecoderTransformer._sample`
  (`depth_lab/models/encoder_decoder.py:234-241`) solo soportan `temperature` y `top_k`.

**Lección Fase D:** parar en `<eos>` y usar los **mismos parámetros de muestreo** en los
tres modelos (para que la comparación sea justa). Si se quisieran top-p / repetition
penalty habría que añadirlos; no es necesario para la Etapa 1.

## 7. Hallazgos adicionales (infra), relevantes para Fase D

- **El checkpoint no guarda estado aleatorio ni posición en los datos.**
  `mini_llm/train/checkpoint.py:19-36` guarda solo pesos + optimizador + paso + config
  (escritura atómica). Al reanudar, el `DataLoader` se recrea con `shuffle=True`
  (`coconut_lab/models/cracked.py:108`, `sliced.py:108`), así que el orden de batches
  tras un corte **no** se reproduce. La Fase D exige reanudación reproducible, así que
  extiende el checkpoint con estados RNG (Python/NumPy/torch CPU+CUDA), scheduler, scaler
  y posición exacta en los datos.
- **No hay medición de memoria de proceso reutilizable.** El docstring de
  `coconut_lab/eval/run_eval.py:10-11` dice que `measure_efficiency` mide "memory", pero la
  función (`run_eval.py:235-260`) solo captura params y latencia. La prueba de memoria del
  deploy de la Fase D (Streamlit Cloud) se construye de cero (p. ej. con `psutil`).
- **`lm-eval` es dependencia fantasma:** se importa en `coconut_lab/eval/` pero no está en
  `requirements.txt`. La Fase D lo añadirá cuando toque el harness (Etapa 2).

## Checklist de lecciones aplicadas a la Fase D

- [x] Tokenizer con `<pad>`, `<bos>`, `<eos>` y tokens de rol de **ids distintos**.
- [x] Máscaras **por longitud / `attention_mask`**, nunca por comparación con pad id;
      tests de que `<bos>` no se enmascara y `<eos>` no se trata como padding.
- [x] Loss solo sobre tokens del asistente, con `<eos>` incluido, en todos los turnos.
- [x] **Una única plantilla de chat** idéntica en train/eval/demo.
- [x] **Descartar** (no truncar) conversaciones que no quepan en el contexto.
- [x] Generación que para en `<eos>` con los mismos parámetros de muestreo en todos.
- [x] Checkpoint con estado RNG + posición de datos para reanudación reproducible.
- [x] Medición de memoria de proceso propia para la prueba de deploy.
