# Pausar y reanudar el entrenamiento de Fase D

Guía práctica para cuando quieras usar el PC (jugar, trabajar en otro proyecto,
lo que sea) mientras el pipeline de `compare_lab` está corriendo, y luego
retomarlo exactamente donde se quedó.

## Resumen rápido

1. **Para pausar:** crea un fichero vacío llamado `STOP` en la carpeta de la
   etapa que esté corriendo ahora mismo (te lo dice `python -m compare_lab.status`).
2. **Para reanudar:** vuelve a ejecutar exactamente el mismo comando con el que
   lo lanzaste la primera vez. No hace falta ningún flag especial de "resume".
3. **El tiempo de trabajo real** (sin contar lo que dure la pausa) se guarda en
   `compare_lab/runs/<tarea>/status.json`, campo **`active_training_seconds`**.

## 0. Los dos tipos de parada (importante entender la diferencia)

No son lo mismo, y **solo uno de los dos no pierde nada**.

### Parada limpia — no se pierde NADA
La provocas tú creando el fichero `STOP`. **Es el único mecanismo de parada
voluntaria**: el manejo de `Ctrl+C` se eliminó a propósito (ver más abajo). No es
periódica, es **inmediata y bajo demanda**. Se guarda un checkpoint en ese momento, en un
punto perfectamente consistente, y no se pierde ni un paso de entrenamiento.

**Por qué el checkpoint es siempre "de calidad":** la comprobación del `STOP` y
la escritura del checkpoint viven *dentro* del bloque que se ejecuta solo cuando
se ha completado un paso de optimizador entero. Es decir: si pides parar mientras
va por el micro-lote 7 de 16, el entrenamiento **termina los 16**, aplica los
gradientes, los pone a cero, y *entonces* guarda y para. **Es imposible que se
guarde un checkpoint en medio de un forward o de un backward.** Por eso la pausa
tarda unos segundos (hasta 16 micro-lotes ≈ 5s, más ~2s de escribir 300 MB):
está terminando el trabajo en curso a propósito.

### Fallo duro — se pierde el trabajo desde el último checkpoint periódico (≤5 min)
Corte de luz, pantallazo azul, **cerrar la ventana**, o matar el proceso. Aquí no
hay ninguna oportunidad de guardar nada: el proceso desaparece donde esté.

La protección es el **checkpoint periódico** (cada ~5 min), que *también* se
escribió en una frontera limpia. Así que el checkpoint del que reanudas nunca
está tomado a mitad de un cálculo: lo que el fallo duro cuesta es **tiempo**, no
integridad.

Qué se pierde exactamente:

| | Parada limpia | Fallo duro |
| --- | --- | --- |
| Pasos de entrenamiento | nada | los de ≤5 min (se repiten) |
| Pesos y estado de Adam | nada | vuelven a hace ≤5 min |
| Puntos de la curva de loss | nada | los de ese intervalo (se recalculan) |
| `active_seconds` | exacto | no cuenta el trabajo perdido (correcto) |
| `batch_hashes.txt` | coincide con el checkpoint | queda **por delante**; al reanudar se revisitan esos lotes, se **verifica** que los hashes coinciden y se continúa (probado) |
| `status.json` | dice `paused` | se queda con datos viejos; `compare_lab.status` lo detecta como `DEAD(process gone)` por el PID |
| Integridad del checkpoint | intacta | **garantizada**: se escribe a `.tmp`, se fuerza a disco con `fsync`, **se recarga y verifica** antes de promoverlo, y se conserva la generación anterior como `.prev`. Si hay un checkpoint, está completo y carga |

**Ojo con cerrar la ventana:** en Windows eso es un **fallo duro**, no una pausa
limpia. El sistema manda `CTRL_CLOSE_EVENT`, da unos
[5 segundos](https://learn.microsoft.com/en-au/windows/console/ctrl-close-signal)
y mata el proceso; Python no lo recibe como señal, así que no hay forma de
guardar nada. **Para pausar sin coste usa el fichero `STOP`, no la X de la ventana.**

**El WiFi no afecta al entrenamiento.** Los datos se leen de binarios locales,
no de internet. Perder WiFi solo importa durante la *preparación de datos*, que
además es la única etapa no reanudable a medias.

### Por qué `Ctrl+C` ya no es una parada limpia
`Ctrl+C` **mata el proceso** como cualquier otro fallo duro: pierdes hasta 5
minutos. Es deliberado. Antes existían dos caminos de apagado y el elegante,
al usarse poco, se pudrió: acumuló varios bugs reales (los manejadores de señal
se restauraban antes del guardado final, un segundo `Ctrl+C` impaciente lo tiraba
a la basura, y no había forma de salir si una escritura se atascaba). Un solo
camino que se ejercita siempre es más fiable que dos, uno de los cuales se pudre
— es la idea de [*Crash-Only Software*, Candea & Fox, HotOS 2003](https://www.usenix.org/legacy/events/hotos03/tech/full_papers/candea/candea.pdf).

**Para pausar sin perder nada, usa el fichero `STOP`.**

## 1. Cómo pausar

El pipeline corre como un proceso normal en su propia ventana de terminal
(`Start-Process`), independiente de esta conversación. **Hay una sola forma de
pararlo sin perder trabajo**: crear un fichero vacío `STOP` en la carpeta de la
etapa activa (te dice cuál `python -m compare_lab.status`):

```powershell
New-Item -ItemType File "compare_lab\runs\pretrain_cracked\STOP"
```

Funciona igual desde el PC que desde el móvil. El proceso comprueba ese fichero
en cada paso de optimización y, en cuanto lo ve, para.

Antes de terminar:

- Guarda un **checkpoint completo** (pesos, optimizador, escalador de
  precisión mixta, estado de los generadores aleatorios de Python/NumPy/PyTorch,
  y la posición exacta en los datos).
- Marca el estado como `"paused"` en `status.json`.
- El fichero `STOP` se borra automáticamente la próxima vez que relances el
  comando, así que no hay que limpiarlo a mano.

**No hace falta pausar para usar el PC en general** (navegar, programar, ver
vídeos): el entrenamiento corre en su propia ventana sin bloquear nada más.
Sí conviene pausarlo si vas a hacer algo que también use la GPU a fondo
(otro entrenamiento, renderizado, un juego exigente), porque competirían por
la misma VRAM — el pico medido del entrenamiento de Fase D es de 3-5 GB en una
RTX 4060 de 8 GB, así que hay margen para cosas ligeras, pero no para otra
carga pesada de GPU a la vez.

## 2. Cómo reanudar

Vuelve a ejecutar el mismo comando con el que lo lanzaste, por ejemplo:

```powershell
python -m compare_lab.train.run_pipeline --arch cracked
```

El pipeline detecta automáticamente, etapa por etapa:

- Si una etapa de **preparación de datos** ya tiene su `stats.json` escrito,
  la salta entera (no vuelve a descargar ni tokenizar).
- Si una etapa de **entrenamiento** (preentrenamiento o fine-tuning) ya tiene
  su checkpoint final (`pretrain_final.pt` / `finetune_final.pt`), la salta.
- Si una etapa de entrenamiento se quedó **a medias** (pausada), carga su
  último checkpoint y continúa desde el mismo paso, con los mismos datos en el
  mismo orden y el mismo estado aleatorio — no repite ni se salta ningún lote.
  Esto está verificado con un test (`test_interrupted_resume_matches_uninterrupted`):
  un entrenamiento pausado y reanudado produce los mismos pesos que uno que
  nunca se para.

No hay ningún paso manual más: relanzar el comando es reanudar.

## 3. Dónde se guarda el tiempo de trabajo (sin contar la pausa)

Cada etapa larga escribe su progreso en:

```
compare_lab/runs/<nombre_tarea>/status.json
compare_lab/runs/<nombre_tarea>/<nombre_tarea>.log
```

(`<nombre_tarea>` es, por ejemplo, `pretrain_cracked`, `finetune_cracked`,
`data_prep_pretrain`, `data_prep_finetune`, o `pipeline_cracked` para el
resumen del conjunto).

`status.json` tiene **dos campos de tiempo distintos** — la distinción importa,
porque son la respuesta a tu pregunta:

- **`active_training_seconds`** — el que quieres. Es el tiempo **acumulado**
  que el proceso ha estado realmente corriendo, sumando todas las tandas
  (antes y después de cada pausa). El tiempo en que no hay ningún proceso
  vivo (mientras está pausado) **no cuenta**, porque nada avanza este número
  durante ese rato: solo se actualiza mientras el bucle de entrenamiento está
  activo.
- `elapsed_seconds` — tiempo desde que **esta tanda concreta** arrancó (se
  reinicia a 0 cada vez que relanzas el comando). Útil para ver "cuánto lleva
  corriendo desde el último relanzamiento", pero no es el total.

Este mismo valor (`active_seconds`) se guarda también **dentro de cada
checkpoint** (`compare_lab/checkpoints/<arquitectura>/*.pt`), que es la fuente
de verdad: al reanudar, el script lo lee del checkpoint y sigue sumando desde
ahí, así que `active_training_seconds` es correcto incluso si status.json de
una tanda antigua ya no existiera.

**Nota sobre la frescura del dato:** `status.json` se escribe como mucho cada
~30s mientras el entrenamiento está `"running"` (para no golpear el disco en
cada paso), así que si lo consultas en ese momento puede tener hasta 30s de
antigüedad. En el instante de **pausar, terminar, evaluar o hacer un
checkpoint**, la escritura se fuerza de inmediato — por eso, justo después de
crear un `STOP`, el `status.json` final coincide exactamente (mismo paso) con
el checkpoint guardado. Verificado lanzando el pipeline real como proceso
independiente, creando `STOP` a mitad de entrenamiento y de fine-tuning: en
ambos casos el proceso terminó en menos de 1.2s, con checkpoint, `status.json`
y `.log` consistentes entre sí.

## Ficheros que pueden quedar tras un fallo duro, y qué hacer

Ninguno impide reanudar, pero conviene saber qué son si los ves:

| Fichero | Qué significa | Qué hacer |
| --- | --- | --- |
| `runs/<tarea>/LOCK` | el candado del proceso que murió | **nada**: al relanzar se detecta que el PID está muerto (comparando también la hora de arranque, por si el PID se reutilizó) y se reutiliza automáticamente |
| `runs/<tarea>/STOP` | el STOP que provocó la pausa | **nada**: se borra solo al relanzar |
| `checkpoints/<arch>/*.pt.tmp` | un checkpoint completo cuyo renombrado quedó bloqueado (antivirus, o lo tenías abierto) | el log lo avisa. El checkpoint anterior es válido, así que puedes ignorarlo; si quieres recuperar ese trabajo, renómbralo quitando el `.tmp` antes de reanudar |
| `stages.json` con `error` | una etapa falló | mira el `error` de `status.json` y el final del log |

### Consultarlo rápido

```powershell
python -m compare_lab.status
```

Te da progreso, tokens/s, pérdida reciente y ETA de todas las tareas a la vez.
Para ver el tiempo de trabajo acumulado de una tarea concreta:

```powershell
python -c "import json; print(json.load(open('compare_lab/runs/pretrain_cracked/status.json'))['active_training_seconds'] / 3600, 'horas')"
```

## 4. Cosas que conviene saber (aprendidas verificando esto en GPU real)

- **No lances dos veces la misma tarea.** Hay un fichero `LOCK` por tarea: si
  relanzas mientras la anterior sigue viva, el proceso nuevo se niega a arrancar
  con un mensaje claro. Antes de ese candado, dos procesos simultáneos
  corrompían el registro de hashes de lotes de forma **silenciosa**, y el daño
  solo aparecía horas después, al entrenar otra arquitectura, disfrazado de
  "config drift". Si la tarea anterior murió, el candado se detecta como obsoleto
  y se reutiliza automáticamente.
- **No reanudes con parámetros distintos.** Si relanzas cambiando `--grad-accum`
  o el número de pasos, el proceso **aborta** en vez de continuar. El motivo es
  que esos valores fijan el denominador del schedule de learning rate, así que
  reanudar con otros estira el schedule a mitad de entrenamiento sin que nada lo
  delate. Reanudar = **el mismo comando**, literalmente.
- **Si `compare_lab.status` dice `quiet Xm (process alive)`, no pasa nada.** La
  preparación de datos se queda legítimamente callada varios minutos (entrenando
  el BPE, llenando el buffer de HuggingFace por red). Solo `DEAD(process gone)`
  indica un problema real, y se determina comprobando el PID, no la antigüedad.
- **La etapa de preparación de datos no se puede reanudar a medias.** Es la única
  que no lo soporta: los binarios se escriben de cero y el stream de HuggingFace
  no se puede rebobinar. Si creas un `STOP` durante esa etapa, termina limpiamente
  pero la próxima vez **empieza de nuevo**. Las etapas de entrenamiento
  (preentrenamiento y fine-tuning) sí se reanudan al lote exacto.

## 5. Nota técnica: por qué `tokens_per_sec` y el ETA no se disparan al reanudar

Antes de esta guía había un fallo sutil: `tokens_per_sec` se calculaba
dividiendo *todos* los tokens procesados desde el principio (incluidos los de
antes de la pausa) entre el tiempo de la tanda actual — justo después de
reanudar, eso daba una cifra absurdamente alta y un ETA poco fiable. Ahora
`tokens_per_sec` se calcula **solo con los tokens y el tiempo de la tanda
actual**, así que es correcto nada más reanudar, no solo al cabo de un rato.
