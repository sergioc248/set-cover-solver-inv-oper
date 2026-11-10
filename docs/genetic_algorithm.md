# Algoritmo genético para el set cover ponderado

El algoritmo genético (GA) resuelve la misma instancia que el solver exacto: 500 clientes, 500 antenas, `data/set_cover_500x500.csv` y `data/antenna_costs_500.csv`. La función objetivo y las restricciones no cambian:

```
min  Z = sum_j c[j] x[j]
s.a. sum_j a[i,j] x[j] >= 1   para todo cliente i
     x[j] en {0, 1}
```

El código está en `src/set_cover/ga/` y no importa Gurobi. Solo `src/set_cover/compare.py` lee los resultados de ambos métodos.

## Datos verificados

- La matriz de cobertura tiene 500 filas (clientes) y 500 columnas (antenas) con valores 0/1. Hay 24.971 enlaces de cobertura. Cada antena cubre entre 33 y 66 clientes, y cada cliente está cubierto por entre 32 y 70 antenas.
- El encabezado del CSV de cobertura numera las antenas de 0 a 499. El gen `j` (base 0) corresponde a la antena `j + 1` del archivo de costos. Toda salida usa ids base 1.
- `Costo_S.xlsx` (fila 1: `Conjunto` 1..500; fila 2: `Costo`) coincide valor por valor con `antenna_costs_500.csv`: 0 diferencias, suma 1.516.821 en ambos. Los costos están entre 2.000 y 3.998 y no se normalizan.
- Todo cliente tiene al menos una antena que lo cubre. El cargador lo verifica, y `CoverProblem.build` lanza `UncoverableError` con la lista de clientes si no fuera así.

## Módulos

| Módulo | Contenido |
|---|---|
| `ga/problem.py` | `CoverProblem` (estructuras precalculadas) y `evaluate` (costo, antenas, clientes cubiertos, clientes sin cobertura, factibilidad) |
| `ga/repair.py` | Reparación voraz, greedy aleatorizado y eliminación de redundancias |
| `ga/operators.py` | Torneo, cruce uniforme, cruce de un punto y mutaciones bit-flip, agregar, quitar, intercambio y mixta |
| `ga/engine.py` | Ciclo evolutivo, elitismo, criterios de parada e historial |
| `ga/config.py` | `GAConfig`: hiperparámetros validados y serializables |
| `ga/calibration.py` | Calibración en etapas, reanudable y en paralelo |
| `ga/benchmark.py` | Mediciones de rendimiento |
| `compare.py` | Comparación con el solver exacto |

## Representación

Un individuo es un vector booleano de 500 genes: `genes[j] = True` instala la antena `j + 1`. `evaluate` siempre recalcula costo y cobertura desde la matriz original y no usa el estado incremental de los operadores. La solución final de cada ejecución se verifica dos veces: con `evaluate` y con `Instance.uncovered_customers` del cargador compartido con el solver.

## Fitness

`f(x) = sum_j c[j] x[j]`, sin penalizaciones. Todo individuo pasa por reparación y eliminación de redundancias antes de entrar a la población, así que todos son factibles. El fitness es la función objetivo y se compara directamente con el valor del solver exacto.

## Operadores

### Reparación voraz (`greedy_complete`)

1. Calcula los clientes sin cobertura (`counts == 0`).
2. Para cada antena calcula `ganancia / costo`, donde ganancia es el número de clientes sin cobertura que cubriría. Un producto matriz-vector (`by_antenna @ uncovered`) da todas las ganancias de una vez.
3. Agrega la antena con mejor razón. Los empates (diferencia relativa ≤ 1e-9) se resuelven al azar con el generador de la semilla.
4. Actualiza los conteos solo para los clientes de esa antena (`customers_of[j]`).
5. Repite hasta cubrir a todos. Si quedan clientes sin cobertura y ninguna antena aporta, lanza `UncoverableError` en lugar de ciclar.

### Greedy aleatorizado (inicialización)

Igual que la reparación, pero elige al azar entre las antenas con razón `>= mejor - α (mejor - peor)`, una lista restringida de candidatos estilo GRASP. Con `rcl_alpha = 0.3` (valor por defecto, fijo durante la calibración) produce soluciones factibles distintas pero buenas.

### Eliminación de redundancias (`remove_redundant`)

Recorre las antenas seleccionadas en el orden configurado (`prune_order`) y retira la antena `j` si todos sus clientes tienen `counts >= 2`.

- `cost_desc`: las más caras primero, porque retirarlas ahorra más.
- `ratio_desc`: primero las de mayor costo por cliente cubierto.
- `random`: orden aleatorio.

Los empates se ordenan al azar con la semilla. Una pasada basta: retirar antenas solo baja conteos, así que una antena que no se pudo retirar sigue sin poder retirarse. El resultado es una cobertura irredundante: retirar cualquier antena deja algún cliente sin cubrir. Como solo retira antenas, el costo nunca aumenta. Las pruebas lo verifican con los tres órdenes.

### Población inicial

- `round(greedy_fraction * población)` individuos greedy: el primero con greedy puro y el resto con greedy aleatorizado.
- El resto, aleatorios: cada gen se activa con probabilidad `random_activation = 0.02`, unas 10 antenas en promedio, y luego se repara. Así ningún individuo arranca con cientos de antenas.
- Si un individuo nuevo duplica a uno existente, se genera otro, hasta 50 intentos. En la instancia real la población inicial no tiene duplicados.

### Selección por torneo

Sortea `tournament_size` individuos con reemplazo y elige el de menor costo. Un torneo más grande aumenta la presión selectiva.

### Cruce

Se aplica con probabilidad `crossover_rate`; si no, los hijos son copias de los padres.

- **Uniforme**: cada gen viene de uno u otro padre con probabilidad 1/2. El segundo hijo es el complemento.
- **Un punto**: corte aleatorio en `1..499`; los hijos intercambian las colas.

Los operadores crean arreglos nuevos y nunca modifican a los padres.

### Mutación

`mutation` elige el operador; `mutation_rate` fija su intensidad. Todos modifican al hijo en el lugar, nunca a los padres, y el hijo pasa después por reparación y eliminación de redundancias.

| Operador | Efecto | Movimientos por hijo |
|---|---|---|
| `bit_flip` | Invierte cada gen con probabilidad `mutation_rate` | Binomial(500, `mutation_rate`) |
| `add` | Activa antenas inactivas al azar | Binomial(500, `mutation_rate`), a lo sumo las inactivas |
| `remove` | Desactiva antenas activas al azar; la reparación cubre los huecos con otras antenas | Binomial(500, `mutation_rate`), a lo sumo las activas |
| `swap` | Por cada movimiento, desactiva una antena activa y activa una inactiva | Binomial(500, `mutation_rate`), a lo sumo min(activas, inactivas) |
| `mixed` | Elige uno de los cuatro anteriores al azar para cada hijo | El del operador elegido |

Con la misma tasa, los cuatro operadores hacen en promedio el mismo número de movimientos.

**Por qué importa el operador.** Una solución usa unas 22 de 500 antenas. Con `bit_flip` casi todas las inversiones caen en genes inactivos, es decir, agregan antenas. La eliminación de redundancias retira enseguida esas antenas, y el hijo vuelve a ser su padre. `add` tiene el mismo problema. `remove` es el único operador que obliga a la reparación a buscar otra forma de cubrir a los clientes: el hijo se aleja del padre y la búsqueda no se detiene. Ejecuciones de 60 s sin parada por estancamiento, con la configuración calibrada anterior a estos operadores (población 200, tasa 0.002, 4 semillas por operador):

| Operador | Costos finales | Última mejora |
|---|---|---:|
| `bit_flip` | 49.051 en las 4 | ≤ 1,9 s |
| `add` | 49.051 (3), 49.790 | ≤ 1,5 s |
| `swap` | 49.051 en las 4 | ≤ 1,4 s |
| `mixed` | 49.051 (3), 49.183 | ≤ 3,2 s |
| `remove` | 49.051 (3), 49.017 | ≤ 1,1 s |
| `remove`, tasa 0.03 (2 semillas) | 48.591 en las 2 | 4,4 s y 32,9 s |

Con `bit_flip` el algoritmo deja de mejorar en segundos, y más tiempo no sirve. Con `remove` a tasa alta sigue encontrando soluciones mejores.

### Reemplazo, elitismo y duplicados

Generacional: las `elite_count` mejores soluciones pasan copiadas a la nueva generación, y el resto se llena con descendientes reparados. Un descendiente idéntico a uno ya presente en la nueva generación se descarta (`deduplicate = true`). Hasta 50 descartes seguidos; después se acepta para no ciclar. Esto retrasa la convergencia prematura.

Aparte de la población, `_Search` guarda la mejor solución de toda la ejecución, que se actualiza en cada evaluación. Es lo que se reporta aunque la ejecución termine por tiempo o estancamiento.

### Criterios de parada

Cualquier combinación de `max_generations`, `time_limit` (segundos), `stall_generations` (generaciones sin mejorar la mejor global) y `max_evaluations`. El presupuesto de evaluaciones se revisa antes de cada evaluación, así que se respeta exactamente. Si se agota a mitad de generación, los cupos libres se llenan con los mejores no élite de la generación anterior, que ya están evaluados.

Una **evaluación** es construir un individuo: reparar, eliminar redundancias y calcular el costo. Los descendientes duplicados descartados también cuentan.

## Rendimiento

`uv run set-cover-ga benchmark` mide cada operación en la instancia real. Resultados en esta máquina (12 CPU, NumPy 2.5.4, un hilo):

| Operación | µs |
|---|---:|
| Conteos de cobertura (producto matriz-vector float32) | 11,4 |
| Factibilidad con máscara booleana | 6,8 |
| Factibilidad con bitsets de 64 bits | 3,5 |
| Cruce + mutación | 8,4 |
| Greedy desde cero (~22 pasos) | 372 |
| Hijo: cruce + mutación + reparación | 81 |
| Hijo: + eliminación de redundancias | 110 |

Dentro del GA, donde los padres se parecen y la reparación es corta, el costo medio es de unos 70 µs por evaluación: 1 millón de evaluaciones en unos 70 s.

Decisiones:

- La matriz se guarda en float32 en ambas orientaciones para que los productos matriz-vector usen BLAS. `customers_of[j]` permite actualizar los conteos tocando solo los clientes de una antena, sin recorrer la matriz.
- Los bitsets reducen a la mitad la verificación de factibilidad, pero esa verificación no está en el ciclo: la reparación mantiene conteos, y la eliminación de redundancias necesita conteos, no solo cobertura. No se adoptaron.
- No hay Pandas dentro del ciclo evolutivo.
- No se usó Numba: el costo por evaluación permite cumplir toda la calibración en minutos, así que no hay un cuello de botella que lo justifique.
- Las ejecuciones de calibración son independientes y corren en paralelo en procesos separados (`--workers`, por defecto la mitad de los CPU). Cada proceso usa BLAS de un hilo.

## Reproducibilidad

Cada ejecución usa `numpy.random.default_rng(seed)` (PCG64), y todo el azar sale de ese generador: inicialización, desempates, selección, cruce, mutación y orden de eliminación. Con los mismos datos, configuración, semilla y versión de NumPy, el resultado es idéntico; una prueba lo verifica. La excepción es `time_limit`, que depende del reloj.

Cada registro de ejecución (`runs_*.jsonl`, o el JSON de `run --output`) incluye semilla, configuración e id de configuración, mejor costo, número e ids de antenas, clientes cubiertos y sin cobertura, factibilidad, generaciones, evaluaciones, tiempo, motivo de parada, generación, evaluación e instante de la mejor solución, y el historial completo por generación: evaluaciones, tiempo, mejor global, mejor, media y peor de la población, individuos distintos y si hubo mejora.

## Calibración

`uv run set-cover-ga calibrate` (presupuesto por defecto: 100.000 evaluaciones; 60 s en la etapa `time`) ejecuta cuatro etapas sobre la instancia real y escribe todo en `results/ga/calibration/`. Cada etapa guarda primero su plan (`plan_<etapa>.json`: configuraciones y semillas) y luego una línea JSON por ejecución terminada (`runs_<etapa>.jsonl`), sincronizada a disco. Ctrl+C espera las ejecuciones en curso, las guarda y sale; el mismo comando continúa sin repetir nada. Si los ajustes cambian, el plan ya no coincide y el comando se niega a mezclar resultados. Los ajustes usados están en `settings.json`.

Las semillas no se solapan entre etapas: explore 1001–1005, refine 2001–2015, time 3001–3010 y final 9001–9030. `rcl_alpha = 0.3`, `random_activation = 0.02` y `deduplicate = true` quedan fijos.

El presupuesto es de 100.000 evaluaciones y no 50.000 porque la mutación `remove` sigue mejorando después de 50.000 evaluaciones; con 50.000 la comparación favorecería a los operadores que convergen rápido y luego se estancan.

### Etapa A — Exploración (`explore`)

- 40 configuraciones distintas muestreadas sin reemplazo de las 131.220 combinaciones del espacio inicial, con semilla de muestreo 20261109.
- 5 semillas por configuración: 200 ejecuciones.
- Mismo presupuesto para todas: 100.000 evaluaciones. La parada por estancamiento usa el valor muestreado.

| Parámetro | Valores |
|---|---|
| Tamaño de población | 50, 100, 200 |
| Probabilidad de cruce | 0.6, 0.8, 0.9 |
| Tasa de mutación | 0.002, 0.005, 0.01, 0.02, 0.03, 0.05 |
| Operador de mutación | bit_flip, add, remove, swap, mixed |
| Tamaño del torneo | 2, 3, 5 |
| Élites | 1, 2, 5 |
| Cruce | uniforme, un punto |
| Proporción greedy inicial | 0.25, 0.5, 0.75 |
| Generaciones sin mejora | 100, 200, 300 |
| Orden de eliminación de redundancias | cost_desc, ratio_desc, random |

### Etapa B — Refinamiento (`refine`)

- Las 6 mejores configuraciones de A, más 4 vecinas de cada una. Una vecina cambia un solo parámetro al valor adyacente de una rejilla más fina (por ejemplo, tasa de mutación 0.03 → 0.02 o 0.04, o élites 5 → 3 u 8) o a otro valor de un parámetro categórico (operador de mutación, cruce, orden de poda). En total, 30 configuraciones.
- 15 semillas nuevas por configuración: 450 ejecuciones, mismo presupuesto de 100.000 evaluaciones.
- La primera del ranking es la configuración seleccionada (`selected_config.json`, copiada a `configs/calibrated.json`).

### Comparación con tiempo fijo (`time`)

Las 3 mejores de B, sin límite de generaciones, estancamiento ni evaluaciones, con 60 s por ejecución y 10 semillas nuevas. Es solo informativa y no cambia la selección.

### Etapa C — Evaluación final (`final`)

La configuración seleccionada, con 30 semillas que no se usaron en la calibración. Sus resultados no cambian los hiperparámetros.

### Criterio de selección

`rank` en `ga/calibration.py`:

1. Solo compiten las configuraciones con 100 % de ejecuciones factibles. En todas las etapas lo fueron las 710 ejecuciones.
2. Las configuraciones con mediana a lo sumo 0,2 % por encima de la mejor mediana se consideran similares.
3. Entre las similares gana la de menor desviación estándar. Los siguientes desempates son el menor tiempo medio y la menor mediana de evaluaciones hasta la mejor solución (velocidad de convergencia).
4. Las demás se ordenan por mediana y luego por los mismos desempates.

## Resultados de la calibración

Archivos completos: `summary_<etapa>.csv` (todas las configuraciones con su ranking y métricas), `parameter_effects_<etapa>.csv` (efecto marginal de cada valor) y `runs_<etapa>.jsonl` (cada ejecución con su historial).

Los tiempos se midieron con 6 procesos en paralelo mientras la máquina ejecutaba otras tareas. Sirven para comparar configuraciones dentro de una etapa, no como tiempos absolutos.

### Exploración: 10 mejores de 40

| # | Pob. | Cruce p | Mutación | Tasa | Torneo | Élites | Cruce | Greedy | Estanc. | Poda | Mediana | Media | Desv. | Mejor | s/ejec. |
|---|---:|---:|---|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 100 | 0.8 | remove | 0.03 | 5 | 5 | uniforme | 0.25 | 200 | random | 48.591 | 48.643,8 | 118,1 | 48.591 | 12,4 |
| 2 | 100 | 0.6 | remove | 0.01 | 5 | 5 | un punto | 0.75 | 200 | ratio_desc | 48.878 | 48.912,6 | 77,4 | 48.878 | 7,6 |
| 3 | 100 | 0.6 | remove | 0.005 | 5 | 2 | un punto | 0.25 | 300 | ratio_desc | 48.878 | 48.912,6 | 77,4 | 48.878 | 10,9 |
| 4 | 50 | 0.9 | mixed | 0.005 | 5 | 2 | uniforme | 0.5 | 100 | random | 49.051 | 49.051,0 | 0 | 49.051 | 1,6 |
| 5 | 100 | 0.9 | remove | 0.002 | 5 | 5 | uniforme | 0.25 | 100 | random | 49.051 | 49.051,0 | 0 | 49.051 | 1,9 |
| 6 | 50 | 0.8 | remove | 0.005 | 3 | 2 | un punto | 0.25 | 200 | random | 49.051 | 49.051,0 | 0 | 49.051 | 2,6 |
| 7 | 200 | 0.8 | bit_flip | 0.01 | 2 | 1 | un punto | 0.5 | 100 | random | 49.051 | 49.051,0 | 0 | 49.051 | 2,7 |
| 8 | 100 | 0.6 | remove | 0.005 | 2 | 1 | un punto | 0.5 | 200 | random | 49.051 | 49.051,0 | 0 | 49.051 | 4,3 |
| 9 | 200 | 0.9 | bit_flip | 0.002 | 5 | 2 | un punto | 0.75 | 200 | random | 49.051 | 49.051,0 | 0 | 49.051 | 6,2 |
| 10 | 50 | 0.8 | mixed | 0.002 | 5 | 2 | un punto | 0.75 | 300 | cost_desc | 49.051 | 49.051,0 | 0 | 49.051 | 6,5 |

Medianas de las 40: 48.591 en 1, 48.878 en 2, 49.051 en 24 y entre 50.146 y 52.063 en las 13 restantes. Las tres primeras usan `remove`. Ver `summary_explore.csv`.

### Refinamiento: 5 mejores de 30 (15 semillas cada una)

| # | id | Pob. | Cruce p | Mutación | Tasa | Torneo | Élites | Greedy | Estanc. | Poda | Mediana | Media | Desv. | Mejor | Peor | s/ejec. |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 8bcd910acdc6 | 100 | 0.8 | remove | 0.03 | 5 | 8 | 0.25 | 200 | random | 48.591 | 48.666,8 | 160,3 | 48.591 | 49.051 | 9,5 |
| 2 | 372a7b9f075b | 100 | 0.8 | remove | 0.03 | 5 | 5 | 0.25 | 200 | random | 48.591 | 48.666,8 | 160,3 | 48.591 | 49.051 | 11,5 |
| 3 | a97c0d208323 | 100 | 0.7 | remove | 0.03 | 5 | 5 | 0.25 | 200 | random | 48.591 | 48.697,5 | 186,6 | 48.591 | 49.051 | 10,0 |
| 4 | 8278d5f8ab23 | 100 | 0.8 | remove | 0.03 | 7 | 5 | 0.25 | 200 | random | 48.591 | 48.683,0 | 190,5 | 48.591 | 49.051 | 10,0 |
| 5 | dd2dc81a7826 | 100 | 0.8 | remove | 0.03 | 5 | 5 | 0.4 | 200 | random | 48.591 | 48.740,2 | 218,9 | 48.591 | 49.051 | 9,3 |

Todas usan cruce uniforme. Medianas de las 30: 48.591 en 5, 48.878 en 4 y 49.051 en 21. Las 5 primeras son similares (misma mediana); decide la desviación y, entre las dos primeras, que tienen la misma, el tiempo medio.

### Efecto de cada parámetro en la exploración

Promedio sobre las ejecuciones de la etapa A (`parameter_effects_explore.csv`). Las configuraciones no forman un diseño factorial, así que el efecto es descriptivo. "Alcanza 48.591" es la fracción de ejecuciones que llegó al menor costo de la etapa.

| Parámetro | Valor | Costo medio | Alcanza 48.591 | s/ejec. |
|---|---|---:|---:|---:|
| Operador de mutación | bit_flip / add / remove / swap / mixed | 49.879 / 50.627 / 48.976 / 50.419 / 49.233 | 0 / 0 / 10,0 / 0 / 0 % | 3,8 / 2,6 / 5,1 / 6,6 / 3,5 |
| Tasa de mutación | 0.002 / 0.005 / 0.01 / 0.02 / 0.03 / 0.05 | 49.385 / 49.118 / 49.503 / 49.487 / 50.196 / 52.063 | 0 / 0 / 0 / 4,0 / 14,3 / 0 % | 4,0 / 4,4 / 3,8 / 3,8 / 4,8 / 3,7 |
| Población | 50 / 100 / 200 | 49.931 / 49.490 / 50.088 | 1,5 / 5,6 / 0 % | 2,1 / 4,5 / 6,3 |
| Prob. cruce | 0.6 / 0.8 / 0.9 | 49.921 / 49.341 / 49.939 | 1,3 / 7,3 / 1,4 % | 4,6 / 4,1 / 3,7 |
| Torneo | 2 / 3 / 5 | 49.902 / 50.037 / 49.507 | 3,1 / 0 / 4,7 % | 3,1 / 3,9 / 5,1 |
| Élites | 1 / 2 / 5 | 50.041 / 49.383 / 49.679 | 1,1 / 0 / 8,3 % | 3,1 / 4,8 / 5,2 |
| Cruce | un punto / uniforme | 49.621 / 49.931 | 1,0 / 5,3 % | 3,9 / 4,4 |
| Greedy inicial | 0.25 / 0.5 / 0.75 | 49.550 / 49.757 / 49.951 | 7,3 / 1,3 / 1,4 % | 4,0 / 3,6 / 4,8 |
| Estancamiento | 100 / 200 / 300 | 50.065 / 49.293 / 49.932 | 0 / 8,6 / 0 % | 2,1 / 4,7 / 7,8 |
| Orden de poda | cost_desc / random / ratio_desc | 49.391 / 50.150 / 49.256 | 0 / 4,5 / 1,7 % | 5,6 / 3,9 / 4,0 |

Lecturas:

- El operador de mutación es el parámetro que más pesa. Solo `remove` llega a 48.591 y tiene el menor costo medio. `add` es el peor: solo agrega antenas que la poda retira.
- La tasa interactúa con el operador. Con `remove`, 0.03 (unas 15 antenas retiradas en promedio, de 22) es la que mejor funciona. Con `bit_flip`, `add` o `swap`, una tasa alta solo destruye soluciones buenas: 0.05 da 52.063 en todas las ejecuciones con esa tasa.
- El cruce uniforme supera al de un punto en la mejor solución alcanzada: el orden de las antenas en el cromosoma no tiene significado espacial.
- Los efectos de población, torneo, élites y estancamiento están confundidos con el operador de mutación de cada configuración muestreada; la etapa B los aísla mejor.

### Comparación con tiempo fijo (60 s, 10 semillas)

Las 3 mejores de B sin límites de evaluaciones ni estancamiento; por eso sus ids cambian.

| id (equivalente en B) | Mediana | Media | Desv. | Mejor | Peor | Evaluaciones medias |
|---|---:|---:|---:|---:|---:|---:|
| d31a8e2b3d59 (8bcd910acdc6, seleccionada) | 48.591 | 48.591 | 0 | 48.591 | 48.591 | 176.819 |
| 5be72e211dcf (372a7b9f075b) | 48.591 | 48.591 | 0 | 48.591 | 48.591 | 179.020 |
| b8d576c1434b (a97c0d208323) | 48.591 | 48.591 | 0 | 48.591 | 48.591 | 178.462 |

Con 60 s y sin parada por estancamiento, las 30 ejecuciones llegan a 48.591. Las que en B quedan en 49.051 se detienen por estancamiento antes de escapar; con más tiempo escapan.

## Configuración calibrada

`configs/calibrated.json`:

| Parámetro | Por defecto (`configs/default.json`) | Calibrado |
|---|---:|---:|
| `population_size` | 100 | 100 |
| `crossover_rate` | 0.8 | 0.8 |
| `mutation` | bit_flip | remove |
| `mutation_rate` | 0.005 | 0.03 |
| `tournament_size` | 3 | 5 |
| `elite_count` | 2 | 8 |
| `crossover` | uniform | uniform |
| `greedy_fraction` | 0.5 | 0.25 |
| `prune_order` | cost_desc | random |
| `stall_generations` | 200 | 200 |
| `max_generations` | 2000 | ninguno |
| `max_evaluations` | ninguno | 100.000 |

La configuración por defecto se eligió antes de calibrar y solo sirve como punto de partida.

## Evaluación final (30 semillas nuevas, 9001–9030)

`final_statistics.json`, `runs_final.jsonl` y `final_convergence.csv` (curva por semilla y generación).

| Métrica | Valor |
|---|---:|
| Ejecuciones factibles | 30 / 30 (500 / 500 clientes en cada una) |
| Costo medio | 48.645,9 |
| Mediana | 48.591 |
| Mejor | 48.538 (semilla 9012) |
| Peor | 49.051 |
| Desviación estándar | 138,8 |
| Tiempo medio por ejecución | 9,54 s (mín. 6,93, máx. 16,29, desv. 2,28) |
| Antenas | 21 o 22 |
| Mediana de evaluaciones hasta la mejor | 5.671 |
| Mediana de tiempo hasta la mejor | 1,90 s |
| Motivo de parada | estancamiento en las 30 |

Distribución: 48.591 en 24 ejecuciones; 48.538, 48.728, 48.878, 48.981, 49.017 y 49.051 en una cada una.

Mejor solución (costo 48.538, 21 antenas, base 1): `13, 19, 29, 44, 63, 94, 100, 105, 121, 175, 266, 357, 366, 405, 411, 418, 444, 450, 453, 475, 497`.

## Ejecuciones largas (30 min)

Para responder si más tiempo mejora el resultado: 6 semillas nuevas (9101–9106) con `configs/calibrated.json`, sin estancamiento ni presupuesto de evaluaciones, 30 min cada una, 6 en paralelo. Resultados en `results/ga/long_runs/`: `summary.jsonl` (una línea por semilla, sin historial) e `improvements.csv` (cada mejora de la mejor global).

```bash
uv run set-cover-ga run --config configs/calibrated.json --seed 9103 \
    --no-limit stall_generations --no-limit max_evaluations --time-limit 1800
```

| Semilla | Costo final | Antenas | Llega a 48.591 | Llega a 48.538 | Evaluaciones | Generaciones |
|---|---:|---:|---:|---:|---:|---:|
| 9101 | 48.591 | 21 | 8,4 s | — | 5.227.876 | 52.152 |
| 9102 | 48.591 | 21 | 7,8 s | — | 5.219.021 | 52.085 |
| 9103 | **48.538** | 21 | 1,6 s | 607 s | 5.274.130 | 52.654 |
| 9104 | 48.591 | 21 | 28,5 s | — | 5.252.318 | 52.420 |
| 9105 | 48.591 | 21 | 2,9 s | — | 5.276.231 | 52.617 |
| 9106 | **48.538** | 21 | 2,7 s | 849 s | 5.162.309 | 51.479 |

- Todas llegan a 48.591 en menos de 30 s.
- 2 de 6 escapan de 48.591 y llegan a 48.538, tras 10 y 14 min. Las otras 4 pasan 29 min sin mejorar.
- 48.538 es la misma solución que encontró la semilla 9012 de la evaluación final en 10 s. Comparte pocas antenas con la de 48.591 (`44, 63, 105, 357, 444`), así que es otra región del espacio, no un ajuste local.
- Ninguna ejecución, corta o larga, bajó de 48.538.

Conclusión: con la mutación `remove`, más tiempo sí ayuda, pero con rendimientos decrecientes. Media hora aumenta la probabilidad de llegar a 48.538, de 1 en 30 ejecuciones cortas a 2 en 6 ejecuciones largas. Con el mismo cómputo, muchas ejecuciones cortas con semillas distintas rinden parecido: 6 ejecuciones de 30 min equivalen a unas 1.100 ejecuciones de 9,5 s, y si la tasa de 1 en 30 observada en la evaluación final se mantiene, casi seguro alguna llegaría a 48.538. Esa tasa sale de una sola ejecución exitosa, así que es una estimación gruesa. Con la mutación `bit_flip` original, más tiempo no sirve: deja de mejorar en segundos.

## Comparación con el solver exacto

`uv run set-cover-ga compare` verifica cada solución contra la matriz original (cobertura y costo) y escribe `results/ga/comparison/comparison.{json,csv,md}`. Las sesiones cortas de Gurobi se ejecutaron con el solver existente (`set-cover solve --gap 0 --time-limit 4` y `--time-limit 112`); sus checkpoints y trayectorias (`checkpoint.json`, `trajectory.csv`) están en `results/ga/comparison/gurobi/<4s|112s|92h>/`. La interfaz (`uv run set-cover-ui`, página Comparación) muestra la misma tabla con gráficos de costo, calidad vs tiempo y antenas elegidas. Gurobi usa 12 hilos; cada ejecución del GA usa uno.

| Método | Costo | Antenas | Clientes | Tiempo | Cota inferior | Brecha del solver | Dif. vs mejor conocida | Cota sup. de la brecha |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GA, mejor de 30 | 48.538 | 21 | 500 | 9,5 s por ejecución | — | — | 0,00 % | 24,86 % |
| GA, mediana de 30 | 48.591 | 21–22 | 500 | 8,7 s | — | — | 0,11 % | 24,99 % |
| Gurobi, 4 s | 52.554 | 23 | 500 | 4 s | 28.183 | 46,37 % | 8,27 % | 35,19 % |
| Gurobi, 112 s | 50.048 | 22 | 500 | 112 s | 32.346 | 35,37 % | 3,11 % | 28,74 % |
| Gurobi, 92,7 h (checkpoint `runs/default`) | 49.017 | 22 | 500 | 333.822 s | 38.875 | 20,69 % | 0,99 % | 26,09 % |

Cómo leer la tabla:

- **Gurobi no demostró optimalidad.** La mejor cota inferior probada es 38.875, así que el óptimo está entre 38.875 y 48.538. Por eso no se calcula `Gap_GA(%) = (Z_GA − Z*) / Z*`: no hay `Z*`.
- **Dif. vs mejor conocida** es `(Z − 48.538) / 48.538`. La mejor solución conocida es del GA.
- **Cota sup. de la brecha** es `(Z − 38.875) / 38.875`. Como `Z* ≥ 38.875`, el `Gap(%)` real respecto al óptimo es como mucho este valor. Es una cota, no la brecha.
- **Brecha del solver** es la definición de Gurobi, `(incumbente − cota) / incumbente`, de cada sesión.
- El GA nunca recibe soluciones del solver. Comparten solo los archivos de datos y el cargador.

En 9 s, la mediana del GA (48.591) ya es mejor que la incumbente de Gurobi tras 92,7 h (49.017). Las 29 de 30 ejecuciones finales del GA igualan o mejoran esa incumbente. Lo que el GA no da es una cota: solo el solver exacto prueba que ninguna solución cuesta menos de 38.875.

## Limitaciones

- Hay una sola instancia real. Las semillas separadas por etapa reducen el sobreajuste a secuencias aleatorias concretas, pero no prueban que estos parámetros sean los mejores para otras instancias de set cover.
- 48.591 es un óptimo local muy atractivo: el 80 % de las ejecuciones finales termina ahí, en la etapa `time` todas llegan y ninguna lo supera en 60 s, y 4 de 6 ejecuciones de 30 min no salen de él. Con los criterios pedidos (mediana y luego desviación), la calibración premia la estabilidad, no la mejor solución posible.
- No se sabe si 48.538 es óptimo: entre 38.875 y 48.538 no hay demostración. Cerrar esa brecha requiere el solver exacto, no más ejecuciones del GA.
- `time_limit` hace la ejecución dependiente del reloj y de la carga de la máquina.

## Comandos

```bash
uv sync

# Una ejecución
uv run set-cover-ga run --config configs/calibrated.json --seed 9012
uv run set-cover-ga run --config configs/calibrated.json --mutation swap --mutation-rate 0.01
uv run set-cover-ga run --config configs/calibrated.json --seed 1 --progress 20 \
    --output results/ga/run.json --convergence results/ga/run_convergence.csv

# Calibración completa o por etapas; reanudable
uv run set-cover-ga calibrate --workers 6
uv run set-cover-ga calibrate --stage explore --stage refine
cp results/ga/calibration/selected_config.json configs/calibrated.json

# Comparación con el solver exacto
uv run set-cover solve --run-dir runs/compare-4s --time-limit 4 --gap 0
uv run set-cover solve --run-dir runs/compare-112s --time-limit 112 --gap 0
uv run set-cover-ga compare \
    --checkpoint "Gurobi 4 s (12 hilos)=runs/compare-4s/checkpoint.json" \
    --checkpoint "Gurobi 112 s (12 hilos)=runs/compare-112s/checkpoint.json" \
    --checkpoint "Gurobi 92.7 h (12 hilos)=runs/default/checkpoint.json"

# Rendimiento y pruebas
uv run set-cover-ga benchmark
uv run pytest
```

## Pruebas

`tests/test_ga_*.py` y `tests/test_compare.py`:

| Requisito | Prueba |
|---|---|
| Carga de datos, 500×500, costos sin alterar y desfase 0/1 | `test_ga_problem.py` |
| Cromosoma de 500 genes binarios; rechazo de cromosomas mal formados | `test_ga_problem.py` |
| Costo y clientes cubiertos correctos (contra un recorrido directo) | `test_ga_problem.py` |
| La reparación produce soluciones factibles; detecta clientes imposibles | `test_ga_repair.py` |
| La poda nunca aumenta el costo ni pierde cobertura; resultado irredundante | `test_ga_repair.py` |
| Operadores no modifican a los padres; límites y efecto de cada mutación | `test_ga_operators.py` |
| Cada operador de mutación produce soluciones factibles y reproducibles | `test_ga_engine.py` |
| Elitismo: la mejor de la población es la mejor global en cada generación | `test_ga_engine.py` |
| Reproducibilidad con la misma semilla | `test_ga_engine.py` |
| Toda solución reportada cumple las 500 restricciones | `test_ga_engine.py` |
| Instancias pequeñas contra enumeración exhaustiva | `test_ga_engine.py` |
| Criterios de parada | `test_ga_engine.py` |
| La calibración ordena y selecciona por métricas calculadas; reanuda; semillas disjuntas | `test_ga_calibration.py` |
| Brechas con y sin óptimo demostrado | `test_compare.py` |
