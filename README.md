# set-cover-solver-inv-oper

Solver exacto para la instancia de set cover ponderado de `docs/PF.pdf`: elegir antenas de modo que todo cliente quede cubierto al mínimo costo total. Usa Gurobi y se puede detener y reanudar entre sesiones.

## Modelo

- `a[i,j] = 1` si la antena `j` cubre al cliente `i` (`data/set_cover_500x500.csv`)
- `c[j]` es el costo de la antena `j` (`data/antenna_costs_500.csv`)
- `x[j] = 1` si se instala la antena `j`

```
min  sum_j c[j] x[j]
s.a. sum_j a[i,j] x[j] >= 1   para todo cliente i
     x[j] en {0, 1}           para toda antena j
```

En toda la salida las antenas se numeran desde 1, igual que el archivo de costos. El encabezado del CSV de cobertura las numera 0..499; el cargador corrige el desfase.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

El paquete `gurobipy` incluye una licencia restringida que admite hasta 2.000 variables y 2.000 restricciones. Esta instancia tiene 500 de cada una, así que no hace falta archivo de licencia. Para usar una licencia académica, defina `GRB_LICENSE_FILE` antes de ejecutar.

## Uso

```bash
# Inicia o reanuda hasta que la brecha sea como máximo 1% (el valor por defecto)
uv run set-cover solve

# Ajusta el objetivo después; continúa desde el mismo checkpoint
uv run set-cover solve --gap 0

# Límite por sesión, número de hilos y parámetros adicionales de Gurobi (opcionales)
uv run set-cover solve --time-limit 86400 --threads 12 --param MIPFocus=3

# Consulta el progreso desde otra terminal mientras corre una sesión
uv run set-cover status
uv run set-cover status --json

# Relajación LP (0 <= x <= 1): cota inferior y antenas fraccionales
uv run set-cover lp
uv run set-cover lp --json
```

### Argumentos de `solve`

| Argumento | Por defecto | Significado |
|---|---|---|
| `--coverage RUTA` | `data/set_cover_500x500.csv` | Matriz de cobertura |
| `--costs RUTA` | `data/antenna_costs_500.csv` | Costos de las antenas |
| `--run-dir RUTA` | `runs/default` | Directorio con el checkpoint y el log; reutilizarlo reanuda la ejecución |
| `--gap X` | `0.01` | Brecha relativa objetivo. `0` prueba el óptimo exacto |
| `--time-limit SEGUNDOS` | ninguno | Límite de esta sesión; sin él corre hasta alcanzar la brecha o hasta detenerla |
| `--threads N` | `0` (automático) | Hilos de Gurobi |
| `--seed N` | `0` | Semilla aleatoria de Gurobi |
| `--node-file-start GB` | `8.0` | Memoria del árbol antes de que Gurobi escriba nodos en `run-dir/nodefiles` |
| `--checkpoint-interval SEGUNDOS` | `60` | Cada cuánto se guarda la cota inferior. Las soluciones se guardan apenas se encuentran |
| `--param NOMBRE=VALOR` | ninguno | Cualquier otro parámetro de Gurobi; repetible |

Todos los argumentos pueden cambiar entre sesiones. Los usados en cada sesión quedan registrados en `checkpoint.json`.

El estado se guarda en `--run-dir`:

| Archivo           | Contenido                                                         |
| ----------------- | ----------------------------------------------------------------- |
| `checkpoint.json` | Mejor incumbente, mejor cota probada, historial de sesiones       |
| `gurobi.log`      | Log de Gurobi; cada sesión agrega al final                        |
| `nodefiles/`      | Nodos de branch-and-bound escritos a disco tras `--node-file-start` GB |
| `lock`            | Permite un solo proceso del solver por directorio de ejecución    |

### Relajación LP

`set-cover lp` resuelve el modelo con `0 <= x[j] <= 1` en lugar de `x[j] en {0, 1}`. Tarda menos de un segundo y no usa ni modifica ningún directorio de ejecución. Acepta `--coverage`, `--costs`, `--threads` y `--json`.

- `LP objective`: óptimo de la relajación. Ninguna solución entera puede costar menos.
- `integer bound`: el mismo valor redondeado al entero superior, válido porque los costos son enteros.
- `antennas at 1` y `fractional`: cuántas antenas quedan enteras y cuántas fraccionales. Las fraccionales muestran por qué la relajación no es una instalación real.

## Detener y reanudar

Detenga con Ctrl+C, `kill <pid>` (SIGTERM) o cerrando la terminal (SIGHUP). La sesión termina su checkpoint y sale. Ejecute `solve` de nuevo para reanudar.

Qué se conserva y qué se pierde al reiniciar:

- **Se conserva:** la mejor solución, escrita a disco cada vez que se encuentra. La siguiente sesión la carga como MIP start.
- **Se conserva:** la mejor cota inferior, guardada cada `--checkpoint-interval` segundos (60 por defecto). Se usa para la brecha reportada y para detenerse: una sesión termina en cuanto la incumbente está dentro de `--gap` de la mejor cota de cualquier sesión.
- **Se pierde:** el árbol de branch-and-bound. Gurobi no puede guardarlo, así que cada sesión reinicia la búsqueda desde la raíz. Las sesiones largas cierran la brecha más rápido que muchas cortas.

Un `kill -9` o un corte de luz pierde como máximo el progreso de la cota desde el último checkpoint. La siguiente ejecución marca esa sesión como `aborted`.

Estados de sesión:

- `optimal`: la incumbente iguala la cota; es óptima probada
- `gap_reached`: dentro de la brecha objetivo
- `time_limit`: se agotó `--time-limit`
- `interrupted`: detenida por una señal

Los costos son enteros, así que el solver redondea las cotas probadas al entero superior. Esto es válido, y una brecha de 0 significa que la solución es óptima probada.

## Algoritmo genético

`set-cover-ga` resuelve la misma instancia con un algoritmo genético con reparación voraz y eliminación de redundancias. No usa Gurobi. La documentación de operadores, calibración y resultados está en [`docs/genetic_algorithm.md`](docs/genetic_algorithm.md).

```bash
# Una ejecución con la configuración por defecto o la calibrada
uv run set-cover-ga run --seed 1
uv run set-cover-ga run --config configs/calibrated.json --seed 1 --progress 50 \
    --output results/ga/run.json --convergence results/ga/run_convergence.csv

# Cualquier parámetro se puede sobrescribir; --no-limit desactiva un criterio de parada
# Mutación: bit_flip, add, remove, swap o mixed
uv run set-cover-ga run --population-size 200 --mutation remove --mutation-rate 0.03 \
    --time-limit 30 --no-limit max_generations --no-limit stall_generations

# Calibración completa (explore, refine, time, final); Ctrl+C y volver a ejecutar reanuda
uv run set-cover-ga calibrate --workers 6
uv run set-cover-ga calibrate --stage final

# Comparación con el solver exacto (lee checkpoints; repetible con ETIQUETA=RUTA)
uv run set-cover-ga compare --checkpoint "Gurobi 92.7 h=runs/default/checkpoint.json"

# Mediciones de rendimiento de los operadores
uv run set-cover-ga benchmark
```

## Desarrollo

```bash
uv run pytest
uv run ruff check
uv run ruff format
uv run mypy
```
