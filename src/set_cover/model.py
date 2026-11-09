"""Formulación del set cover ponderado, compartida por el MIP y su relajación LP.

Modelo matemático
-----------------

Conjuntos:
    I = {1, ..., 500}   clientes, índice i
    J = {1, ..., 500}   ubicaciones posibles de antenas, índice j

Parámetros:
    a[i,j] = 1 si la antena j cubre al cliente i, 0 si no  (data/set_cover_500x500.csv)
    c[j]   = costo de instalar la antena j                  (data/antenna_costs_500.csv)

Variables:
    x[j] = 1 si se instala la antena j, 0 si no

Modelo:
    min  sum_j c[j] x[j]
    s.a. sum_j a[i,j] x[j] >= 1     para todo cliente i en I
         x[j] en {0, 1}             para toda antena j en J   (MIP)
         0 <= x[j] <= 1             para toda antena j en J   (relajación LP)

Correspondencia entre la matemática y el código
-----------------------------------------------

    | Matemática               | Código                                                |
    |--------------------------|-------------------------------------------------------|
    | x[j] en {0, 1}           | model.addVars(J, vtype=GRB.BINARY)                    |
    | 0 <= x[j] <= 1  (LP)     | model.addVars(J, lb=0, ub=1, vtype=GRB.CONTINUOUS)    |
    | min sum_j c[j] x[j]      | model.setObjective(quicksum(c[j] * x[j] ...), MINIMIZE) |
    | sum_j a[i,j] x[j] >= 1   | model.addConstr(quicksum(x[j] for j cubre i) >= 1)    |

Los índices i y j empiezan en 1, igual que en la matemática y en el archivo de
costos. Los arreglos de ``Instance`` empiezan en 0, de ahí los ``- 1``.
"""

import gurobipy
import numpy
from gurobipy import GRB

from set_cover.instance import Instance


def add_set_cover(
    model: gurobipy.Model, instance: Instance, vtype: str
) -> gurobipy.tupledict[int, gurobipy.Var]:
    """Agrega el modelo a ``model`` y retorna las variables ``x[j]`` indexadas por j.

    ``vtype`` es ``GRB.BINARY`` para el MIP o ``GRB.CONTINUOUS`` para la relajación.
    """
    customers = range(1, instance.num_customers + 1)  # I
    antennas = range(1, instance.num_antennas + 1)  # J

    # c[j]
    cost = {j: float(instance.costs[j - 1]) for j in antennas}
    # Antenas j con a[i,j] = 1: los términos de la fila i que no son cero.
    covering = {
        i: [int(j) + 1 for j in numpy.flatnonzero(instance.coverage[i - 1])] for i in customers
    }

    # x[j] para todo j en J, con cotas [0, 1]
    x = model.addVars(antennas, lb=0.0, ub=1.0, vtype=vtype, name="x")

    # min sum_j c[j] x[j]
    model.setObjective(gurobipy.quicksum(cost[j] * x[j] for j in antennas), GRB.MINIMIZE)

    # sum_j a[i,j] x[j] >= 1 para todo i en I
    for i in customers:
        model.addConstr(gurobipy.quicksum(x[j] for j in covering[i]) >= 1, name=f"cover[{i}]")

    return x
