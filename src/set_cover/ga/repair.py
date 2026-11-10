"""Reparación voraz y eliminación de antenas redundantes.

Ambos operadores modifican ``genes`` y ``counts`` en el lugar. ``counts[i]`` es el
número de antenas seleccionadas que cubren al cliente ``i``; mantenerlo actualizado
evita recalcular la cobertura completa tras cada cambio.
"""

import numpy

from set_cover.ga.config import PruneOrder
from set_cover.ga.problem import Counts, CoverProblem, Genes, UncoverableError

# Dos razones a esta distancia relativa se consideran empatadas.
_TIE_TOLERANCE = 1e-9


def greedy_complete(
    problem: CoverProblem,
    genes: Genes,
    counts: Counts,
    rng: numpy.random.Generator,
    rcl_alpha: float = 0.0,
) -> int:
    """Agrega antenas hasta cubrir a todos los clientes; retorna cuántas agregó.

    En cada paso calcula, para cada antena, la razón entre clientes aún sin cobertura
    que cubriría y su costo. Con ``rcl_alpha = 0`` elige la mejor razón y desempata al
    azar. Con ``rcl_alpha > 0`` elige al azar entre las antenas cuya razón está a lo
    sumo ``rcl_alpha * (mejor - peor)`` por debajo de la mejor (greedy aleatorizado).
    """
    uncovered = counts == 0
    added = 0
    while uncovered.any():
        gains = problem.by_antenna @ uncovered.astype(numpy.float32)
        ratios = gains * problem.ratio_weights
        best = float(ratios.max())
        if best <= 0.0:
            raise UncoverableError((numpy.flatnonzero(uncovered) + 1).tolist())
        if rcl_alpha > 0.0:
            worst = float(ratios[ratios > 0.0].min())
            threshold = best - rcl_alpha * (best - worst)
        else:
            threshold = best * (1.0 - _TIE_TOLERANCE)
        candidates = numpy.flatnonzero(ratios >= threshold)
        chosen = (
            int(candidates[rng.integers(candidates.size)])
            if candidates.size > 1
            else int(candidates[0])
        )
        customers = problem.customers_of[chosen]
        genes[chosen] = True
        counts[customers] += 1
        uncovered[customers] = False
        added += 1
    return added


def remove_redundant(
    problem: CoverProblem,
    genes: Genes,
    counts: Counts,
    order: PruneOrder,
    rng: numpy.random.Generator,
) -> int:
    """Retira antenas cuya salida no deja clientes sin cobertura; retorna cuántas retiró.

    Una sola pasada basta: retirar antenas solo reduce conteos, así que una antena que
    no se pudo retirar sigue sin poder retirarse al final. El resultado es una cobertura
    sin antenas redundantes y su costo nunca supera al de la entrada.
    """
    selected = numpy.flatnonzero(genes)
    if selected.size == 0:
        return 0
    # La permutación previa desempata al azar; argsort estable la respeta.
    selected = selected[rng.permutation(selected.size)]
    if order is PruneOrder.COST_DESC:
        selected = selected[numpy.argsort(-problem.costs[selected], kind="stable")]
    elif order is PruneOrder.RATIO_DESC:
        # Una antena sin clientes nunca sobrevive a la poda; el piso evita dividir por 0.
        sizes = numpy.maximum(problem.coverage_sizes[selected], 1.0)
        selected = selected[numpy.argsort(-problem.costs[selected] / sizes, kind="stable")]
    removed = 0
    for antenna in selected.tolist():
        customers = problem.customers_of[antenna]
        if customers.size == 0 or counts[customers].min() > 1:
            genes[antenna] = False
            counts[customers] -= 1
            removed += 1
    return removed


def repair_and_prune(
    problem: CoverProblem,
    genes: Genes,
    order: PruneOrder,
    rng: numpy.random.Generator,
    rcl_alpha: float = 0.0,
) -> float:
    """Hace factible a ``genes``, elimina redundancias y retorna su costo."""
    counts = problem.coverage_counts(genes)
    greedy_complete(problem, genes, counts, rng, rcl_alpha)
    remove_redundant(problem, genes, counts, order, rng)
    return problem.cost(genes)
