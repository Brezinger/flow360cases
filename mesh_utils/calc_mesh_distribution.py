from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SurfaceMeshZones:
    n_le: int
    n_const: int
    n_te: int

    x_le_end: float
    x_te_start: float

    h_const: float
    h_le_last: float
    h_te_last: float

    le_zone_length: float
    te_zone_length: float

    nodes: np.ndarray
    element_sizes: np.ndarray


def geometric_sum(first_size: float, growth_rate: float, n: int) -> float:
    """Sum of a geometric growth zone with n elements."""
    if n <= 0:
        return 0.0

    if abs(growth_rate - 1.0) < 1e-14:
        return n * first_size

    return first_size * (growth_rate**n - 1.0) / (growth_rate - 1.0)


def geometric_growth_rate_between(first_size: float, last_size: float, n: int) -> float:
    """Growth rate for n elements with prescribed first and last element sizes."""
    if n <= 0:
        raise ValueError("n must be positive.")

    if n == 1:
        if not np.isclose(first_size, last_size):
            raise ValueError("For n=1, first_size and last_size must match.")
        return 1.0

    return (last_size / first_size) ** (1.0 / (n - 1))


def geometric_sizes_between(first_size: float, last_size: float, n: int) -> np.ndarray:
    """Element sizes for a geometric zone with prescribed endpoint sizes."""
    if n <= 0:
        return np.array([])

    growth_rate = geometric_growth_rate_between(first_size, last_size, n)
    return np.array([first_size * growth_rate**i for i in range(n)])


def geometric_sum_between(first_size: float, last_size: float, n: int) -> float:
    """Sum of a geometric zone with prescribed first and last element sizes."""
    growth_rate = geometric_growth_rate_between(first_size, last_size, n)
    return geometric_sum(first_size, growth_rate, n)


def solve_constant_zone_size(
    chord: float,
    h_le_first: float,
    h_te_first: float,
    n_le: int,
    n_const: int,
    n_te: int,
) -> Optional[float]:
    """Solve the middle-zone size that closes the chord exactly for fixed counts."""
    if n_const <= 0:
        return None
    if n_le < 2 or n_te < 2:
        return None

    lower = max(h_le_first, h_te_first)

    def length_error(h_const: float) -> float:
        le_length = geometric_sum_between(h_le_first, h_const, n_le)
        te_length = geometric_sum_between(h_te_first, h_const, n_te)
        return le_length + n_const * h_const + te_length - chord

    if length_error(lower) > 0.0:
        return None

    upper = max(chord, lower * 2.0)
    max_upper = max(chord, lower, 1.0) * 100.0
    while length_error(upper) < 0.0:
        upper *= 2.0
        if upper > max_upper:
            return None

    for _ in range(50):
        mid = 0.5 * (lower + upper)
        if length_error(mid) > 0.0:
            upper = mid
        else:
            lower = mid

    return 0.5 * (lower + upper)


def minimum_growth_elements(
    first_size: float,
    last_size: float,
    max_growth_rate: float,
) -> int:
    """Smallest count needed to connect two sizes without exceeding max growth."""
    if last_size < first_size:
        raise ValueError("h_const_target must be >= the first LE/TE element sizes.")

    if np.isclose(first_size, last_size):
        return 2

    if np.isclose(max_growth_rate, 1.0):
        raise ValueError(
            "growth_rate=1.0 cannot connect different first and constant sizes."
        )

    return int(np.ceil(np.log(last_size / first_size) / np.log(max_growth_rate))) + 1


def local_growth_count_candidates(
    first_size: float,
    target_size: float,
    max_growth_rate: float,
    search_radius: int,
) -> range:
    """Local candidate growth-zone counts for automatic total-element mode."""
    min_count = minimum_growth_elements(first_size, target_size, max_growth_rate)

    return range(min_count, min_count + search_radius + 1)


def compute_surface_mesh_zones(
    total_elements: Optional[int],
    h_le_first: float,
    h_te_first: float,
    growth_rate: float,
    chord: float = 1.0,
    min_const_elements: int = 1,
    h_const_target: Optional[float] = None,
    auto_count_search_radius: int = 6,
    plot: bool = True,
):
    """
    Compute the split of one airfoil side into LE growth, constant, and TE growth zones.

    If h_const_target is None, total_elements is required and the original behavior is
    used: the routine searches integer LE/TE counts that minimize the jumps into the
    constant zone.

    If h_const_target is set, it is interpreted as an absolute length. The LE/TE
    growth-zone boundary element sizes are forced to equal the constant-zone size.
    In this mode total_elements may be None; then the routine derives the total count
    from h_const_target, chord, and growth_rate.
    """
    if total_elements is None and h_const_target is None:
        raise ValueError("total_elements must be set if h_const_target is None.")

    if total_elements is not None and total_elements < 3:
        raise ValueError("total_elements must be at least 3.")

    if h_le_first <= 0.0 or h_te_first <= 0.0:
        raise ValueError("The first element sizes must be positive.")

    if growth_rate < 1.0:
        raise ValueError("growth_rate must be >= 1.0.")

    if min_const_elements < 0:
        raise ValueError("min_const_elements must be >= 0.")

    if h_const_target is not None and h_const_target <= 0.0:
        raise ValueError("h_const_target must be positive.")

    if auto_count_search_radius < 0:
        raise ValueError("auto_count_search_radius must be >= 0.")

    length_total = chord
    h_le_first_abs = h_le_first * chord
    h_te_first_abs = h_te_first * chord

    best = None
    best_score = np.inf

    if total_elements is None:
        n_le_values = local_growth_count_candidates(
            h_le_first_abs, h_const_target, growth_rate, auto_count_search_radius
        )
        n_te_values = local_growth_count_candidates(
            h_te_first_abs, h_const_target, growth_rate, auto_count_search_radius
        )
        count_candidates = []
        for n_le in n_le_values:
            for n_te in n_te_values:
                le_length = geometric_sum_between(h_le_first_abs, h_const_target, n_le)
                te_length = geometric_sum_between(h_te_first_abs, h_const_target, n_te)
                remaining_length = length_total - le_length - te_length
                if remaining_length <= 0.0:
                    continue

                n_const_best = max(
                    min_const_elements, int(round(remaining_length / h_const_target))
                )
                for n_const in range(
                    max(min_const_elements, n_const_best - 2), n_const_best + 3
                ):
                    count_candidates.append((n_le, n_const, n_te))
    else:
        min_growth_zone_elements = 2 if h_const_target is not None else 1
        count_candidates = [
            (n_le, total_elements - n_le - n_te, n_te)
            for n_le in range(min_growth_zone_elements, total_elements + 1)
            for n_te in range(min_growth_zone_elements, total_elements + 1 - n_le)
        ]

    for n_le, n_const, n_te in count_candidates:
        if n_const < min_const_elements:
            continue

        if h_const_target is not None:
            h_const = solve_constant_zone_size(
                length_total,
                h_le_first_abs,
                h_te_first_abs,
                n_le,
                n_const,
                n_te,
            )
            if h_const is None:
                continue

            le_growth_rate = geometric_growth_rate_between(h_le_first_abs, h_const, n_le)
            te_growth_rate = geometric_growth_rate_between(h_te_first_abs, h_const, n_te)
            if le_growth_rate > growth_rate or te_growth_rate > growth_rate:
                continue

            le_length = geometric_sum_between(h_le_first_abs, h_const, n_le)
            te_length = geometric_sum_between(h_te_first_abs, h_const, n_te)
            h_le_last = h_const
            h_te_last = h_const
            score = np.log(h_const / h_const_target) ** 2
            if total_elements is None:
                score += 1e-12 * (n_le + n_const + n_te)
        else:
            le_length = geometric_sum(h_le_first_abs, growth_rate, n_le)
            te_length = geometric_sum(h_te_first_abs, growth_rate, n_te)
            remaining_length = length_total - le_length - te_length
            if remaining_length <= 0.0:
                continue

            if n_const > 0:
                h_const = remaining_length / n_const
            else:
                h_const = np.nan

            h_le_last = h_le_first_abs * growth_rate ** (n_le - 1)
            h_te_last = h_te_first_abs * growth_rate ** (n_te - 1)
            if n_const > 0:
                score = (
                    np.log(h_const / h_le_last) ** 2
                    + np.log(h_const / h_te_last) ** 2
                )
            else:
                score = np.log(h_le_last / h_te_last) ** 2

        if score < best_score:
            best_score = score
            best = (
                n_le,
                n_const,
                n_te,
                le_length,
                te_length,
                h_const,
                h_le_last,
                h_te_last,
            )

    if best is None:
        raise RuntimeError(
            "No valid zone split found. The requested sizes/growth zones are likely "
            "too long for the chord or incompatible with the count constraints."
        )

    n_le, n_const, n_te, le_length, te_length, h_const, h_le_last, h_te_last = best

    if h_const_target is not None:
        le_sizes = geometric_sizes_between(h_le_first_abs, h_const, n_le)
        te_sizes_from_te = geometric_sizes_between(h_te_first_abs, h_const, n_te)
    else:
        le_sizes = np.array([h_le_first_abs * growth_rate**i for i in range(n_le)])
        te_sizes_from_te = np.array([h_te_first_abs * growth_rate**i for i in range(n_te)])

    const_sizes = np.full(n_const, h_const) if n_const > 0 else np.array([])
    te_sizes = te_sizes_from_te[::-1]
    element_sizes = np.concatenate([le_sizes, const_sizes, te_sizes])

    nodes = np.concatenate([[0.0], np.cumsum(element_sizes)])
    if h_const_target is None:
        nodes *= chord / nodes[-1]
    else:
        nodes[-1] = chord
    element_sizes = np.diff(nodes)

    x_le_end = nodes[n_le]
    x_te_start = nodes[n_le + n_const]

    result = SurfaceMeshZones(
        n_le=n_le,
        n_const=n_const,
        n_te=n_te,
        x_le_end=x_le_end,
        x_te_start=x_te_start,
        h_const=h_const,
        h_le_last=h_le_last,
        h_te_last=h_te_last,
        le_zone_length=le_length,
        te_zone_length=te_length,
        nodes=nodes,
        element_sizes=element_sizes,
    )

    if plot:
        visualize_surface_mesh_zones(result, chord=chord)

    return result


def visualize_surface_mesh_zones(result: SurfaceMeshZones, chord: float = 1.0):
    """Visualize nodes and zone boundaries on a 1D ray."""
    nodes = result.nodes
    y = np.zeros_like(nodes)

    plt.figure(figsize=(12, 2.8))
    plt.plot([0, chord], [0, 0], linewidth=2)
    plt.scatter(nodes, y, s=18, zorder=3)
    plt.axvline(result.x_le_end, linestyle="--", linewidth=1.5)
    plt.axvline(result.x_te_start, linestyle="--", linewidth=1.5)

    plt.text(
        result.x_le_end,
        0.04,
        f"End of LE zone\ns/c = {result.x_le_end / chord:.5f}",
        ha="center",
        va="bottom",
    )

    plt.text(
        result.x_te_start,
        0.04,
        f"Start of TE zone\ns/c = {result.x_te_start / chord:.5f}",
        ha="center",
        va="bottom",
    )

    plt.text(0.0, -0.05, "LE", ha="left", va="top")
    plt.text(chord, -0.05, "TE", ha="right", va="top")

    plt.title(
        "Surface mesh distribution on one airfoil side\n"
        f"n_LE={result.n_le}, n_const={result.n_const}, n_TE={result.n_te}"
    )

    plt.xlabel("s/c")
    plt.yticks([])
    plt.ylim(-0.1, 0.15)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # AFT-SA discretization (V3)
    """result = compute_surface_mesh_zones(
        total_elements=300,
        h_le_first=0.0005,
        h_te_first=0.00328 * 2 / 6,
        growth_rate=1.05,
        chord=1.0,
        plot=True,
    )"""

    # Fully turbulent models. With h_const_target, total_elements can be None.
    result = compute_surface_mesh_zones(
        total_elements=None,
        h_le_first=0.001,
        h_te_first=0.5 / 105 / 4,
        growth_rate=1.2,
        chord=1.0,
        h_const_target=0.01,
        plot=True,
    )

    print(result)
