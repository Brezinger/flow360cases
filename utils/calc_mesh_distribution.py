import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


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
    """
    Summe einer geometrischen Wachstumszone mit n Elementen.
    """
    if n <= 0:
        return 0.0

    if abs(growth_rate - 1.0) < 1e-14:
        return n * first_size

    return first_size * (growth_rate**n - 1.0) / (growth_rate - 1.0)


def compute_surface_mesh_zones(
    total_elements: int,
    h_le_first: float,
    h_te_first: float,
    growth_rate: float,
    chord: float = 1.0,
    min_const_elements: int = 1,
    plot: bool = True,
):
    """
    Berechnet die optimale Aufteilung einer Profilseite in:
    - Vorderkanten-Wachstumszone
    - konstante Zone
    - Hinterkanten-Wachstumszone

    Die Optimierung sucht ganzzahlige Elementzahlen n_le und n_te, sodass
    die konstante Elementgröße möglichst gut zu den letzten Elementgrößen
    der Wachstumszonen passt.

    Parameter
    ---------
    total_elements : int
        Gesamtzahl der Elemente pro Profilseite.
        Beispiel: 200 Punkte pro Seite -> 199 Elemente.

    h_le_first : float
        Größe des ersten Elements an der Nase, bezogen auf die Profiltiefe.
        Beispiel: 0.05 % c -> 0.0005

    h_te_first : float
        Größe des ersten Elements an der Endleiste, bezogen auf die Profiltiefe.
        Beispiel: 1/3000 -> 1/3000

    growth_rate : float
        Wachstumsrate der Wachstumszonen.
        Beispiel: 1.2

    chord : float
        Profiltiefe. Standardmäßig 1.0.

    min_const_elements : int
        Minimale Anzahl an Elementen in der konstanten Zone.

    plot : bool
        Falls True, wird eine Visualisierung erzeugt.

    Rückgabe
    --------
    SurfaceMeshZones
        Dataclass mit Zonengrenzen, Elementzahlen, Knoten und Elementgrößen.
    """

    if total_elements < 3:
        raise ValueError("total_elements muss mindestens 3 sein.")

    if h_le_first <= 0 or h_te_first <= 0:
        raise ValueError("Die ersten Elementgrößen müssen positiv sein.")

    if growth_rate < 1.0:
        raise ValueError("growth_rate sollte >= 1.0 sein.")

    if min_const_elements < 0:
        raise ValueError("min_const_elements muss >= 0 sein.")

    length_total = chord

    best = None
    best_score = np.inf

    # Suche über alle sinnvollen ganzzahligen Kombinationen
    for n_le in range(1, total_elements + 1):
        for n_te in range(1, total_elements + 1 - n_le):

            n_const = total_elements - n_le - n_te

            if n_const < min_const_elements:
                continue

            le_length = geometric_sum(h_le_first * chord, growth_rate, n_le)
            te_length = geometric_sum(h_te_first * chord, growth_rate, n_te)

            remaining_length = length_total - le_length - te_length

            if remaining_length <= 0:
                continue

            if n_const > 0:
                h_const = remaining_length / n_const
            else:
                # Falls keine konstante Zone erlaubt ist
                h_const = np.nan

            h_le_last = h_le_first * chord * growth_rate ** (n_le - 1)
            h_te_last = h_te_first * chord * growth_rate ** (n_te - 1)

            if n_const > 0:
                # Bewertet wird der relative Sprung an den Übergängen.
                # Logarithmen bestrafen Verhältnisse symmetrisch:
                # Faktor 2 und Faktor 1/2 sind gleich schlecht.
                score = (
                    np.log(h_const / h_le_last) ** 2
                    + np.log(h_const / h_te_last) ** 2
                )
            else:
                # Wenn keine konstante Zone existiert, Übergang LE direkt zu TE
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
            "Keine gültige Zoneneinteilung gefunden. "
            "Vermutlich sind die Wachstumszonen zu lang für die Gesamtzahl der Elemente."
        )

    n_le, n_const, n_te, le_length, te_length, h_const, h_le_last, h_te_last = best

    # Elementgrößen aufbauen: von Nase nach Endleiste
    le_sizes = np.array(
        [h_le_first * chord * growth_rate**i for i in range(n_le)]
    )

    if n_const > 0:
        const_sizes = np.full(n_const, h_const)
    else:
        const_sizes = np.array([])

    # Hinterkante wächst von der Endleiste nach innen.
    # In Laufrichtung Nase -> Endleiste müssen die Größen daher umgedreht werden.
    te_sizes_from_te = np.array(
        [h_te_first * chord * growth_rate**i for i in range(n_te)]
    )
    te_sizes = te_sizes_from_te[::-1]

    element_sizes = np.concatenate([le_sizes, const_sizes, te_sizes])

    # Numerische Rundungsfehler am Ende korrigieren
    nodes = np.concatenate([[0.0], np.cumsum(element_sizes)])
    nodes *= chord / nodes[-1]

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
    """
    Visualisiert Knotenpunkte und Zonengrenzen auf einem Strahl.
    """

    nodes = result.nodes
    y = np.zeros_like(nodes)

    plt.figure(figsize=(12, 2.8))

    # Strahl
    plt.plot([0, chord], [0, 0], linewidth=2)

    # Knotenpunkte
    plt.scatter(nodes, y, s=18, zorder=3)

    # Zonengrenzen
    plt.axvline(result.x_le_end, linestyle="--", linewidth=1.5)
    plt.axvline(result.x_te_start, linestyle="--", linewidth=1.5)

    plt.text(
        result.x_le_end,
        0.04,
        f"Ende VK-Zone\ns/c = {result.x_le_end / chord:.5f}",
        ha="center",
        va="bottom",
    )

    plt.text(
        result.x_te_start,
        0.04,
        f"Beginn EK-Zone\ns/c = {result.x_te_start / chord:.5f}",
        ha="center",
        va="bottom",
    )

    plt.text(0.0, -0.05, "Nase", ha="left", va="top")
    plt.text(chord, -0.05, "Endleiste", ha="right", va="top")

    plt.title(
        "Oberflächennetz auf einer Profilseite\n"
        f"n_LE={result.n_le}, "
        f"n_const={result.n_const}, "
        f"n_TE={result.n_te}"
    )

    plt.xlabel("s/c")
    plt.yticks([])
    plt.ylim(-0.1, 0.15)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    result = compute_surface_mesh_zones(
        total_elements=300,  # 200 Punkte pro Profilseite
        h_le_first=0.0005,  # 0.05 % der Profiltiefe
        h_te_first= 0.00328*2/6,  # Endleistenelement, 4 Elemente pro Dicke
        growth_rate=1.05,
        chord=1.0,
        plot=True,
    )

    print(result)

    pass