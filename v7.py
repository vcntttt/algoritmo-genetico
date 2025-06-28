import argparse
import json
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# GPU fallback
try:
    import cupy as cp

    USING_CUPY = True
    print(">> CuPy detectado — usando GPU")
except ImportError:
    import numpy as cp  # type: ignore

    USING_CUPY = False
    print(">> CuPy no disponible — usando NumPy (CPU)")
    raise ImportError("Pongase serio.")

# Timestamp
TS = datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------- I/O ----------
def load_voteview(path: str, idx: int = 0) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    votes = js["rollcalls"][idx]["votes"]
    return pd.DataFrame(
        {
            "id": [v["icpsr"] for v in votes],
            "x": [v["x"] for v in votes],
            "y": [v["y"] for v in votes],
            "party": [v.get("party", "") for v in votes],
            "name": [v["name"] for v in votes],
        }
    )


def convex_hull(points):
    pts = sorted(points)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    hull = lo[:-1] + hi[:-1]
    return [idx for *_, idx in hull]


# ---------- GA Core ----------
def dist_matrix(coords: np.ndarray) -> "cp.ndarray":
    arr = cp.asarray(coords)
    diff = arr[:, None, :] - arr[None, :, :]
    D = cp.linalg.norm(diff, axis=2)
    cp.fill_diagonal(D, 0)
    return D


def init_population(n: int, q: int, size: int):
    return [random.sample(range(n), q) for _ in range(size)]


def fitness_batch(pop, D):
    idx = cp.array(pop, dtype=cp.int32)
    sub = D[idx[:, :, None], idx[:, None, :]]
    tri = cp.triu(sub, 1)
    return cp.sum(tri, axis=(1, 2))


def tournament_select(pop, ranks_cdf):
    r = random.random()
    k = np.searchsorted(ranks_cdf, r)
    return pop[k]


def crossover(p1, p2, q):
    cut = random.randint(1, q - 1)
    child = p1[:cut] + [g for g in p2 if g not in p1[:cut]]
    return child[:q]


def mutate(ind, n, pm):
    if random.random() < pm:
        drop = random.choice(ind)
        add = random.choice(list(set(range(n)) - set(ind)))
        ind[ind.index(drop)] = add
    return ind


def genetic(coords, q, pop_size, generations, p_mut, p_sel):
    n = coords.shape[0]
    D = dist_matrix(coords)
    pop = init_population(n, q, pop_size)
    # Ranking-based probability CDF
    probs = np.array([p_sel * (1 - p_sel) ** i for i in range(pop_size)])
    probs /= probs.sum()
    ranks_cdf = np.cumsum(probs)
    best, best_fit = None, float("inf")
    hist = []
    t0 = time.time()

    for g in range(1, generations + 1):
        fits_gpu = fitness_batch(pop, D)
        order = cp.argsort(fits_gpu).get()
        fits = fits_gpu.get()[order]
        pop = [pop[i] for i in order]
        if fits[0] < best_fit:
            best_fit, best = fits[0], pop[0].copy()
        hist.append(fits[0])
        print(f"Gen {g}/{generations} | Best: {fits[0]:.2f} | Global: {best_fit:.2f}")
        # Elitismo + next gen
        new_pop = [best.copy()]
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, ranks_cdf)
            p2 = tournament_select(pop, ranks_cdf)
            child = crossover(p1, p2, q)
            child = mutate(child, n, p_mut)
            new_pop.append(child)
        pop = new_pop

    print(f"Tiempo total: {time.time() - t0:.1f}s | Mejor fitness: {best_fit:.2f}")
    return best, best_fit, hist


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Buscador de CGM mediante Algoritmo Genético (GPU)"
    )
    grupo = parser.add_mutually_exclusive_group(required=False)
    grupo.add_argument(
        "--json", default="data.json", help="Archivo JSON con votaciones"
    )
    parser.add_argument(
        "--votacion", type=int, default=0, help="Índice de la votación a usar"
    )
    parser.add_argument("--quorum", type=int, default=216, help="Quorum requerido")
    parser.add_argument(
        "--size_poblacion", type=int, default=38, help="Tamaño de la población"
    )
    parser.add_argument(
        "--generaciones", type=int, default=100, help="Cantidad de generaciones"
    )
    parser.add_argument(
        "--p_mutacion", type=float, default=0.1700019, help="Probabilidad de mutación"
    )
    parser.add_argument(
        "--p_seleccion",
        type=float,
        default=0.141,
        help="Probabilidad de selección (rho)",
    )
    args = parser.parse_args()

    df = load_voteview(args.json, args.votacion)
    coords = df[["x", "y"]].values

    sol, fit, hist = genetic(
        coords,
        args.quorum,
        args.size_poblacion,
        args.generaciones,
        args.p_mutacion,
        args.p_seleccion,
    )

    hull_idx = convex_hull([(coords[i][0], coords[i][1], i) for i in sol])
    df.loc[hull_idx].to_csv(f"coalicion_{TS}.csv", index=False)
    print(f"Fitness final: {fit:.2f} — CSV guardado coalicion_{TS}.csv")

    plt.figure()
    plt.plot(range(1, len(hist) + 1), hist, label="Best per Gen")
    plt.xlabel("Generación")
    plt.ylabel("Fitness Z")
    plt.title("Convergencia GA")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
