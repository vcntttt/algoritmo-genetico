#!/usr/bin/env python3
"""
MWC solver via Genetic Algorithm with parallel CPU evaluation, benchmarking CSV output,
progress logs, and cost history plotting.
"""

import os
import numpy as np
import json
import pandas as pd
import argparse
import matplotlib

# Use Agg backend for non-interactive environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import time
from multiprocessing import Pool, cpu_count

print(f">> Using NumPy (CPU) with {cpu_count()} cores for parallel evaluation")

# --- Utility functions ---


def convex_hull(points):
    """
    Monotone Chain convex hull. Input: list of (x,y,idx). Returns list of idx in hull order.
    """
    pts = sorted(points, key=lambda p: (p[0], p[1]))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for x, y, idx in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], (x, y)) <= 0:
            lower.pop()
        lower.append((x, y, idx))
    for x, y, idx in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], (x, y)) <= 0:
            upper.pop()
        upper.append((x, y, idx))
    hull = lower[:-1] + upper[:-1]
    return [idx for _, _, idx in hull]


def compute_distance_matrix(coords):
    """
    Compute full pairwise Euclidean distance matrix (NumPy).
    """
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=2)


def coalition_cost(G, dist_matrix):
    """
    Sum of pairwise distances for coalition G using NumPy.
    """
    G_arr = np.array(G)
    sub = dist_matrix[np.ix_(G_arr, G_arr)]
    return float(np.sum(np.triu(sub, 1)))


# Worker helper for parallel fitness


def _cost_worker(args):
    ind, dist_matrix = args
    return coalition_cost(ind, dist_matrix)


# --- Genetic Algorithm components ---


def initialize_population(n, q, pop_size):
    return [random.sample(range(n), q) for _ in range(pop_size)]


def tournament_selection(pop, costs, k):
    idxs = random.sample(range(len(pop)), k)
    return pop[min(idxs, key=lambda i: costs[i])].copy()


def crossover(p1, p2, q):
    k = q // 2
    child1 = p1[:k] + [g for g in p2 if g not in p1[:k]][: q - k]
    child2 = p2[: q - k] + [g for g in p1 if g not in p2[: q - k]][:k]
    return child1, child2


def mutate(ind, n):
    outsiders = list(set(range(n)) - set(ind))
    if outsiders:
        ind[random.randrange(len(ind))] = random.choice(outsiders)
    return ind


def genetic_algorithm(
    coords,
    q,
    dist_matrix,
    pop_size=100,
    generations=200,
    cx_prob=0.8,
    mut_prob=0.2,
    tournament_k=3,
):
    """
    Genetic algorithm with parallel fitness evaluation on CPU.
    Returns best coalition, its cost, cost history, and generation times.
    """
    n = coords.shape[0]
    population = initialize_population(n, q, pop_size)
    best, best_cost = None, float("inf")
    cost_history = []
    gen_times = []
    t_start = time.time()

    # Set up multiprocessing pool
    pool = Pool(processes=cpu_count())

    for gen in range(1, generations + 1):
        start_gen = time.time()
        costs = pool.map(_cost_worker, [(ind, dist_matrix) for ind in population])
        end_gen = time.time()
        curr_min = min(costs)
        cost_history.append(curr_min)
        gen_times.append(end_gen - start_gen)
        print(
            f"Gen {gen}/{generations}: Gen time {end_gen - start_gen:.2f}s, min cost {curr_min:.4f}",
            flush=True,
        )
        for ind, c in zip(population, costs):
            if c < best_cost:
                best_cost, best = c, ind.copy()
        if gen % 10 == 0 or gen in (1, generations):
            print(f"  >> Best so far: {best_cost:.4f}", flush=True)
        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, costs, tournament_k)
            p2 = tournament_selection(population, costs, tournament_k)
            if random.random() < cx_prob:
                c1, c2 = crossover(p1, p2, q)
            else:
                c1, c2 = p1, p2
            if random.random() < mut_prob:
                c1 = mutate(c1, n)
            if random.random() < mut_prob:
                c2 = mutate(c2, n)
            new_pop.extend([c1, c2])
        population = new_pop[:pop_size]

    pool.close()
    pool.join()
    total_time = time.time() - t_start
    print(f">> Total time: {total_time:.2f}s", flush=True)
    return best, best_cost, cost_history, gen_times, total_time


# --- Data loading ---


def json_to_positions(json_path, rollcall_idx=0):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    votes = data["rollcalls"][rollcall_idx]["votes"]
    rows = []
    for v in votes:
        rows.append(
            {"id": v["icpsr"], "x": v["x"], "y": v["y"], "party": v.get("party", "")}
        )
    return pd.DataFrame(rows)


# --- Plotting ---


def plot_results(coords, parties, coalition, title=None, output="resultado.png"):
    plt.figure()
    dem = [i for i, p in enumerate(parties) if p.upper().startswith("D")]
    rep = [i for i, p in enumerate(parties) if p.upper().startswith("R")]
    oth = [i for i in range(len(parties)) if i not in dem + rep]
    plt.scatter(coords[dem, 0], coords[dem, 1], c="blue", s=20, label="Demócratas")
    plt.scatter(coords[rep, 0], coords[rep, 1], c="red", s=20, label="Republicanos")
    if oth:
        plt.scatter(coords[oth, 0], coords[oth, 1], c="gray", s=20, label="Otros")
    hull = convex_hull([(coords[i][0], coords[i][1], i) for i in coalition])
    poly = coords[hull]
    poly = np.vstack([poly, poly[0]])
    plt.plot(poly[:, 0], poly[:, 1], "--", label="Convex Hull")
    if title:
        plt.title(title)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.savefig(output, dpi=300)


def plot_cost_history(cost_history, output="cost_history.png"):
    plt.figure()
    plt.plot(range(1, len(cost_history) + 1), cost_history, linewidth=1)
    plt.xlabel("Generación")
    plt.ylabel("Costo mínimo")
    plt.title("Evolución del costo por generación")
    plt.grid(True)
    plt.savefig(output, dpi=300)


# --- Main ---


def main():
    parser = argparse.ArgumentParser(description="MWC solver via Genetic Algorithm.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--json", help="JSON file with rollcalls")
    grp.add_argument("--data", help="CSV file with id,x,y,party")
    parser.add_argument("--rollcall", type=int, default=0, help="Rollcall index")
    parser.add_argument("--quorum", type=int, required=True, help="Coalition size")
    parser.add_argument("--popsize", type=int, default=100, help="Population size")
    parser.add_argument("--gens", type=int, default=200, help="Number of generations")
    parser.add_argument("--cx", type=float, default=0.8, help="Crossover probability")
    parser.add_argument("--mut", type=float, default=0.2, help="Mutation probability")
    parser.add_argument("--tour", type=int, default=3, help="Tournament size")
    parser.add_argument(
        "--benchmark-out", default="benchmark.csv", help="Benchmark CSV filepath"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--out", default="resultado.png", help="Output image filename")
    parser.add_argument(
        "--cost-out", default="cost_history.png", help="Cost history image filename"
    )
    args = parser.parse_args()

    if args.json:
        df = json_to_positions(args.json, args.rollcall)
    else:
        df = pd.read_csv(args.data)

    coords = df[["x", "y"]].values
    parties = df["party"].tolist()
    dist_matrix = compute_distance_matrix(coords)

    best, cost, cost_history, gen_times, total_time = genetic_algorithm(
        coords,
        args.quorum,
        dist_matrix,
        pop_size=args.popsize,
        generations=args.gens,
        cx_prob=args.cx,
        mut_prob=args.mut,
        tournament_k=args.tour,
    )
    print(f"Found MWC size={args.quorum} cost={cost:.4f}", flush=True)

    # --- Save benchmark CSV ---
    records = []
    for gen, (ct, ch) in enumerate(zip(gen_times, cost_history), start=1):
        records.append(
            {
                "quorum": args.quorum,
                "popsize": args.popsize,
                "gens": args.gens,
                "cx_prob": args.cx,
                "mut_prob": args.mut,
                "tour": args.tour,
                "generation": gen,
                "generation_time": ct,
                "min_cost": ch,
                "total_time": total_time,
                "result_cost": cost,
            }
        )
    bench_df = pd.DataFrame(records)
    write_header = not os.path.isfile(args.benchmark_out)
    bench_df.to_csv(args.benchmark_out, mode="a", index=False, header=write_header)
    print(f"Benchmark saved to {args.benchmark_out}", flush=True)

    if not args.no_plot:
        plot_results(
            coords,
            parties,
            best,
            title=f"MWC GA (size={args.quorum}, cost={cost:.2f})",
            output=args.out,
        )
        plot_cost_history(cost_history, output=args.cost_out)


if __name__ == "__main__":
    main()
