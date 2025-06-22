#!/usr/bin/env python3
import numpy as np
import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt


def convex_hull(points):
    """
    Compute the convex hull of a set of 2D points using the Monotone Chain algorithm.
    `points` is a list of (x, y, idx) tuples.
    Returns a list of idx of points on the hull in counter-clockwise order.
    """
    pts = sorted(points, key=lambda p: (p[0], p[1]))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for x, y, idx in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], (x, y)) <= 0:
            lower.pop()
        lower.append((x, y, idx))

    upper = []
    for x, y, idx in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], (x, y)) <= 0:
            upper.pop()
        upper.append((x, y, idx))

    hull = lower[:-1] + upper[:-1]
    return [idx for (_, _, idx) in hull]


def compute_distance_matrix(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=2)


def coalition_cost(G, dist_matrix):
    cost = 0.0
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            cost += dist_matrix[G[i], G[j]]
    return cost


def initial_candidate(dist_matrix, q):
    n = dist_matrix.shape[0]
    best_cost = np.inf
    best_G = None
    for i in range(n):
        dists = dist_matrix[i].copy()
        dists[i] = np.inf
        neighbors = np.argpartition(dists, q - 1)[: q - 1]
        neighbors = neighbors[np.argsort(dists[neighbors])]
        G = [i] + list(neighbors)
        cost = coalition_cost(G, dist_matrix)
        if cost < best_cost:
            best_cost = cost
            best_G = G
    return best_G, best_cost


def find_mwc(coords, q, alpha=1.1, max_iter=100):
    n = coords.shape[0]
    dist_matrix = compute_distance_matrix(coords)
    Gc, current_cost = initial_candidate(dist_matrix, q)
    for _ in range(max_iter):
        C = coords[Gc].mean(axis=0)
        pts = [(coords[idx][0], coords[idx][1], idx) for idx in Gc]
        H = convex_hull(pts)
        dists_H = {idx: np.linalg.norm(coords[idx] - C) for idx in H}
        H_sorted = sorted(H, key=lambda idx: -dists_H[idx])
        r = alpha * max(dists_H.values())
        outside = [idx for idx in range(n) if idx not in Gc]
        P_c = [idx for idx in outside if np.linalg.norm(coords[idx] - C) <= r]
        P_c_sorted = sorted(P_c, key=lambda idx: np.linalg.norm(coords[idx] - C))
        improved = False
        for h in H_sorted:
            best_local_cost = current_cost
            best_local_G = None
            for p in P_c_sorted:
                G_trial = Gc.copy()
                G_trial.remove(h)
                G_trial.append(p)
                cost_trial = coalition_cost(G_trial, dist_matrix)
                if cost_trial < best_local_cost:
                    best_local_cost = cost_trial
                    best_local_G = G_trial
            if best_local_G is not None:
                Gc = best_local_G
                current_cost = best_local_cost
                improved = True
                break
        if not improved:
            break
    return Gc, current_cost


def json_to_positions(json_path, rollcall_idx=0):
    """
    Extracts id, x, y and party from JSON rollcalls.
    Assumes each vote has 'icpsr', 'x', 'y', and 'party' fields
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    votes = data["rollcalls"][rollcall_idx]["votes"]
    rows = []
    for v in votes:
        rows.append(
            {"id": v["icpsr"], "x": v["x"], "y": v["y"], "party": v.get("party", "")}
        )
    return pd.DataFrame(rows)


def plot_results(coords, parties, coalition, title=None):
    plt.figure()
    # Democrats in blue
    dem_idx = [i for i, p in enumerate(parties) if p.upper().startswith("D")]
    plt.scatter(
        coords[dem_idx, 0], coords[dem_idx, 1], c="blue", s=20, label="Demócratas"
    )
    # Republicans in red
    rep_idx = [i for i, p in enumerate(parties) if p.upper().startswith("R")]
    plt.scatter(
        coords[rep_idx, 0], coords[rep_idx, 1], c="red", s=20, label="Republicanos"
    )
    # Others in gray
    other_idx = [i for i in range(len(parties)) if i not in dem_idx + rep_idx]
    if other_idx:
        plt.scatter(
            coords[other_idx, 0], coords[other_idx, 1], c="gray", s=20, label="Otros"
        )
    # Convex hull of coalition
    hull_idxs = convex_hull([(coords[i][0], coords[i][1], i) for i in coalition])
    hull_coords = coords[hull_idxs]
    hull_coords = np.vstack([hull_coords, hull_coords[0]])
    plt.plot(hull_coords[:, 0], hull_coords[:, 1], linestyle="--", label="Convex Hull")
    if title:
        plt.title(title)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    # Save image
    plt.savefig("resultado.png", dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Find and plot Minimum Winning Coalition (MWC)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", help="Path to JSON file with rollcalls data")
    group.add_argument("--data", help="Path to CSV file with id,x,y columns")
    parser.add_argument(
        "--rollcall", type=int, default=0, help="Index of rollcall in JSON"
    )
    parser.add_argument(
        "--quorum", type=int, required=True, help="Size of the winning coalition"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.1, help="Scaling factor for search radius"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    if args.json:
        df = json_to_positions(args.json, rollcall_idx=args.rollcall)
    else:
        df = pd.read_csv(args.data)

    coords = df[["x", "y"]].values
    parties = df["party"].tolist()
    coalition, cost = find_mwc(coords, args.quorum, alpha=args.alpha)
    print(f"Found MWC of size {args.quorum} with cost {cost:.4f}:")
    # print ids if needed
    if "id" in df.columns:
        ids = df["id"].tolist()
        print([ids[i] for i in coalition])

    if not args.no_plot:
        title = f"MWC (size={args.quorum}, cost={cost:.2f})"
        plot_results(coords, parties, coalition, title=title)


if __name__ == "__main__":
    main()
