import networkx as nx
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count

def randomized_vertex_cover(G):
    edges = list(G.edges())
    cover = set()
    while edges:
        u, v = random.choice(edges)
        cover.add(random.choice([u, v]))
        edges = [(a, b) for a, b in edges if a not in cover and b not in cover]
    return cover

def trial(args):
    n, p, seed = args
    random.seed(seed)
    G = nx.fast_gnp_random_graph(n, p, seed=seed)
    C = randomized_vertex_cover(G)
    cover_size = len(C)
    mm = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    matching_size = len(mm) if len(mm) > 0 else 0
    return cover_size, matching_size

densities = [0.01, 0.05, 0.1, 0.2]
n = 1000
trials = 100
workers = max(1, cpu_count() - 1)

mean_sizes, variances, mean_times, mean_ratios = [], [], [], []

for p in densities:
    args = [(n, p, 1000 + i) for i in range(trials)]
    t0 = time.time()
    with Pool(processes=workers) as pool:
        res = pool.map(trial, args)
    t1 = time.time()
    covers = np.array([r[0] for r in res])
    matches = np.array([r[1] for r in res])
    ratios = np.array([c / m if m > 0 else np.nan for c, m in zip(covers, matches)])
    mean_sizes.append(np.nanmean(covers))
    variances.append(np.nanvar(covers))
    mean_times.append(t1 - t0)
    mean_ratios.append(np.nanmean(ratios))
    print(f"p={p:.3f} mean_size={mean_sizes[-1]:.2f} var={variances[-1]:.2f} time={mean_times[-1]:.2f}s ratio={mean_ratios[-1]:.2f}")

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(densities, mean_sizes, marker='o')
plt.title("Average Vertex Cover Size vs Graph Density")
plt.xlabel("Graph Density (p)")
plt.ylabel("Average |C|")

plt.subplot(2,2,2)
plt.plot(densities, variances, marker='o', color='orange')
plt.title("Variance of Vertex Cover Size vs Density")
plt.xlabel("Graph Density (p)")
plt.ylabel("Variance")

plt.subplot(2,2,3)
plt.plot(densities, mean_times, marker='o', color='red')
plt.title("Execution Time vs Graph Density")
plt.xlabel("Graph Density (p)")
plt.ylabel("Time (s)")

plt.subplot(2,2,4)
plt.plot(densities, mean_ratios, marker='o', color='green')
plt.title("Empirical Approximation Ratio (|C| / Matching)")
plt.xlabel("Graph Density (p)")
plt.ylabel("Mean Ratio")

plt.tight_layout()
plt.show()