import networkx as nx
import numpy as np
import random, time, matplotlib.pyplot as plt
from scipy.optimize import linprog

def generate_graph(n, p):
    return nx.fast_gnp_random_graph(n, p, seed=1)

def coverage(G, C):
    return sum(u in C or v in C for u, v in G.edges()) / G.number_of_edges()

def greedy_vertex_cover(G):
    start = time.time()
    C, H = set(), G.copy()
    while H.edges():
        u, v = next(iter(H.edges()))
        C.update([u, v])
        H.remove_nodes_from([u, v])
    return C, time.time() - start, coverage(G, C)

def lp_relaxation(G):
    start = time.time()
    n = len(G)
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    c = np.ones(n)
    A, b = [], []
    for u, v in G.edges():
        row = np.zeros(n)
        row[idx[u]] = row[idx[v]] = -1
        A.append(row)
        b.append(-1)
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method="highs")
    x = res.x if res.success else np.ones(n) * 0.5
    C = {nodes[i] for i in range(n) if x[i] >= 0.5}
    for u, v in G.edges():
        if u not in C and v not in C:
            C.add(random.choice([u, v]))
    return C, time.time() - start, coverage(G, C)

def randomized_heuristic(G):
    start = time.time()
    C, E = set(), set(G.edges())
    while E:
        u, v = random.choice(tuple(E))
        C.add(random.choice([u, v]))
        E = {e for e in E if e[0] not in C and e[1] not in C}
    return C, time.time() - start, coverage(G, C)

n, p = 500, 0.005
G = generate_graph(n, p)
print(f"Graph with {n} nodes, p={p}, edges={G.number_of_edges()}")

names, sizes, times, covs = [], [], [], []
for name, algo in [("Greedy", greedy_vertex_cover), ("LP", lp_relaxation), ("Randomized", randomized_heuristic)]:
    C, t, cov = algo(G)
    names.append(name); sizes.append(len(C)); times.append(t); covs.append(cov)
    print(f"{name:11s} | Selected: {len(C):4d} | Time: {t:6.3f}s | Coverage: {cov:.2f}")

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(names, sizes, color='skyblue', label='Selected Sensors')
ax2.plot(names, times, color='orange', marker='o', label='Time (s)')
ax1.set_ylabel('Number of Sensors')
ax2.set_ylabel('Execution Time (s)')
plt.title('Algorithm Comparison for IoT Sensor Placement')
plt.show()
