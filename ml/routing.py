# routing.py
import heapq
from topology import LINKS

def build_graph(costs):
    graph = {}
    for i, (u, v) in enumerate(LINKS):
        graph.setdefault(u, []).append((v, costs[i]))
    return graph

def shortest_path(graph, src, dst):
    pq = [(0, src, [])]
    visited = set()

    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in visited:
            continue

        visited.add(node)
        path = path + [node]

        if node == dst:
            return path

        for nxt, w in graph.get(node, []):
            heapq.heappush(pq, (cost + w, nxt, path))

    return []
