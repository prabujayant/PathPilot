import time
import json
import requests
import numpy as np
from topology import LINKS
from routing import build_graph, shortest_path

STATS_URL = "http://localhost:8001/stats"
OUTPUT_FILE = "dataset.npz"

WINDOW = 10          # time steps
SAMPLES = 1500     # total samples to collect
SLEEP = 1            # seconds

def extract_features(stats):
    features = []

    for (src, dst) in LINKS:
        dpid = src.replace("s", "")
        port = "1"

        p = stats.get(dpid, {}).get(port, {})

        tx = p.get("tx_bps", 0) / 1e6
        rx = p.get("rx_bps", 0) / 1e6
        drop = p.get("tx_dropped", 0) / 100.0
        q = p.get("queue_len", 0) / 100.0

        features.extend([tx, rx, drop, q])

    return np.array(features, dtype=np.float32)



def compute_label(stats):
    """
    Heuristic: lowest utilization path
    """
    costs = []
    for (src, dst) in LINKS:
        dpid = src.replace("s", "")
        port = "1"
        p = stats.get(dpid, {}).get(port, {})
        cost = (p.get("tx_bps", 0) / 1e6) + (p.get("queue_len", 0) / 100)
        costs.append(cost)

    graph = build_graph(costs)
    path = shortest_path(graph, "h1", "h2")

    return costs, path


X, Y = [], []
history = []

print("[*] Collecting data... generate traffic now!")

for i in range(SAMPLES):
    stats = requests.get(STATS_URL).json()
    feat = extract_features(stats)
    history.append(feat)

    if len(history) < WINDOW:
        time.sleep(SLEEP)
        continue

    history = history[-WINDOW:]
    costs, _ = compute_label(stats)

    X.append(np.array(history))
    Y.append(np.array(costs))

    print(f"Collected {len(X)}/{SAMPLES}")
    time.sleep(SLEEP)

np.savez("dataset.npz", X=np.array(X), Y=np.array(Y))
print("Saved dataset.npz")
