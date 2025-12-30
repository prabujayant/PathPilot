# server.py
import torch
import requests
from fastapi import FastAPI
from model import QoSLSTM
from features import extract_features
from routing import build_graph, shortest_path
from topology import NUM_LINKS

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


model = QoSLSTM(NUM_LINKS)
model.load_state_dict(torch.load("model.pt"))
model.eval()

history = []

STATS_URL = "http://localhost:8001/stats"

@app.get("/predict")
def predict():
    global history

    stats = requests.get(STATS_URL).json()
    feat = extract_features(stats)

    history.append(feat)
    if len(history) < 10:
        return {"status": "warming_up"}

    history = history[-10:]
    x = torch.tensor([history], dtype=torch.float32)

    with torch.no_grad():
        costs = model(x).squeeze().tolist()

    graph = build_graph(costs)
    path = shortest_path(graph, "h1", "h2")

    return {
        "path": path,
        "costs": costs
    }
