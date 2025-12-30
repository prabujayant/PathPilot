import torch
import torch.nn as nn
import numpy as np
from model import QoSLSTM

data = np.load("dataset.npz")
X = torch.tensor(data["X"], dtype=torch.float32)
Y = torch.tensor(data["Y"], dtype=torch.float32)

model = QoSLSTM(num_links=7)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

EPOCHS = 30

for epoch in range(EPOCHS):
    pred = model(X)
    loss = loss_fn(pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model.pt")
print("✅ Model saved as model.pt")
