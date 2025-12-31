import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================
# Model Definition
# ============================

class QoSLSTM(nn.Module):
    def __init__(self, num_links, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_links * 4,   # 4 features per link
            hidden_size=hidden,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(hidden, num_links)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last timestep


# ============================
# Load Dataset
# ============================

print("[INFO] Loading dataset...")

data = np.load("dataset.npz")
X = torch.tensor(data["X"], dtype=torch.float32)
Y = torch.tensor(data["Y"], dtype=torch.float32)

print(f"[INFO] X shape: {X.shape}")
print(f"[INFO] Y shape: {Y.shape}")

NUM_LINKS = Y.shape[1]

# ============================
# Model Setup
# ============================

model = QoSLSTM(num_links=NUM_LINKS)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# ============================
# Metric Helper
# ============================

def compute_metrics(y_true, y_pred, threshold=1.0):
    """
    Convert regression → binary classification
    """
    y_true_bin = (y_true > threshold).int()
    y_pred_bin = (y_pred > threshold).int()

    y_true_flat = y_true_bin.view(-1).cpu().numpy()
    y_pred_flat = y_pred_bin.view(-1).cpu().numpy()

    acc = accuracy_score(y_true_flat, y_pred_flat)
    prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    rec = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)

    return acc, prec, rec, f1


# ============================
# Training Loop
# ============================

EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()

    preds = model(X)
    loss = loss_fn(preds, Y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    acc, prec, rec, f1 = compute_metrics(Y, preds)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Loss: {loss.item():.4f} | "
        f"Acc: {acc:.3f} | "
        f"Prec: {prec:.3f} | "
        f"Recall: {rec:.3f} | "
        f"F1: {f1:.3f}"
    )

# ============================
# Save Model
# ============================

torch.save(model.state_dict(), "model.pt")
print("\n✅ Model saved as model.pt")
