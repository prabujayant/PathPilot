# model.py
import torch
import torch.nn as nn

class QoSLSTM(nn.Module):
    def __init__(self, num_links, hidden=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_links * 4,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(hidden, num_links)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
