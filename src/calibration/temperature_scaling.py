
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):

    def __init__(self):

        super().__init__()

        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):

        return logits / self.temperature


def fit_temperature(logits, labels):

    logits_tensor = torch.tensor(
        logits,
        dtype=torch.float32
    )

    labels_tensor = torch.tensor(
        labels,
        dtype=torch.float32
    )

    temperature_model = TemperatureScaler()

    optimizer = optim.LBFGS(
        [temperature_model.temperature],
        lr=0.01,
        max_iter=50
    )

    criterion = nn.BCEWithLogitsLoss()

    def closure():

        optimizer.zero_grad()

        scaled_logits = temperature_model(logits_tensor)

        loss = criterion(
            scaled_logits,
            labels_tensor
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    return temperature_model.temperature.item()
