
import numpy as np
import torch

from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    running_loss = 0.0

    for images, labels in tqdm(loader):

        images = images.to(device)

        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(loader)

    return epoch_loss


def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    val_losses = []

    all_labels = []
    all_probs = []

    with torch.no_grad():

        for images, labels in tqdm(loader):

            images = images.to(device)

            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)

            loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)

            val_losses.append(loss.item())

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

    val_loss = np.mean(val_losses)

    auc = roc_auc_score(
        all_labels,
        all_probs
    )

    return val_loss, auc
