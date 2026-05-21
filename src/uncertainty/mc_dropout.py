
import numpy as np
import torch


def enable_dropout(model):

    """
    Enable dropout layers during inference
    """

    for module in model.modules():

        if module.__class__.__name__.startswith("Dropout"):

            module.train()


def mc_dropout_predict(
    model,
    image,
    device,
    num_passes=30
):

    model.eval()

    enable_dropout(model)

    image = image.unsqueeze(0).to(device)

    predictions = []

    with torch.no_grad():

        for _ in range(num_passes):

            output = model(image)

            prob = torch.sigmoid(output)

            predictions.append(prob.item())

    predictions = np.array(predictions)

    mean_prediction = predictions.mean()

    uncertainty = predictions.var()

    return mean_prediction, uncertainty
