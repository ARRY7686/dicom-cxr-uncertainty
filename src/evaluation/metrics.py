
from sklearn.metrics import roc_auc_score


def compute_auc(labels, probabilities):

    auc = roc_auc_score(labels, probabilities)

    return auc
