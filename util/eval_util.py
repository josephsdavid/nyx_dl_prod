import numpy as np
import torch
from sklearn.metrics import roc_curve
import sklearn.metrics as sk_metrics


def most_recent_n(x, orig, n, metric):
    sgn = np.sign(orig)
    orig_mask = orig.copy()
    for i in range(orig_mask.shape[0]):
        num_revs = orig_mask[i,:].astype(bool).sum()
        if n > num_revs:
            orig_mask[i,:] = 0
        else:
            orig_mask[i,:][np.where(orig_mask[i,:] > 0)[0][-n:]] = 0
    return metric(x * sgn, orig_mask)
