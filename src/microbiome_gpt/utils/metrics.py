import numpy as np


def cos_sim(X, X_pred):
    X_flat      = np.concatenate(np.array(X))
    X_pred_flat = np.concatenate(np.array(X_pred))
    return X_pred_flat.dot(X_flat) / np.sqrt(X_pred_flat.dot(X_pred_flat) * X_flat.dot(X_flat))


def exp_var(X, X_pred):
    return 1 - (np.sum((np.array(X) - np.array(X_pred)) ** 2) / np.sum(np.array(X) ** 2))


def rss(X, X_pred):
    return np.sum((np.array(X) - np.array(X_pred)) ** 2)


def l2norm(X, X_pred):
    return np.sqrt(np.sum((np.array(X) - np.array(X_pred)) ** 2))
