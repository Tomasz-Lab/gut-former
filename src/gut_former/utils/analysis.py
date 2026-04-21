import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes
from skbio.stats.ordination import pcoa


def _to_pcoa(arr):
    """Compute PCoA coordinates from a 2D array."""
    return pcoa(squareform(pdist(arr))).samples


def get_pcoa_coords(true_df, pred_df):
    """Combine true and predicted into one PCoA space for visualization."""
    true_arr = true_df.values.clip(min=0)
    pred_arr = pred_df.values.clip(min=0)
    return _to_pcoa(np.vstack([true_arr, pred_arr]))


def get_disparity(true_df, pred_df):
    """Procrustes disparity between separately computed PCoAs."""
    true_arr = true_df.values.clip(min=0)
    pred_arr = pred_df.values.clip(min=0)
    _, _, disparity = procrustes(_to_pcoa(true_arr), _to_pcoa(pred_arr))
    return disparity
