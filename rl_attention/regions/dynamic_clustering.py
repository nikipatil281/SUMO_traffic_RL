import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans

# Ensure project root on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RL_ROOT = os.path.dirname(THIS_DIR)      # .../rl_attention
PROJECT_ROOT = os.path.dirname(RL_ROOT)  # .../generating_traffic_mac
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def cluster_tls_by_embedding(
    emb_dict: Dict[str, np.ndarray],
    n_clusters: int,
    random_state: int = 0,
) -> Dict[str, int]:
    """
    Clusters intersections into regions using k-means on lane-context embeddings.

    Args:
      emb_dict: dict[tid] -> np.ndarray of shape [D]
      n_clusters: number of regions
      random_state: for reproducibility

    Returns:
      region_id: dict[tid] -> int in [0, n_clusters-1]
    """
    tls_ids: List[str] = sorted(emb_dict.keys())
    X = np.stack([emb_dict[tid] for tid in tls_ids], axis=0)  # [N, D]

    # If n_clusters > N, cap it
    k = min(n_clusters, len(tls_ids))
    if k <= 1:
        # Degenerate case: everyone is in region 0
        return {tid: 0 for tid in tls_ids}

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)  # [N]

    return {tid: int(labels[i]) for i, tid in enumerate(tls_ids)}

