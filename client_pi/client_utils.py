import numpy as np

def calculate_local_prob_dist_array(X_discretized, y_labels, num_bins, num_classes, sim_client_id_for_log="N/A_CLIENT"):
    """Calcula P(Xi, Y) usando np.histogram2d."""
    if X_discretized is None or y_labels is None or not isinstance(num_bins, int) or not isinstance(num_classes, int) or num_bins <= 0 or num_classes <= 0:
        return np.array([])

    if X_discretized.ndim != 2 or X_discretized.shape[0] != y_labels.shape[0]:
        return np.array([])
        
    n_samples, n_features = X_discretized.shape
    if n_samples == 0:
        return np.zeros((n_features, num_bins, num_classes), dtype=float)

    local_p_xy = np.zeros((n_features, num_bins, num_classes), dtype=float)
    for feature_idx in range(n_features):
        try:
            counts, _, _ = np.histogram2d(
                X_discretized[:, feature_idx],
                y_labels,
                bins=[num_bins, num_classes],
                range=[[-0.5, num_bins - 0.5], [-0.5, num_classes - 0.5]]
            )
            local_p_xy[feature_idx, :, :] = counts
        except Exception as e:
            print(f"[{sim_client_id_for_log}]: Excepción en histogram2d para feat {feature_idx}: {e}")
            local_p_xy[feature_idx, :, :] = np.zeros((num_bins, num_classes), dtype=float)

    if n_samples > 0:
        local_p_xy /= n_samples
    return local_p_xy

def calculate_local_triplet_prob_dist(X_client_discretized, y_client_partition, k_idx, j_idx, num_bins, num_classes, sim_client_id_for_log="N/A_CLIENT"):
    """Calcula P_l(X_k, X_j, Y) usando np.histogramdd."""

    n_samples = X_client_discretized.shape[0]
    if n_samples == 0:
        return np.zeros((num_bins, num_bins, num_classes), dtype=float)

    try:
        sample_data = (
            X_client_discretized[:, k_idx],
            X_client_discretized[:, j_idx],
            y_client_partition
        )
        p_xyz_local, _ = np.histogramdd(
            sample_data,
            bins=[num_bins, num_bins, num_classes],
            range=[
                [-0.5, num_bins - 0.5],
                [-0.5, num_bins - 0.5],
                [-0.5, num_classes - 0.5]
            ]
        )
        if n_samples > 0:
            p_xyz_local /= n_samples
        return p_xyz_local
    except Exception as e:
        print(f"[{sim_client_id_for_log}]: Excepción en histogramdd para par ({k_idx},{j_idx}): {e}")
        return np.zeros((num_bins, num_bins, num_classes), dtype=float)