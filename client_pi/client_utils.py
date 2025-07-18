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
    
    
def calculate_local_triplet_prob_dist_monolithic_style(X_client_discretized, y_client_partition, k_idx, j_idx, num_bins, num_classes, sim_client_id_for_log="N/A_CLIENT"):
    """
    Calcula P_l(X_k, X_j, Y) replicando la lógica ineficiente de un bucle
    de Python sobre las muestras, similar al script monolítico.

    Esta función es intencionadamente más lenta que la versión vectorizada con
    np.histogramdd para fines de comparación y análisis de escalabilidad.
    """
    n_samples = X_client_discretized.shape[0]
    if n_samples == 0:
        return np.zeros((num_bins, num_bins, num_classes), dtype=float)

    counts_3d = np.zeros((num_bins, num_bins, num_classes), dtype=int)

    for i in range(n_samples):
        val_k = X_client_discretized[i, k_idx]
        val_j = X_client_discretized[i, j_idx]
        val_y = y_client_partition[i]
        counts_3d[val_k, val_j, val_y] += 1

    if n_samples > 0:
        prob_dist_3d = counts_3d / n_samples
    else:
        prob_dist_3d = counts_3d.astype(float)

    return prob_dist_3d


def calculate_local_triplet_prob_dist_bincount(X_client_discretized, y_client_partition, k_idx, j_idx, num_bins, num_classes):
    """
    Calcula P_l(X_k, X_j, Y) de la forma más eficiente posible usando np.bincount.
    """
    n_samples = X_client_discretized.shape[0]
    if n_samples == 0:
        return np.zeros((num_bins, num_bins, num_classes), dtype=float)

    x_k = X_client_discretized[:, k_idx]
    x_j = X_client_discretized[:, j_idx]
    y = y_client_partition

    raveled_indices = x_k * (num_bins * num_classes) + x_j * num_classes + y
    
    total_bins = num_bins * num_bins * num_classes
    counts_1d = np.bincount(raveled_indices, minlength=total_bins)

    counts_3d = counts_1d.reshape((num_bins, num_bins, num_classes))

    prob_dist_3d = counts_3d / n_samples
    
    return prob_dist_3d

import numpy as np

def calculate_local_prob_dist_array_bincount(X_discretized, y_labels, num_bins, num_classes, sim_client_id_for_log="N/A_CLIENT"):
    """
    Calcula P(Xi, Y) para todas las características de forma vectorizada usando np.bincount.
    
    Este método es mucho más eficiente que un bucle sobre np.histogram2d porque
    procesa todas las características a la vez. Aplana las coordenadas 3D 
    (feature_idx, bin_value, class_label) en un único índice 1D y realiza un 
    único conteo masivo.
    """
    # --- Validación de entradas (idéntica a la original) ---
    if X_discretized is None or y_labels is None or not isinstance(num_bins, int) or not isinstance(num_classes, int) or num_bins <= 0 or num_classes <= 0:
        print(f"[{sim_client_id_for_log}]: Entradas inválidas.")
        return np.array([])

    if X_discretized.ndim != 2 or X_discretized.shape[0] != y_labels.shape[0]:
        print(f"[{sim_client_id_for_log}]: Dimensiones inconsistentes.")
        return np.array([])
        
    n_samples, n_features = X_discretized.shape
    if n_samples == 0:
        return np.zeros((n_features, num_bins, num_classes), dtype=float)

    feature_indices = np.tile(np.arange(n_features), n_samples)
    bin_values = X_discretized.ravel()
    class_labels = np.repeat(y_labels, n_features)

    raveled_indices = feature_indices * (num_bins * num_classes) + bin_values * num_classes + class_labels
    
    total_bins = n_features * num_bins * num_classes
    counts_1d = np.bincount(raveled_indices.astype(int), minlength=total_bins)
    
    counts_3d = counts_1d.reshape((n_features, num_bins, num_classes))
    
    local_p_xy = counts_3d / n_samples
    
    return local_p_xy