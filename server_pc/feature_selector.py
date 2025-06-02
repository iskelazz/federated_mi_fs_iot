import numpy as np
import time

def calculate_mi_for_feature(p_XY_2D_table):
    """
    Calcula la Información Mutua I(X;Y) para una característica X y la clase Y.
    Utiliza logaritmo base 2 para expresar la MI en bits.
    """
    if p_XY_2D_table is None or p_XY_2D_table.ndim != 2:
        return 0.0

    p_XY_work = np.copy(p_XY_2D_table)
    p_XY_clean = p_XY_work[np.sum(p_XY_work, axis=1) > 0, :]
    if p_XY_clean.shape[0] == 0: return 0.0
    p_XY_clean = p_XY_clean[:, np.sum(p_XY_clean, axis=0) > 0]
    if p_XY_clean.shape[1] == 0: return 0.0

    if np.sum(p_XY_clean) < 0:
        return 0.0
    
    p_X = np.sum(p_XY_clean, axis=1) 
    p_Y = np.sum(p_XY_clean, axis=0) 

    mi_val = 0.0
    for i in range(p_XY_clean.shape[0]): 
        for j in range(p_XY_clean.shape[1]): 
            if p_XY_clean[i, j] > 1e-12 and p_X[i] > 0 and p_Y[j] > 0:
                mi_val += p_XY_clean[i, j] * np.log2(p_XY_clean[i, j] / (p_X[i] * p_Y[j]))
    
    return mi_val if mi_val > 1e-12 else 0.0


def select_features_mim(p_XY_data_array_3D, top_k=15):
    """
    Realiza la selección de características utilizando el método MIM.
    """

    if p_XY_data_array_3D is None or p_XY_data_array_3D.ndim != 3 or p_XY_data_array_3D.shape[0] == 0:
        print("FS_MODULE (MIM): Datos P(Xi,Y) de entrada inválidos o vacíos.")
        return None, 0.0

    start_time = time.time()
    num_features = p_XY_data_array_3D.shape[0]
    mi_scores = np.zeros(num_features)

    for i in range(num_features):
        try:
            current_mi_score = calculate_mi_for_feature(p_XY_data_array_3D[i, :, :])
            mi_scores[i] = current_mi_score 
        except Exception as e_mi:
            print(f"FS_MODULE (MIM): Error calculando MI para característica {i}: {e_mi}.")
            mi_scores[i] = 0.0 

    valid_feature_indices = np.where(mi_scores > 0)[0] 
    if len(valid_feature_indices) == 0:
        print("FS_MODULE (MIM): No se encontraron características con scores MI positivos significativos.")
        return [], time.time() - start_time 

    scores_to_sort = mi_scores[valid_feature_indices]
    sorted_indices_within_valid_subset = np.argsort(scores_to_sort)[::-1]
    selected_original_indices = valid_feature_indices[sorted_indices_within_valid_subset]
    
    final_selected_indices = selected_original_indices[:top_k].tolist()
    elapsed_time = time.time() - start_time
    
    print(f"FS_MODULE (MIM): Selección completada en {elapsed_time:.4f}s. "
          f"Seleccionadas {len(final_selected_indices)}/{min(top_k, len(selected_original_indices))} características.")
    return final_selected_indices, elapsed_time


def calculate_mi_for_triplet(p_XkXjY_table_3D):
    """
    Calcula la Información Mutua I( (Xk,Xj); Y ) para un par de características (Xk,Xj) y la clase Y.
    """

    p_XkXj_2D = np.sum(p_XkXjY_table_3D, axis=2) 
    p_Y_1D = np.sum(p_XkXjY_table_3D, axis=(0, 1))

    mi_val = 0.0
    num_bins_k, num_bins_j, num_classes = p_XkXjY_table_3D.shape

    for i in range(num_bins_k):      
        for j in range(num_bins_j):  
            if p_XkXj_2D[i, j] > 0:
                for c in range(num_classes): 
                    if p_XkXjY_table_3D[i, j, c] > 1e-12 and p_Y_1D[c] > 1e-12:
                        term_p_triplet = p_XkXjY_table_3D[i, j, c]
                        term_p_pair_marginal = p_XkXj_2D[i, j]    
                        term_p_class_marginal = p_Y_1D[c]        
                        mi_val += term_p_triplet * np.log2(term_p_triplet / (term_p_pair_marginal * term_p_class_marginal))
    
    return mi_val if mi_val > 0 else 0.0
