# -*- coding: utf-8 -*-

from typing import Dict, Tuple, Union, List
import numpy as np
import time
import scipy.io as sp
import os

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def load_dataset_bin(name):
    info = {
        'train': {}, 'validation': {}
    }

    dataset_specific_folder = os.path.join(PROJECT_ROOT, 'datasets', name)

    file_path = os.path.join(dataset_specific_folder, name + '_train.labels')
    info['train']['label'] = np.loadtxt(file_path, dtype=np.int16)
    info['train']['label'][info['train']['label'] < 0] = 0

    file_path = os.path.join(dataset_specific_folder, name + '_train.data')
    info['train']['data'] = np.loadtxt(file_path, dtype=np.int16).astype(np.float32)

    file_path = os.path.join(dataset_specific_folder, name + '_valid.labels')
    info['validation']['label'] = np.loadtxt(file_path, dtype=np.int16)
    info['validation']['label'][info['validation']['label'] < 0] = 0

    file_path = os.path.join(dataset_specific_folder, name + '_valid.data')
    info['validation']['data'] = np.loadtxt(file_path, dtype=np.int16).astype(np.float32)

    X = np.concatenate((info['train']['data'], info['validation']['data']))
    y = np.concatenate((info['train']['label'], info['validation']['label']))
    _, _, y = np.unique(y, return_index=True,return_inverse=True)

    return (X, y)

def load_dataset_mat(name) -> Tuple: 
    file_path = os.path.join(PROJECT_ROOT, 'datasets', name + '.mat')
    data = sp.loadmat(file_path)
    X = np.float32(data["data"])
    y = np.float32(data["labels"]).ravel() 
    _, _, y = np.unique(y, return_index=True,return_inverse=True)
    return (X, y)

def load_dataset(dataset_name) -> Tuple:
    if dataset_name == "mnist":
        data_tuple = load_dataset_mat("MNIST_et")
    elif dataset_name == "internet_ads":
        data_tuple = load_dataset_mat("internet_ads")
    elif dataset_name == "human":
        data_tuple = load_dataset_mat("humanActivity")
    elif dataset_name == "gas_sensor":
        data_tuple = load_dataset_mat("gas_sensor")
    elif dataset_name == "gisette":
        data_tuple = load_dataset_bin("gisette")
    elif dataset_name == "arcene":
        data_tuple = load_dataset_bin("arcene")
    elif dataset_name == "madelon":
        data_tuple = load_dataset_bin("madelon")
    else:
        raise Exception("Invalid dataset", dataset_name) 
    return data_tuple

def build_iid_data(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        user_items = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - user_items)
        dict_users[i] = list(user_items)
    return dict_users


def build_noniid_data(dataset, labels, num_users):
    num_shards, num_imgs = 2 * num_users, len(labels) // (2 * num_users)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels[0:len(idxs)]))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        dict_users[i]  = list(dict_users[i].astype('int'))
    return dict_users

def partition(X: np.ndarray, y: np.ndarray, users: dict) -> XYList:
    data = []
    for user in users.values():
        data.append((X[user,:],y[user]))
    return list(data)


def disc_equalwidth(X: np.ndarray, bins: int):
    disc_X = np.zeros(X.shape, dtype=int)
    for fea_index in range(X.shape[1]):
        fea_values = X[:,fea_index]
        bin_edges = np.linspace(fea_values.min(), fea_values.max()+1e-10, bins, endpoint=False)
        disc_fea = np.digitize(fea_values, bin_edges, right=False)
        _,_,disc_fea = np.unique(disc_fea, axis=0, return_index=True, return_inverse=True)
        disc_X[:, fea_index] = disc_fea
    return disc_X

def discretize_with_global_ranges(X: np.ndarray, bins: int, feature_ranges: List[List[float]]):
    disc_X = np.zeros(X.shape, dtype=int)
    num_features = X.shape[1]

    if not feature_ranges or len(feature_ranges) != num_features:
        raise ValueError(
            f"feature_ranges debe ser una lista con un rango [min, max] para cada una de las {num_features} características."
        )

    for fea_index in range(num_features):
        fea_values = X[:, fea_index]
        min_g, max_g = feature_ranges[fea_index]

        if min_g == max_g:
            disc_X[:, fea_index] = 0
            continue

        current_bin_edges = np.linspace(min_g, max_g, bins + 1, endpoint=True)
        if current_bin_edges[-1] == max_g :
             current_bin_edges[-1] += 1e-9
        discretized_feature = np.digitize(fea_values, current_bin_edges, right=False)
        discretized_feature = discretized_feature - 1 
        discretized_feature[fea_values < min_g] = 0 
        discretized_feature[fea_values >= max_g] = bins - 1
        discretized_feature[discretized_feature < 0] = 0
        discretized_feature[discretized_feature >= bins] = bins - 1
        
        disc_X[:, fea_index] = discretized_feature
        
    return disc_X


def calculate_uneven_quotas(total_samples: int, num_users: int, unevenness_factor: float) -> List[int]:
    """
    Calcula cuotas de muestras (enteros) para cada usuario de forma desbalanceada,
    asegurando que la suma de las cuotas sea igual a total_samples.
    """

    if not (0.0 <= unevenness_factor < 1.0):
         raise ValueError("unevenness_factor debe estar en el rango [0.0, 1.0).")

    if unevenness_factor == 0.0 or num_users == 1:
        base_quota = total_samples // num_users
        remainder = total_samples % num_users
        quotas = [base_quota] * num_users
        for i in range(remainder):
            quotas[i % num_users] += 1 # Distribuir remanente de forma más equitativa
        return quotas

    # Generar pesos aleatorios y normalizarlos
    # Usar un min_scale_factor pequeño pero >0 para que los clientes con pocas muestras
    min_raw_weight = 0.01 if total_samples >= num_users else 0.0
    raw_weights = np.random.uniform(max(min_raw_weight, 1.0 - unevenness_factor), 1.0 + unevenness_factor, num_users)

    if np.sum(raw_weights) == 0: 
        return calculate_uneven_quotas(total_samples, num_users, 0.0)

    normalized_weights = raw_weights / np.sum(raw_weights)
    quotas_float = normalized_weights * total_samples

    # Método del resto mayor (Hamilton) para redondear a enteros
    quotas_int = np.floor(quotas_float).astype(int)
    remainder_to_distribute = total_samples - np.sum(quotas_int)

    if remainder_to_distribute > 0: # Debería ser siempre >= 0 si se usa floor
        fractional_parts = quotas_float - quotas_int
        indices_to_increment = np.argsort(fractional_parts)[-remainder_to_distribute:]
        for idx in indices_to_increment:
            quotas_int[idx] += 1
    
        # Corrección simple: poner cuotas negativas a 0 y redistribuir el déficit
        total_deficit = 0
        for i in range(len(quotas_int)):
            if quotas_int[i] < 0:
                total_deficit += abs(quotas_int[i])
                quotas_int[i] = 0
        
        if total_deficit > 0:
            positive_indices = [i for i, q_val in enumerate(quotas_int) if q_val > 0]
            if positive_indices:
                # Por simplicidad, quitamos secuencialmente de las positivas hasta cubrir el déficit.
                # Esto es una simplificación y podría no ser ideal.
                for _ in range(total_deficit):
                    if not positive_indices: break 
                    idx_to_reduce = positive_indices[0] # Tomar de la primera positiva
                    if quotas_int[idx_to_reduce] > 0:
                        quotas_int[idx_to_reduce] -=1
                        if quotas_int[idx_to_reduce] == 0:
                            positive_indices.pop(0)
                    else: 
                        positive_indices.pop(0) 
            else: 
                  print("Error crítico en la corrección de cuotas negativas, recurriendo a equitativo.")
                  return calculate_uneven_quotas(total_samples, num_users, 0.0)

    # Última comprobación de suma, si todo lo demás falló.
    final_sum_check = np.sum(quotas_int)
    if final_sum_check != total_samples:
        quotas_int[0] += (total_samples - final_sum_check)
        if quotas_int[0] < 0: 
            quotas_int[0] = 0
            if total_samples > 0 and sum(quotas_int) != total_samples:
                 return calculate_uneven_quotas(total_samples, num_users, 0.0)

    return quotas_int.tolist()


def build_noniid_uneven_no_loss(labels: np.ndarray, num_users: int, unevenness_factor: float = 0.5) -> Dict[int, List[int]]:
    total_samples = len(labels)
    all_original_indices = np.arange(total_samples)

    if total_samples == 0:
        return {i: [] for i in range(num_users)}
    if num_users == 0:
        return {}

    # 1. Ordenar todos los índices originales según las etiquetas
    sorted_global_indices = all_original_indices[np.argsort(labels, kind='mergesort')]

    # 2. Calcular las cuotas (número de muestras) para cada usuario
    user_quotas = calculate_uneven_quotas(total_samples, num_users, unevenness_factor)

    # 3. Distribuir bloques contiguos de la "pila ordenada"
    dict_users = {}
    current_pointer = 0
    for user_id in range(num_users):
        num_samples_for_this_user = user_quotas[user_id]

        start_index = current_pointer
        end_index = current_pointer + num_samples_for_this_user

        # Asegurar que los índices no se salgan del array (no debería si las cuotas suman bien)
        if end_index > total_samples:
            end_index = total_samples 
        
        dict_users[user_id] = sorted_global_indices[start_index:end_index].tolist()
        current_pointer = end_index

    return dict_users