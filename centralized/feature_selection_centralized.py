import json
import os
import sys
import time
import numpy as np

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
except NameError:
    # __file__ no está definido, probablemente se está ejecutando en un entorno interactivo
    # Usar el directorio de trabajo actual como PROJECT_ROOT si no está ya en sys.path
    PROJECT_ROOT = os.getcwd() # Guardamos el CWD para usarlo consistentemente
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

try:
    from utils import load_dataset, discretize_equalwidth
    from mutual_information import MIM, JMI
except ImportError as e:
    print(f"Error importando módulos desde la raíz del proyecto ('{PROJECT_ROOT}'): {e}")
    print("Asegúrate de que la estructura de tu proyecto es correcta y que utils.py y mutual_information.py están accesibles.")
    exit(1)

# --- Parámetros de Configuración ---
#DATASET_NAME = "arcene"
#TOP_K_FEATURES = 75
#N_BINS_DISCRETIZATION = 5
#MI_TECHNIQUE_FUNCTION = JMI # Puedes cambiar esto a MIM si lo deseas


def load_simulation_config(project_root_path, config_filename="config.json"):
    """Carga la configuración de simulación desde un archivo JSON."""
    config_filepath = os.path.join(project_root_path, config_filename)
    default_config = {
        "DATASET_TO_LOAD_GLOBALLY": "arcene",
        "TOP_K_FEATURES_TO_SELECT": 75,
        "NUM_BINS": 5,
        "MI_FS_METHOD": "JMI"
    }
    try:
        with open(config_filepath, 'r') as f:
            all_config = json.load(f)
        print(f"Configuración cargada desde '{config_filepath}'.")
        config = all_config.get("FS_CENTRALIZED")
        if config is None:
            print(f"Advertencia: La clave 'FS_CENTRALIZED' no se encontró en '{config_filepath}'. "
                  f"Usando la configuración por defecto completa para 'FS_CENTRALIZED'.")
            # Si "FEATURE_SELECTION" no está, devolvemos el default completo para esta sección.
            return config
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
                print(f"Advertencia: Usando valor por defecto para '{key}': {default_config[key]}")
        return config
    except Exception as e:
        print(f"Error cargando configuración desde '{config_filepath}': {e}. Usando configuración por defecto.")
        return default_config

def load_and_prepare_data(dataset_name_to_load, n_bins_for_discretization):
    """
    Carga el dataset, asegura la forma correcta de X y lo discretiza.
    Devuelve X_original, X_discretizado, y, o (None, None, None) en caso de error.
    """
    try:
        X, y, _ = load_dataset(dataset_name_to_load)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del dataset para '{dataset_name_to_load}' según lo configurado en utils.py.")
        return None, None, None
    except Exception as e:
        print(f"Ocurrió un error al cargar el dataset '{dataset_name_to_load}': {e}")
        return None, None, None

    if X.shape[0] == 0 or X.shape[1] == 0:
        print(f"Error: X para '{dataset_name_to_load}' está vacío (0 muestras o 0 características).")
        return None, None, None

    # Discretizar características
    try:
        X_discrete = discretize_equalwidth(X.astype(np.float32), bins=n_bins_for_discretization)
    except Exception as e:
        print(f"Error durante la discretización de X para '{dataset_name_to_load}': {e}")
        return None, None, None

    return X, X_discrete, y

def select_features_centralized(X_discrete_data, y_labels,
                                mi_function, top_k, n_original_features):
    """
    Aplica la selección de características centralizada.
    Devuelve los índices de las características seleccionadas (ordenados por relevancia) o None en caso de error.
    """
    effective_top_k = min(top_k, n_original_features)
    try:
        # Las funciones MIM y JMI deben devolver los índices ya ordenados por relevancia
        selected_indices = mi_function(X_discrete_data, y_labels, topK=effective_top_k)
        print(f"   Índices de las características seleccionadas (Primeras 20 de {len(selected_indices)}): {selected_indices[:20]}...")
        return selected_indices
    except Exception as e:
        print(f"Ocurrió un error durante la selección de características con {mi_function.__name__}: {e}")
        return None

def save_selected_features_txt(selected_feature_indices,
                               dataset_name_str, top_k_val, technique_name,
                               project_root_path):
    """
    Guarda los índices de las características seleccionadas (ordenados por relevancia) en un archivo .txt.
    """
    main_datasets_folder = "selected_features"
    #selected_subfolder = "datasets_seleccionados_central" # Misma subcarpeta para consistencia

    output_dir = os.path.join(project_root_path, main_datasets_folder)
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{dataset_name_str}_centralized_selected_top{top_k_val}_{technique_name}_feature_indices.txt"
    output_filepath = os.path.join(output_dir, output_filename)

    try:
        with open(output_filepath, 'w') as f:
            for feature_index in selected_feature_indices:
                f.write(f"{feature_index}\n")
        print(f"Índices de características seleccionadas (.txt) guardados en: {output_filepath}")
    except Exception as e:
        print(f"Error guardando el archivo .txt de índices de características en '{output_filepath}': {e}")


def main():
    cfg = load_simulation_config(PROJECT_ROOT)

    # Extraer los parámetros de la configuración cargada
    dataset_name = cfg["DATASET_TO_LOAD_GLOBALLY"]
    top_k_features = cfg["TOP_K_FEATURES_TO_SELECT"]
    n_bins_discretization = cfg["NUM_BINS"]
    mi_fs_method = cfg["MI_FS_METHOD"]
    mi_function = None
    if mi_fs_method.upper() == "JMI":
        mi_function = JMI  
    elif mi_fs_method.upper() == "MIM":
        mi_function = MIM
    else:
        print("Error: Técnica de IM {mi_fs_method} incorrecta")
        return
    
    global_start_time = time.time()
    print(f"--- Iniciando Selección de Características Centralizada para el dataset: {dataset_name} ---")
    print(f"Usando Técnica de MI: {mi_function.__name__}")
    print(f"Top K Características a seleccionar: {top_k_features}")
    print(f"Número de bins para discretización: {n_bins_discretization}\n")

    X_original, X_discrete, y = load_and_prepare_data(dataset_name, n_bins_discretization)
    if X_original is None: # Si la carga o preparación falló
        print(f"--- Proceso abortado para {dataset_name} debido a errores en la carga/preparación ---")
        return

    n_features_original = X_original.shape[1]

    selected_indices = select_features_centralized(X_discrete, y,
                                                 mi_function,
                                                 top_k_features,
                                                 n_features_original)

    if selected_indices is None:
        print(f"--- Proceso abortado para {dataset_name} debido a errores en la selección de características ---")
        return

    actual_top_k = min(top_k_features, n_features_original)

    # Guardar los índices de las características seleccionadas en formato .txt
    save_selected_features_txt(selected_indices,
                               dataset_name, actual_top_k,
                               mi_function.__name__,
                               PROJECT_ROOT)
    
    global_end_time = time.time()
    total_elapsed_time = global_end_time - global_start_time
    print(f"--- TIEMPO TOTAL DE EJECUCIÓN DEL SERVIDOR: {total_elapsed_time:.4f} segundos ---")

    print(f"\n--- Proceso de Selección de Características Centralizada para {dataset_name} finalizado ---")

if __name__ == "__main__":
    main()