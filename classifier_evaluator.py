import json
import os
import sys
import numpy as np
import time # Añadido para los tiempos de entrenamiento/predicción
import scipy.io as sp 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

# --- Configuración para asegurar que se encuentren utils.py y los datasets ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = SCRIPT_DIR
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
except NameError:
    PROJECT_ROOT = os.getcwd()
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

try:
    from utils import load_dataset
except ImportError as e:
    print(f"Error importando 'load_dataset' desde 'utils.py': {e}")
    sys.exit(1)

def load_simulation_config(project_root_path, config_filename="config.json"):
    """Carga la configuración de simulación desde un archivo JSON."""
    config_filepath = os.path.join(project_root_path, config_filename)
    default_config = {
        "DATASET_NAME": "mnist",
        "CLASSIFIER_CHOICE": "KNN",
        "TEST_SPLIT_RATIO":0.3,
        "SCALE_FEATURES": True,
        "USE_ALL_FEATURES": True,
        "ITERATIONS": 5,
        "FILE_NAME_FS": "mnist_federated_selected_top75_JMI_federated_feature_indices.txt"
    }
    try:
        with open(config_filepath, 'r') as f:
            all_config = json.load(f)
        print(f"Configuración cargada desde '{config_filepath}'.")
        config = all_config.get("CLASSIFIER")
        if config is None:
            print(f"Advertencia: La clave 'CLASSIFIER' no se encontró en '{config_filepath}'. "
                  f"Usando la configuración por defecto completa para 'CLASSIFIER'.")
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

def load_predefined_test_set(dataset_base_name, project_root_path):
    """
        Carga los dataset con conjunto de test independiente.
    """
    print(f"Intentando cargar conjunto de test predefinido para '{dataset_base_name}'...")
    X_test, y_test = None, None
    
    dataset_specific_folder = os.path.join(project_root_path, 'datasets', dataset_base_name)
    test_data_path = os.path.join(dataset_specific_folder, dataset_base_name + '_test.data')
    test_labels_path = os.path.join(dataset_specific_folder, dataset_base_name + '_test.labels')

    if os.path.exists(test_data_path) and os.path.exists(test_labels_path):
        print(f"Encontrados archivos _test.data y _test.labels para '{dataset_base_name}'.")
        try:
            X_test = np.loadtxt(test_data_path, dtype=np.int16).astype(np.float32)
            y_test_raw = np.loadtxt(test_labels_path, dtype=np.int16)
            
            if np.any(y_test_raw < 0): 
                 y_test_raw[y_test_raw < 0] = 0
            _, _, y_test = np.unique(y_test_raw, return_index=True, return_inverse=True)

            print(f"Cargado X_test: {X_test.shape}, y_test: {y_test.shape} desde archivos .data/.labels")
            return X_test, y_test
        except Exception as e:
            print(f"Error cargando test set tipo _bin para '{dataset_base_name}': {e}")
            return None, None

    test_mat_path = os.path.join(project_root_path, 'datasets', dataset_base_name + '_test.mat')
    if os.path.exists(test_mat_path):
        print(f"Encontrado archivo _test.mat para '{dataset_base_name}'.")
        try:
            data_mat = sp.loadmat(test_mat_path)
            if 'test_data' in data_mat and 'test_labels' in data_mat:
                X_test = np.float32(data_mat["test_data"])
                y_test_raw = np.float32(data_mat["test_labels"]).ravel()
            elif 'data' in data_mat and 'labels' in data_mat: 
                X_test = np.float32(data_mat["data"])
                y_test_raw = np.float32(data_mat["labels"]).ravel()
            else:
                print(f"Error: El archivo .mat de test '{test_mat_path}' no contiene las claves esperadas.")
                return None, None
            _, _, y_test = np.unique(y_test_raw, return_index=True, return_inverse=True)
            print(f"Cargado X_test: {X_test.shape}, y_test: {y_test.shape} desde archivo .mat")
            return X_test, y_test
        except Exception as e:
            print(f"Error cargando test set tipo .mat para '{dataset_base_name}': {e}")
            return None, None
            
    print(f"No se encontró un conjunto de test predefinido para '{dataset_base_name}' con las convenciones esperadas.")
    return None, None

def load_selected_feature_indices(filepath):
    """
        Carga los dataset sin conjunto de test independiente.
    """
    if not os.path.exists(filepath):
        print(f"Error: El archivo de índices de características '{filepath}' no fue encontrado.")
        return None
    try:
        with open(filepath, 'r') as f:
            indices = [int(line.strip()) for line in f if line.strip()]
        if not indices:
            print(f"Advertencia: El archivo de índices '{filepath}' está vacío o no contiene índices válidos.")
            return []
        return np.array(indices)
    except Exception as e:
        print(f"Error leyendo el archivo de índices '{filepath}': {e}")
        return None

def evaluate_classifier(X_train, X_test, y_train, y_test, classifier_name, dataset_name_log):
    """
        Ejecuta el proceso de clasificación, posteriormente muestra los resultados para un conjunto de metricas.
    """
    print(f"\n--- Evaluando Clasificador: {classifier_name} en Dataset: {dataset_name_log} ---")
    model = None
    train_time_start = time.time() # Mover inicio de tiempo de entrenamiento aquí

    if classifier_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif classifier_name == "RF":
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    elif classifier_name == "LOGISTIC_REGRESSION":
        model = LogisticRegression(max_iter=1000, solver='liblinear')
    elif classifier_name == "NAIVE_BAYES":
        model = GaussianNB()
    else:
        print(f"Error: Clasificador '{classifier_name}' no reconocido.")
        return

    print("Entrenando el modelo...")
    model.fit(X_train, y_train)
    train_time_end = time.time()
    train_time = train_time_end - train_time_start
    print(f"Tiempo de entrenamiento: {train_time:.4f} segundos.")

    print("Realizando predicciones en el conjunto de prueba...")
    predict_time_start = time.time()
    y_pred = model.predict(X_test)
    predict_time_end = time.time()
    predict_time = predict_time_end - predict_time_start
    print(f"Tiempo de predicción: {predict_time:.4f} segundos.")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nExactitud (Accuracy): {accuracy:.4f}")

    print("\nReporte de Clasificación:")
    try:
        all_unique_labels = np.unique(np.concatenate((y_train, y_test)))
        target_names = [f"Clase {i}" for i in all_unique_labels]
        report = classification_report(y_test, y_pred, labels=all_unique_labels, target_names=target_names, zero_division=0)
    except Exception as e_report:
        print(f"Advertencia: No se pudo generar reporte con target_names ({e_report}). Usando reporte por defecto.")
        report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    print("\nMatriz de Confusión:")
    try:
        cm = confusion_matrix(y_test, y_pred, labels=all_unique_labels)
    except Exception:
        cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_test, y_pred, average='micro', zero_division=0)
    #precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    #precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    print("\nMétricas Promediadas Detalladas:")
    print(f"  Precision (Micro):   {precision_micro:.4f}")
    print(f"  Recall (Micro):      {recall_micro:.4f}")
    print(f"  F1-score (Micro):    {f1_micro:.4f}")
    
    results = {
        "accuracy": accuracy,
        "train_time": train_time,
        "predict_time": predict_time
    }
    return results

def main():
    cfg = load_simulation_config(PROJECT_ROOT)

    # Extraer los parámetros de la configuración cargada
    dataset_name = cfg["DATASET_NAME"]
    classifier_choice = cfg["CLASSIFIER_CHOICE"]
    test_split_ratio = cfg["TEST_SPLIT_RATIO"]
    scale_features = cfg["SCALE_FEATURES"]
    use_all_features = cfg["USE_ALL_FEATURES"]
    iterations = cfg["ITERATIONS"]
    file_name_for_selected_features = cfg["FILE_NAME_FS"]
    
    print(f"--- Iniciando Evaluación de Clasificador ---")
    print(f"Dataset: {dataset_name}")
    print(f"Clasificador Elegido: {classifier_choice}")
    print(f"Usar todas las características: {'Sí' if use_all_features else 'No'}")

    if not use_all_features:
        # Actualizar la ruta del archivo de características para usar el DATASET_NAME configurado
        current_selected_features_file_path = os.path.join(PROJECT_ROOT, "selected_features", file_name_for_selected_features)
        # O ajusta la subcarpeta si es para resultados federados:
        print(f"Archivo de Características Seleccionadas: {current_selected_features_file_path}")
    
    print(f"Ratio de División para Prueba: {test_split_ratio}")
    print(f"Escalar Características: {'Sí' if scale_features else 'No'}")

    selected_indices = None
    if not use_all_features:
        selected_indices = load_selected_feature_indices(current_selected_features_file_path)
        if selected_indices is None or len(selected_indices) == 0:
            print(f"No se pudieron cargar los índices de características o no hay características seleccionadas. Abortando.")
            return
        print(f"Se usarán {len(selected_indices)} características seleccionadas.")
    else:
        print("Se usarán todas las características disponibles del dataset.")


    X_global_raw, y_global_raw = None, None # Para el caso de train_test_split

    
    try:
        print(f"Usando train_test_split para '{dataset_name}'.")
        X_global_raw, y_global_raw = load_dataset(dataset_name)
        if X_global_raw is None or y_global_raw is None: raise ValueError("load_dataset devolvió None.")
        print(f"Dataset '{dataset_name}' cargado. Forma original de X: {X_global_raw.shape}, Forma de y: {y_global_raw.shape}")
    except Exception as e:
        print(f"Error cargando datos para '{dataset_name}': {e}")
        return
    results_acuraccy = []
    results_train_time = []
    results_predict_time = []
    for i in range(iterations):
        X_train_final, X_test_final = None, None
        y_train_final, y_test_final = None, None

        # --- Aplicar selección de características o usar todas ---
        if not use_all_features:
            if selected_indices is None: 
                print("Error: Se indicó usar características seleccionadas pero los índices no están disponibles.")
                return
            try:
                # Validar que los índices no excedan las dimensiones del dataset cargado
                max_dim_check = X_global_raw.shape[1]
                if np.any(selected_indices >= max_dim_check) or np.any(selected_indices < 0):
                    print(f"Error: Índices de características ({np.min(selected_indices)}-{np.max(selected_indices)}) fuera de rango (0 a {max_dim_check-1}).")
                    return
                
                X_selected_global = X_global_raw[:, selected_indices]
                X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
                    X_selected_global, y_global_raw, test_size=test_split_ratio,
                    )
                print(f"Forma de X_train después de seleccionar características: {X_train_final.shape}")
                if X_test_final is not None: print(f"Forma de X_test después de seleccionar características: {X_test_final.shape}")

            except IndexError as e:
                print(f"Error de Indexación al seleccionar características: {e}")
                return
            except Exception as e:
                print(f"Error inesperado durante la selección de características: {e}")
                return
        else: # Usar todas las características
            X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
                X_global_raw, y_global_raw, test_size=test_split_ratio,
                stratify=y_global_raw if len(np.unique(y_global_raw)) > 1 else None
            )
            print(f"Usando todas las características. Forma de X_train: {X_train_final.shape}")
            if X_test_final is not None: print(f"Forma de X_test: {X_test_final.shape}")


        if X_train_final is None or X_test_final is None or y_train_final is None or y_test_final is None:
            print("Error: Los conjuntos de datos finales (train/test) no se generaron correctamente.")
            return

        # Escalar características
        if scale_features:
            print("Escalando características (StandardScaler)...")
            scaler = StandardScaler()
            X_train_final = scaler.fit_transform(X_train_final)
            X_test_final = scaler.transform(X_test_final)

        # Entrenar y evaluar el clasificador
        results = evaluate_classifier(X_train_final, X_test_final, y_train_final, y_test_final, classifier_choice, dataset_name)
        
        # Almacenar parametros para evaluación final
        results_acuraccy.append(results["accuracy"])
        results_train_time.append(results["train_time"])
        results_predict_time.append(results["predict_time"])

        print(f"\n--- Evaluación de Clasificador Finalizada ---")
    mean_accuracy = np.mean(results_acuraccy)
    std_accuracy = np.std(results_acuraccy)
    mean_train_time = np.mean(results_train_time)
    mean_predict_time = np.mean(results_predict_time)
        
    print(f"\n\n--- RESULTADOS FINALES DE LAS {len(results_acuraccy)} ITERACIONES ---")
    print(f"Exactitud (Accuracy): Media = {mean_accuracy:.4f}, Desviación Típica = {std_accuracy:.4f}")
    print(f"Tiempo medio: Entrenamiento = {mean_train_time:.4f} segundos, Predicción: {mean_predict_time:.4f} segundos")
    

if __name__ == "__main__":
    main()