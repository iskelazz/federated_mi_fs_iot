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

# --- Parámetros de Configuración Global ---
DATASET_NAME = "arcene"
CLASSIFIER_CHOICE = "KNN"
TEST_SPLIT_RATIO = 0.3
RANDOM_STATE = 42
SCALE_FEATURES = True
USE_ALL_FEATURES = True # Cambia a True para usar el dataset completo
FILE_NAME =  f"{DATASET_NAME}_federated_selected_top75_JMI_federated_feature_indices.txt"
# --------------------------------------------

SELECTED_FEATURES_FILE_PATH = os.path.join(PROJECT_ROOT, "selected_features", FILE_NAME)
DATASETS_WITH_PREDEFINED_TEST = [] # Datasets con conjunto de test separado

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
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    elif classifier_name == "LOGISTIC_REGRESSION":
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver='liblinear')
    elif classifier_name == "NAIVE_BAYES":
        model = GaussianNB()
    else:
        print(f"Error: Clasificador '{classifier_name}' no reconocido.")
        return

    print("Entrenando el modelo...")
    model.fit(X_train, y_train)
    train_time_end = time.time()
    print(f"Tiempo de entrenamiento: {train_time_end - train_time_start:.4f} segundos.")

    print("Realizando predicciones en el conjunto de prueba...")
    predict_time_start = time.time()
    y_pred = model.predict(X_test)
    predict_time_end = time.time()
    print(f"Tiempo de predicción: {predict_time_end - predict_time_start:.4f} segundos.")

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
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    print("\nMétricas Promediadas Detalladas:")
    print(f"  Precision (Micro):   {precision_micro:.4f}")
    print(f"  Recall (Micro):      {recall_micro:.4f}")
    print(f"  F1-score (Micro):    {f1_micro:.4f}")
    print(f"  Precision (Macro):   {precision_macro:.4f}")
    print(f"  Recall (Macro):      {recall_macro:.4f}")
    print(f"  F1-score (Macro):    {f1_macro:.4f}")
    print(f"  Precision (Weighted):{precision_weighted:.4f}")
    print(f"  Recall (Weighted):   {recall_weighted:.4f}")
    print(f"  F1-score (Weighted): {f1_weighted:.4f}")

def main():
    print(f"--- Iniciando Evaluación de Clasificador ---")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Clasificador Elegido: {CLASSIFIER_CHOICE}")
    print(f"Usar todas las características: {'Sí' if USE_ALL_FEATURES else 'No'}")

    if not USE_ALL_FEATURES:
        # Actualizar la ruta del archivo de características para usar el DATASET_NAME configurado
        current_selected_features_file_path = os.path.join(PROJECT_ROOT, "selected_features", FILE_NAME)
        # O ajusta la subcarpeta si es para resultados federados:
        print(f"Archivo de Características Seleccionadas: {current_selected_features_file_path}")
    
    print(f"Ratio de División para Prueba: {TEST_SPLIT_RATIO if DATASET_NAME not in DATASETS_WITH_PREDEFINED_TEST else 'N/A (Test predefinido)'}")
    print(f"Escalar Características: {'Sí' if SCALE_FEATURES else 'No'}")

    selected_indices = None
    if not USE_ALL_FEATURES:
        selected_indices = load_selected_feature_indices(current_selected_features_file_path)
        if selected_indices is None or len(selected_indices) == 0:
            print(f"No se pudieron cargar los índices de características o no hay características seleccionadas. Abortando.")
            return
        print(f"Se usarán {len(selected_indices)} características seleccionadas.")
    else:
        print("Se usarán todas las características disponibles del dataset.")

    X_train_raw, y_train_raw = None, None
    X_test_raw, y_test_raw = None, None
    X_global_raw, y_global_raw = None, None # Para el caso de train_test_split

    use_predefined_test = DATASET_NAME in DATASETS_WITH_PREDEFINED_TEST
    
    try:
        if use_predefined_test:
            print(f"Usando conjunto de entrenamiento y test predefinidos para '{DATASET_NAME}'.")
            X_train_raw, y_train_raw = load_dataset(DATASET_NAME) # Asume que esto carga el conjunto de entrenamiento/validación
            if X_train_raw is None or y_train_raw is None: raise ValueError(f"load_dataset({DATASET_NAME}) devolvió None para datos de entrenamiento.")
            print(f"Cargado X_train_raw: {X_train_raw.shape}, y_train_raw: {y_train_raw.shape}")

            X_test_raw, y_test_raw = load_predefined_test_set(DATASET_NAME, PROJECT_ROOT)
            if X_test_raw is None or y_test_raw is None: raise ValueError(f"load_predefined_test_set({DATASET_NAME}) no pudo cargar el conjunto de test.")
            print(f"Cargado X_test_raw: {X_test_raw.shape}, y_test_raw: {y_test_raw.shape}")
        else:
            print(f"Usando train_test_split para '{DATASET_NAME}'.")
            X_global_raw, y_global_raw = load_dataset(DATASET_NAME)
            if X_global_raw is None or y_global_raw is None: raise ValueError("load_dataset devolvió None.")
            print(f"Dataset '{DATASET_NAME}' cargado. Forma original de X: {X_global_raw.shape}, Forma de y: {y_global_raw.shape}")
    except Exception as e:
        print(f"Error cargando datos para '{DATASET_NAME}': {e}")
        return

    X_train_final, X_test_final = None, None
    y_train_final, y_test_final = None, None

    # --- Aplicar selección de características o usar todas ---
    if not USE_ALL_FEATURES:
        if selected_indices is None: 
             print("Error: Se indicó usar características seleccionadas pero los índices no están disponibles.")
             return
        try:
            # Validar que los índices no excedan las dimensiones del dataset cargado
            max_dim_check = X_train_raw.shape[1] if use_predefined_test else X_global_raw.shape[1]
            if np.any(selected_indices >= max_dim_check) or np.any(selected_indices < 0):
                print(f"Error: Índices de características ({np.min(selected_indices)}-{np.max(selected_indices)}) fuera de rango (0 a {max_dim_check-1}).")
                return
            
            if use_predefined_test:
                X_train_final = X_train_raw[:, selected_indices]
                X_test_final = X_test_raw[:, selected_indices]
                y_train_final = y_train_raw
                y_test_final = y_test_raw
            else:
                X_selected_global = X_global_raw[:, selected_indices]
                X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
                    X_selected_global, y_global_raw, test_size=TEST_SPLIT_RATIO,
                    random_state=RANDOM_STATE, stratify=y_global_raw if len(np.unique(y_global_raw)) > 1 else None
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
        if use_predefined_test:
            X_train_final = X_train_raw
            X_test_final = X_test_raw
            y_train_final = y_train_raw
            y_test_final = y_test_raw
        else:
            X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
                X_global_raw, y_global_raw, test_size=TEST_SPLIT_RATIO,
                random_state=RANDOM_STATE, stratify=y_global_raw if len(np.unique(y_global_raw)) > 1 else None
            )
        print(f"Usando todas las características. Forma de X_train: {X_train_final.shape}")
        if X_test_final is not None: print(f"Forma de X_test: {X_test_final.shape}")


    if X_train_final is None or X_test_final is None or y_train_final is None or y_test_final is None:
        print("Error: Los conjuntos de datos finales (train/test) no se generaron correctamente.")
        return

    # Escalar características
    if SCALE_FEATURES:
        print("Escalando características (StandardScaler)...")
        scaler = StandardScaler()
        X_train_final = scaler.fit_transform(X_train_final)
        X_test_final = scaler.transform(X_test_final)

    # Entrenar y evaluar el clasificador
    evaluate_classifier(X_train_final, X_test_final, y_train_final, y_test_final, CLASSIFIER_CHOICE, DATASET_NAME)

    print(f"\n--- Evaluación de Clasificador Finalizada ---")

if __name__ == "__main__":
    main()