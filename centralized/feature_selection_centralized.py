import json
import os
import sys
import time
import pandas as pd

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
except NameError:
    PROJECT_ROOT = os.getcwd()
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

try:
    from utils import load_dataset, discretize_equalwidth, load_splits
    from mutual_information import MIM, JMI
    from model_trainer import ModelTrainer
except ImportError as e:
    print(f"Error importando módulos desde la raíz del proyecto ('{PROJECT_ROOT}'): {e}")
    print("Asegúrate de que la estructura de tu proyecto es correcta y que utils.py y mutual_information.py están accesibles.")
    exit(1)

# --- Parámetros de Configuración --- 
# # O cargado de tu config

def load_simulation_config(project_root_path, config_filename="config.json"):
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
            return default_config
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
                print(f"Advertencia: Usando valor por defecto para '{key}': {default_config[key]}")
        return config
    except Exception as e:
        print(f"Error cargando configuración desde '{config_filepath}': {e}. Usando configuración por defecto.")
        return default_config

def save_selected_features_txt(selected_feature_indices, dataset_name_str, top_k_val, technique_name, project_root_path, rep_id=None, fold_id=None):
    main_datasets_folder = "selected_features"
    output_dir = os.path.join(project_root_path, main_datasets_folder)
    os.makedirs(output_dir, exist_ok=True)
    fold_tag = f"_rep{rep_id}_fold{fold_id}" if rep_id is not None and fold_id is not None else ""
    output_filename = f"{dataset_name_str}_centralized_selected_top{top_k_val}_{technique_name}{fold_tag}_feature_indices.txt"
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

    dataset_name = cfg["DATASET_TO_LOAD_GLOBALLY"]
    top_k_features = cfg["TOP_K_FEATURES_TO_SELECT"]
    n_bins_discretization = cfg["NUM_BINS"]
    mi_fs_method = cfg["MI_FS_METHOD"]
    clf_type = cfg["CLASSIFIER_METHOD"]
    mi_function = JMI if mi_fs_method.upper() == "JMI" else MIM

    print(f"--- Iniciando Selección de Características Centralizada con validación cruzada 3x5 para: {dataset_name} ---")
    print(f"Usando Técnica de MI: {mi_function.__name__}")
    print(f"Top K Características a seleccionar: {top_k_features}")
    print(f"Número de bins para discretización: {n_bins_discretization}\n")

    X, y, _ = load_dataset(dataset_name)
    splits_path = os.path.join(SCRIPT_DIR, "..", "datasets", "splits", f"splits_{dataset_name}.json")
    splits = load_splits(splits_path)

    results_dict = {
        "knn": [],
        "rf": []
    }
    
    
    for rep_id, rep_splits in enumerate(splits):
        for fold_id, split in enumerate(rep_splits):
            
                print(f"\n>>> Repetición {rep_id+1} Fold {fold_id+1} <<<")
                train_idx = split["train_idx"]
                test_idx = split["test_idx"]
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]

                # Discretización: min/max SOLO en train
                feature_ranges = [(X_train[:, i].min(), X_train[:, i].max()) for i in range(X_train.shape[1])]
                X_train_disc = discretize_equalwidth(X_train, bins=n_bins_discretization, feature_ranges=feature_ranges)
                X_test_disc  = discretize_equalwidth(X_test,  bins=n_bins_discretization, feature_ranges=feature_ranges)

                # Selección de características
                t0 = time.time()
                features_sel = mi_function(X_train_disc, y_train, topK=top_k_features)
                t1 = time.time()
                for ml_type in clf_type:
                    t2 = time.time()
                    X_train_fs = X_train #[:, features_sel]
                    X_test_fs  = X_test #[:, features_sel]
                    
                    trainer = ModelTrainer(clf_type=ml_type, random_state=42)
                    result = trainer.fit_predict(X_train_fs, y_train, X_test_fs, y_test)
                    acc = result["accuracy"]
                    t3 = time.time()

                    print(f"{ml_type} = Acc: {acc:.3f} | Tiempo selección: {t1-t0:.2f}s | Tiempo total fold: {(t3-t2)+(t1-t0):.2f}s | Features (top 10): {features_sel[:10]}")
                    results_dict[ml_type].append({
                        "rep": rep_id+1,
                        "fold": fold_id+1,
                        "accuracy": acc,
                        "recall": result["recall"],
                        "f1_score": result["f1_score"],
                        "features_selected": features_sel.tolist(),
                        "time_selection": t1-t0,
                        "time_total": (t3-t2)+(t1-t0),
                        "train_time": result["train_time"],
                        "pred_time": result["pred_time"]
                    })
             # Guardar features seleccionadas de este fold (opcional)
                save_selected_features_txt(features_sel, dataset_name, top_k_features, mi_function.__name__, PROJECT_ROOT, rep_id+1, fold_id+1)
            

    # Guardar resultados en CSV
    try:
        os.makedirs("results", exist_ok=True)
        for ml_type in clf_type:
            df = pd.DataFrame(results_dict[ml_type])
            df.to_csv(f"results/results_{dataset_name}_{mi_fs_method}_{ml_type}_centralized_crossval.csv", index=False)
            print(f"\nResultados de validación cruzada guardados en: results/results_{dataset_name}_{mi_fs_method}_{ml_type}_centralized_crossval.csv")
            print(f"{ml_type} = Accuracy medio por repetición:")
            print(df.groupby("rep")["accuracy"].mean())
                
                # --- Media y desviación estándar de todas las precisiones (sobre todos los folds) ---

            print(f"\n{ml_type} = Accuracy media: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
            print(f"\n{ml_type} = Recall macro medio: {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")
            print(f"\n{ml_type} = F1-score macro medio: {df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f}")
            print(f"\n{ml_type} = Tiempo medio entrenamiento: {df['train_time'].mean():.4f} ± {df['train_time'].std():.4f}")
            print(f"\n{ml_type} = Tiempo medio prediccion: {df['pred_time'].mean():.4f} ± {df['pred_time'].std():.4f}")
    except ImportError:
        print("\nPandas no instalado, resultados no guardados en CSV.")

    print(f"\n--- Proceso de Selección de Características Centralizada con validación cruzada 3x5 finalizado para {dataset_name} ---")

if __name__ == "__main__":
    main()
