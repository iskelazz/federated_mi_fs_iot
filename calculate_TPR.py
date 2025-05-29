import os
import argparse
import numpy as np

def load_feature_indices(filepath):
    """Carga los índices de características desde un archivo .txt a un conjunto."""
    try:
        with open(filepath, 'r') as f:
            indices = {int(line.strip()) for line in f if line.strip()}
        return indices
    except Exception as e:
        print(f"Error leyendo el archivo de índices '{filepath}': {e}")
        return set()

def calculate_tpr(ideal_features_set, evaluated_features_set, k_value: int = 0):
    """
    Calcula la Tasa de Verdaderos Positivos (TPR) entre dos conjuntos de características.
    TPR = TP / k
    donde TP es el número de características comunes, y k es el número de características seleccionadas
    en el conjunto centralizado (o el k especificado).
    """

    k = k_value if k_value > 0 else len(ideal_features_set)

    # La intersección de conjuntos da los elementos comunes.
    true_positives_set = ideal_features_set.intersection(evaluated_features_set)
    tp_count = len(true_positives_set)
    
    tpr = tp_count / k
    
    print(f"\n--- Cálculo de TPR ---")
    print(f"Valor de k usado para el denominador: {k}")
    print(f"TPR = TP / k = {tp_count} / {k} = {tpr:.4f}")
    print(f"----------------------")
    
    return tpr

def main():
    parser = argparse.ArgumentParser(description="Calcula la Tasa de Verdaderos Positivos (TPR) comparando dos listas de características seleccionadas.")
    parser.add_argument("ideal_file_path", type=str)
    parser.add_argument("evaluated_file_path", type=str)
    parser.add_argument("-k", "--k_value", type=int, default=0)

    args = parser.parse_args()

    ideal_features = load_feature_indices(args.ideal_file_path)
    evaluated_features = load_feature_indices(args.evaluated_file_path)


    # El k_value de los argumentos es el k para el denominador.
    k_for_tpr = args.k_value
    if k_for_tpr == 0:
        k_for_tpr = len(ideal_features)
        print(f"Usando k={k_for_tpr} (basado en el tamaño del conjunto centralizado) para el cálculo de TPR.")
    elif k_for_tpr != len(ideal_features):
         print(f"Advertencia: Se especificó k={k_for_tpr}, pero el archivo centralizado tiene {len(ideal_features)} características. Se usará k={k_for_tpr} como denominador.")


    if k_for_tpr == 0 and not ideal_features: 
        print("Error: No se puede determinar un valor k válido para el denominador (k es 0).")
        return

    _ = calculate_tpr(ideal_features, evaluated_features, k_value=k_for_tpr)

if __name__ == "__main__":
    main()
    
    
#Ejemplo de uso: python3 .\calculate_TPR.py selected_features/gisette_federated_selected_top15_JMI_feature_indices.txt selected_features/gisette_federated_selected_top15_JMI_feature_indices.txt