import os
import numpy as np
import argparse
# Use example: python3 compare_tpr_cv.py --fed_dir .\selected_features\ --cent_dir .\selected_features\centralized\madelon\ --dataset madelon --method JMI --k 75
def load_feature_indices(filepath):
    try:
        with open(filepath, 'r') as f:
            indices = {int(line.strip()) for line in f if line.strip()}
        return indices
    except Exception as e:
        print(f"Error leyendo el archivo de índices '{filepath}': {e}")
        return set()

def calculate_tpr(ideal_features_set, evaluated_features_set, k_value = 0):
    k = k_value if k_value > 0 else len(ideal_features_set)
    tp_count = len(ideal_features_set & evaluated_features_set)
    tpr = tp_count / k if k > 0 else 0
    return tpr, tp_count, k

def main():
    parser = argparse.ArgumentParser(description="Compara ficheros de características federados y centralizados para calcular el TPR medio (crossval 3x5)")
    parser.add_argument("--fed_dir", type=str, required=True, help="Directorio de features federados")
    parser.add_argument("--cent_dir", type=str, required=True, help="Directorio de features centralizados")
    parser.add_argument("--dataset", type=str, required=True, help="Nombre del dataset (ej: madelon)")
    parser.add_argument("--method", type=str, required=True, help="Método FS (ej: JMI)")
    parser.add_argument("--k", type=int, default=0, help="Valor K del TPR (si no se pasa, se usa el tamaño del set centralizado)")

    args = parser.parse_args()

    num_reps, num_folds = 3, 5
    k_value = args.k

    tprs = []
    print("\nComparando archivos rep x fold:")
    for rep in range(1, num_reps+1):
        for fold in range(1, num_folds+1):
            fed_file = os.path.join(
                args.fed_dir,
                f"{args.dataset}_federated_selected_top{k_value if k_value else '*'}_{args.method}_federated_rep{rep}_fold{fold}_feature_indices.txt"
            )
            cent_file = os.path.join(
                args.cent_dir,
                f"{args.dataset}_centralized_selected_top{k_value if k_value else '*'}_{args.method}_rep{rep}_fold{fold}_feature_indices.txt"
            )

            # Buscar archivo real si el k_value es 0 y hay varios k en la carpeta
            if not os.path.exists(fed_file) or not os.path.exists(cent_file):
                fed_candidates = [f for f in os.listdir(args.fed_dir) if f"_rep{rep}_fold{fold}_" in f]
                cent_candidates = [f for f in os.listdir(args.cent_dir) if f"_rep{rep}_fold{fold}_" in f]
                fed_file = os.path.join(args.fed_dir, fed_candidates[0]) if fed_candidates else None
                cent_file = os.path.join(args.cent_dir, cent_candidates[0]) if cent_candidates else None

            if not fed_file or not cent_file or not os.path.exists(fed_file) or not os.path.exists(cent_file):
                print(f"Faltan archivos para rep {rep}, fold {fold}. Skipping.")
                continue

            ideal_features = load_feature_indices(cent_file)
            evaluated_features = load_feature_indices(fed_file)
            k_for_tpr = k_value if k_value > 0 else len(ideal_features)
            tpr, tp_count, k_denom = calculate_tpr(ideal_features, evaluated_features, k_for_tpr)
            tprs.append(tpr)
            print(f"rep{rep}-fold{fold} | TPR = {tp_count}/{k_denom} = {tpr:.4f} | {os.path.basename(cent_file)} vs {os.path.basename(fed_file)}")

    if tprs:
        print(f"\nTPR medio: {np.mean(tprs):.4f} ± {np.std(tprs):.4f} (n={len(tprs)})")
    else:
        print("\nNo se han calculado TPRs, comprueba las rutas y archivos.")

if __name__ == "__main__":
    main()
