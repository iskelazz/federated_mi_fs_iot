import json
import time
import sys
import os
import numpy as np
import pandas as pd
import gc

SCRIPT_DIR_ORCH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_ORCH = os.path.dirname(SCRIPT_DIR_ORCH)
if PROJECT_ROOT_ORCH not in sys.path: sys.path.insert(0, PROJECT_ROOT_ORCH)

try:
    from mqtt_handlers.mqtt_communicator import MQTTCommunicator
    from server_logic import ServerLogic 
    from server_logic import SERVER_ID_PREFIX_PC
    from utils import load_dataset, build_iid_data, build_noniid_data, build_noniid_uneven_no_loss, plot_label_dispersion_matplotlib_only, build_by_subject, load_splits
    from model_trainer import ModelTrainer
except ImportError as e:
    print(f"ERROR crítico importando módulos: {e}. Verifique PYTHONPATH.")
    sys.exit(1)
    
def time_resume(benchmarks):
    df = pd.DataFrame(benchmarks)    
    print("\n================== RESUMEN FINAL ==================")
    def mostrar_media_std(nombre, valores):
        v = np.array(valores)
        print(f"{nombre:18s}: {np.mean(v):8.3f} ± {np.std(v):7.3f}")

    # Tiempos federados (en segundos)
    mostrar_media_std("Carga datos",          df["t_load_max"])
    mostrar_media_std("Preproc (disc)",       df["t_pre_max"])
    mostrar_media_std("Cómputo cliente",      df["t_compute_max"])
    mostrar_media_std("Comunicación",         df["t_comm_sum"])
    mostrar_media_std("Otros (servidor)",     df["t_others"])
    mostrar_media_std("TOTAL federado",       df["total_elapsed_time"])

    print("===================================================")


def load_simulation_config(project_root_path, config_filename="config.json"):
    config_filepath = os.path.join(project_root_path, config_filename)
    default_config = {
        "DATASET_TO_LOAD_GLOBALLY": "mnist",
        "MI_FS_METHOD": "MIM", 
        "NUM_SIMULATED_CLIENTS_TOTAL": 2,
        "DISTRIBUTION_TYPE": "iid",
        "NUM_BINS": 5,
        "TOP_K_FEATURES_TO_SELECT": 15,
        "TIMEOUT_SECONDS_OVERALL": 20,
        "BROKER_ADDRESS_FOR_SERVER": "localhost",
        "PORT": 1883,
        "AGGREGATION_METHOD": "simple",
        "UNEVENNESS_FACTOR_NONIID": 0.0,
        "PLOT_DISPERSION": "false",
        "OPPORTUNITY_CROSS_SILO": "false",
        "CLASSIFIER_TYPE": "rf"
    }
    try:
        with open(config_filepath, 'r') as f:
            all_config = json.load(f)
        print(f"Configuración cargada desde '{config_filepath}'.")
        config = all_config.get("FS_FEDERATED")
        if config is None:
            print(f"Advertencia: La clave 'FS_FEDERATED' no se encontró en '{config_filepath}'. "
                  f"Usando la configuración por defecto completa para 'FS_FEDERATED'.")
            return default_config
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
                print(f"Advertencia: Usando valor por defecto para '{key}': {default_config[key]}")
        return config
    except Exception as e:
        print(f"Error cargando configuración desde '{config_filepath}': {e}. Usando configuración por defecto.")
        return default_config

def build_client_indices_map(DISTRIBUTION_TYPE, labels_train, train_idx, subj_train, NUM_SIMULATED_CLIENTS_TOTAL, UNEVENNESS_FACTOR_NONIID, DATASET_TO_LOAD_GLOBALLY, OPPORTUNITY_CROSS_SILO):
    if NUM_SIMULATED_CLIENTS_TOTAL == 4 and DATASET_TO_LOAD_GLOBALLY == "opportunity" and OPPORTUNITY_CROSS_SILO:
        return build_by_subject(subj_train)
    elif DISTRIBUTION_TYPE == "iid":
        return build_iid_data(train_idx, labels_train, NUM_SIMULATED_CLIENTS_TOTAL)
    elif DISTRIBUTION_TYPE == "non-iid":
        if (UNEVENNESS_FACTOR_NONIID > 0):
            return build_noniid_uneven_no_loss(train_idx, labels_train, NUM_SIMULATED_CLIENTS_TOTAL, UNEVENNESS_FACTOR_NONIID)
        else:
            return build_noniid_data(train_idx, labels_train, NUM_SIMULATED_CLIENTS_TOTAL)
    else:
        print(f"Tipo de distribución '{DISTRIBUTION_TYPE}' no reconocido. Usando IID.")
        return build_iid_data(train_idx, labels_train, NUM_SIMULATED_CLIENTS_TOTAL)

def wait_for_federated_selection(server_handler, MI_FS_METHOD, TIMEOUT_SECONDS_OVERALL):
    """
    Espera a que acabe el hilo de selección federada (JMI/MIM) como en tu lógica actual.
    """
    start_wait_time = time.time()
    initial_phase_monitor_timeout = 180.0 
    while time.time() - start_wait_time < initial_phase_monitor_timeout:
        if server_handler.expected_clients_in_round == 0:
            break 
        time.sleep(0.1)

    elapsed_since_global_start = time.time() - server_handler.initial_time
    remaining_timeout_for_fs = TIMEOUT_SECONDS_OVERALL - elapsed_since_global_start

    # Gestión hilos JMI/MIM
    if MI_FS_METHOD == "JMI":
        jmi_thread_to_join = None
        jmi_thread_creation_wait_start = time.time()
        jmi_thread_creation_timeout = 10.0 
        while server_handler.jmi_orchestrator_thread is None and \
              (time.time() - jmi_thread_creation_wait_start < jmi_thread_creation_timeout):
            time.sleep(0.1) 
        jmi_thread_to_join = server_handler.jmi_orchestrator_thread
        if jmi_thread_to_join is not None:
            if jmi_thread_to_join.is_alive():
                print(f"Proceso JMI en ejecución. Esperando su finalización (timeout restante: {max(0, remaining_timeout_for_fs):.2f}s)...")
                jmi_thread_to_join.join(timeout=max(0, remaining_timeout_for_fs))
                if jmi_thread_to_join.is_alive():
                    print("ADVERTENCIA: Timeout general esperando al hilo JMI.")
                else:
                    print("Hilo JMI finalizado.")

    else: # MIM y otros
        aggregation_thread_to_join = server_handler.aggregation_thread
        if aggregation_thread_to_join is not None:
            if aggregation_thread_to_join.is_alive():
                print(f"Hilo de {MI_FS_METHOD} detectado. Haciendo join (timeout restante: {max(0, remaining_timeout_for_fs):.2f}s)...")
                aggregation_thread_to_join.join(timeout=max(0, remaining_timeout_for_fs))
                if aggregation_thread_to_join.is_alive():
                    print(f"ADVERTENCIA: Timeout esperando al hilo de {MI_FS_METHOD} (join).")

def get_selected_features_from_server(server_handler, MI_FS_METHOD, DATASET_TO_LOAD_GLOBALLY, TOP_K_FEATURES_TO_SELECT, rep_id, fold_id):
    """
    Recupera los índices seleccionados por la federada (buscando en el .txt generado).
    """
    main_datasets_folder = "selected_features"
    out_name = f"{DATASET_TO_LOAD_GLOBALLY}_federated_selected_top{TOP_K_FEATURES_TO_SELECT}_{MI_FS_METHOD}_federated_rep{rep_id+1}_fold{fold_id+1}_feature_indices.txt"
    out_path = os.path.join(server_handler.project_root_app, main_datasets_folder, out_name)
    if not os.path.exists(out_path):
        raise RuntimeError(f"No se encuentra el archivo de features seleccionadas: {out_path}")
    with open(out_path, 'r') as f:
        features = [int(line.strip()) for line in f.readlines()]
    return features

def generate_and_display_label_dispersion(
    config, 
    unique_global_labels, 
    num_total_clients, 
    client_data_indices_map, 
    global_labels_array, 
    dataset_name_global, 
    distribution_type_global
):
    """
    Prepara los datos y genera un gráfico de dispersión de etiquetas si está habilitado en la config.
    """
    if not config.get("PLOT_DISPERSION", False):
        return # No hacer nada si no está habilitado

    print("Generando gráfico de dispersión de etiquetas...")
    
    all_possible_labels_sorted_str = sorted([str(l) for l in unique_global_labels])
    client_names_for_plot_order = [f"Cliente {idx}" for idx in range(num_total_clients)]
    
    device_label_counts_for_plot = {}
    for client_idx, client_name in enumerate(client_names_for_plot_order):
        local_indices = client_data_indices_map.get(client_idx, [])
        
        current_client_label_counts = {}
        if local_indices:
            y_local_client = global_labels_array[local_indices]
            unique_labels, counts = np.unique(y_local_client, return_counts=True)
            current_client_label_counts = {str(label): count for label, count in zip(unique_labels, counts)}
        
        device_label_counts_for_plot[client_name] = {
            global_label_str: current_client_label_counts.get(global_label_str, 0)
            for global_label_str in all_possible_labels_sorted_str
        }

    plot_title = f"Distribución de Etiquetas para Dataset '{dataset_name_global}' ({distribution_type_global.upper()})"
    
    try:
       
        plot_label_dispersion_matplotlib_only(
            device_label_counts_for_plot,
            client_names_for_plot_order,
            all_possible_labels_sorted_str,
            title=plot_title
        )
    except ImportError:
        print("ADVERTENCIA: La función 'plot_label_dispersion_matplotlib_only' no se pudo importar. No se generará el gráfico.")
    except Exception as e_plot:
        print(f"ERROR al generar el gráfico de dispersión: {e_plot}")

def main():
    print("Iniciando servidor y orquestación (cross-validation federada con hilos y sincronía)...")
    config = load_simulation_config(PROJECT_ROOT_ORCH)

    DATASET_TO_LOAD_GLOBALLY = config["DATASET_TO_LOAD_GLOBALLY"]
    MI_FS_METHOD = config["MI_FS_METHOD"]
    NUM_SIMULATED_CLIENTS_TOTAL = config["NUM_SIMULATED_CLIENTS_TOTAL"]
    DISTRIBUTION_TYPE = config["DISTRIBUTION_TYPE"]
    NUM_BINS = config["NUM_BINS"]
    TOP_K_FEATURES_TO_SELECT = config["TOP_K_FEATURES_TO_SELECT"]
    TIMEOUT_SECONDS_OVERALL = config["TIMEOUT_SECONDS_OVERALL"] 
    BROKER_ADDRESS = config["BROKER_ADDRESS_FOR_SERVER"]
    PORT = config["PORT"]
    AGGREGATION_METHOD = config["AGGREGATION_METHOD"]
    UNEVENNESS_FACTOR_NONIID = config["UNEVENNESS_FACTOR_NONIID"]
    OPPORTUNITY_CROSS_SILO = config["OPPORTUNITY_CROSS_SILO"]
    PLOT_DISPERSION = config["PLOT_DISPERSION"]
    CLASSIFIER_TYPE = config.get("CLASSIFIER_TYPE", "rf")

    SPLITS_FILE = os.path.join(PROJECT_ROOT_ORCH, "datasets", "splits", f"splits_{DATASET_TO_LOAD_GLOBALLY}.json")
    
    if isinstance(CLASSIFIER_TYPE, str):
        clf_types = [CLASSIFIER_TYPE]
    else:
        clf_types = list(CLASSIFIER_TYPE)

    try:
        X_global, y_global, subj_global = load_dataset(DATASET_TO_LOAD_GLOBALLY)
        splits = load_splits(SPLITS_FILE)
    except Exception as e_load:
        print(f"ERROR cargando dataset o splits: {e_load}"); sys.exit(1)

    results = []
    benchmarks = []

    for rep_id, rep_splits in enumerate(splits):
        for fold_id, split in enumerate(rep_splits):
            print(f"\n====== Rep {rep_id+1} Fold {fold_id+1} ======")
            train_idx = np.array(split["train_idx"])
            test_idx  = np.array(split["test_idx"])
            X_train = X_global[train_idx]
            labels_train = y_global[train_idx]
            subj_train = subj_global[train_idx] if subj_global is not None else None

            user_indices_map = build_client_indices_map(
                DISTRIBUTION_TYPE, labels_train, train_idx, subj_train, NUM_SIMULATED_CLIENTS_TOTAL,
                UNEVENNESS_FACTOR_NONIID, DATASET_TO_LOAD_GLOBALLY, OPPORTUNITY_CROSS_SILO
            )
            if rep_id == 0 and fold_id == 0 and PLOT_DISPERSION:
                unique_global_labels = np.unique(y_global)
                generate_and_display_label_dispersion(
                    config,
                    unique_global_labels,
                    NUM_SIMULATED_CLIENTS_TOTAL,
                    user_indices_map,
                    y_global,
                    DATASET_TO_LOAD_GLOBALLY,
                    DISTRIBUTION_TYPE
                )
            # --- Inicializa server_handler y communicator
            server_handler = ServerLogic(PROJECT_ROOT_ORCH)
            communicator = MQTTCommunicator(
                BROKER_ADDRESS, PORT, 
                client_id_prefix=SERVER_ID_PREFIX_PC
            )
            server_handler.set_communicator(communicator)
            communicator.set_message_callback(server_handler.on_server_message_received)
            communicator.set_connect_callback(server_handler.on_connected_to_broker)
            communicator.set_disconnect_callback(server_handler.on_disconnected_from_broker)
            if not communicator.connect():
                print("Fallo en conexión MQTT. Abortando fold.")
                continue
            communicator.start_listening()

            # --- Parámetros de la ronda y reparto de datos
            server_handler.set_round_parameters(MI_FS_METHOD, TOP_K_FEATURES_TO_SELECT, AGGREGATION_METHOD, NUM_BINS, rep_id, fold_id)
            server_handler.initialize_new_round(NUM_SIMULATED_CLIENTS_TOTAL) 

            # --- Comandos iniciales a clientes
            active_client_ids_for_round = []
            actual_clients_commanded = 0
            for client_idx in range(NUM_SIMULATED_CLIENTS_TOTAL):
                if client_idx not in user_indices_map or not user_indices_map[client_idx]:
                    continue
                sim_client_id = f"sim_client_{client_idx}"
                client_indices = [int(i) for i in user_indices_map[client_idx]]
                if server_handler.send_processing_command_to_pi(sim_client_id, DATASET_TO_LOAD_GLOBALLY, client_indices, len(np.unique(labels_train))):
                    active_client_ids_for_round.append(sim_client_id)
                    actual_clients_commanded += 1
                else:
                    print(f"Fallo al enviar comando a {sim_client_id}.")
            if actual_clients_commanded < server_handler.expected_clients_in_round:
                print(f"Ajustando clientes esperados de {server_handler.expected_clients_in_round} a {actual_clients_commanded}.")
                server_handler.expected_clients_in_round = actual_clients_commanded

            # --- Espera sincronizada a la selección federada (esta es la clave)
            print(f"Comandos iniciales enviados. Esperando fase inicial de {server_handler.expected_clients_in_round} clientes...")

            start_wait_time = time.time()
            initial_phase_monitor_timeout = 180.0

            while time.time() - start_wait_time < initial_phase_monitor_timeout:
                if server_handler.expected_clients_in_round == 0:
                    break

                should_break_loop = False
                with server_handler.jmi_lock:
                    initial_phase_done_count = server_handler.clients_reported_initial_XY_count
                    jmi_thread_is_running = (
                        server_handler.jmi_orchestrator_thread is not None
                        and server_handler.jmi_orchestrator_thread.is_alive()
                    )
                    initial_phase_done_by_jmi_start = (MI_FS_METHOD == "JMI" and jmi_thread_is_running)
                    error_count = sum(
                        1
                        for cid in active_client_ids_for_round
                        if cid in server_handler.active_sim_clients
                        and server_handler.active_sim_clients[cid].error_message
                        and not server_handler.active_sim_clients[cid].local_XY_prob_dist_received
                    )

                    cond1 = (
                        initial_phase_done_count >= server_handler.expected_clients_in_round
                        and server_handler.expected_clients_in_round > 0
                    )
                    cond2 = initial_phase_done_by_jmi_start
                    cond3 = (
                        error_count >= server_handler.expected_clients_in_round
                        and server_handler.expected_clients_in_round > 0
                    )

                    if cond1 or cond2 or cond3:
                        should_break_loop = True

                if should_break_loop:
                    print("Fase inicial (P(Xi,Y)) completada, JMI iniciado o todos los clientes esperados fallaron antes.")
                    break

                time.sleep(0.1)
            else:
                print(f"Timeout esperando finalización de la fase inicial después de {initial_phase_monitor_timeout}s.")

            elapsed_since_global_start = time.time() - server_handler.initial_time
            remaining_timeout_for_fs = TIMEOUT_SECONDS_OVERALL - elapsed_since_global_start

            # --- Gestión de hilos federados
            if MI_FS_METHOD == "JMI":
                jmi_thread_to_join = None
                jmi_thread_creation_wait_start = time.time()
                jmi_thread_creation_timeout = 10.0

                while (
                    server_handler.jmi_orchestrator_thread is None
                    and (time.time() - jmi_thread_creation_wait_start < jmi_thread_creation_timeout)
                ):
                    time.sleep(0.1)

                jmi_thread_to_join = server_handler.jmi_orchestrator_thread

                if jmi_thread_to_join is not None:
                    if jmi_thread_to_join.is_alive():
                        print(f"Proceso JMI en ejecución. Esperando su finalización (timeout restante: {max(0, remaining_timeout_for_fs):.2f}s)...")
                        jmi_thread_to_join.join(timeout=max(0, remaining_timeout_for_fs))
                        if jmi_thread_to_join.is_alive():
                            print("ADVERTENCIA: Timeout general esperando al hilo JMI.")
                        else:
                            print("Hilo JMI finalizado.")

            else:  # MIM y otros
                print(f"Método es {MI_FS_METHOD}. Esperando finalización del hilo de trabajo correspondiente...")
                aggregation_thread_to_join = server_handler.aggregation_thread

                if aggregation_thread_to_join is not None:
                    if aggregation_thread_to_join.is_alive():
                        print(f"Hilo de {MI_FS_METHOD} detectado. Haciendo join (timeout restante: {max(0, remaining_timeout_for_fs):.2f}s)...")
                        aggregation_thread_to_join.join(timeout=max(0, remaining_timeout_for_fs))
                        if aggregation_thread_to_join.is_alive():
                            print(f"ADVERTENCIA: Timeout esperando al hilo de {MI_FS_METHOD} (join).")

            # --- Paramos timer de selección de caracteristicas antes de entrenamiento
            global_end_time = time.time()
            # --- Recupera features seleccionados (del txt generado)
            try:
                features_selected = get_selected_features_from_server(
                    server_handler, MI_FS_METHOD, DATASET_TO_LOAD_GLOBALLY, TOP_K_FEATURES_TO_SELECT, rep_id, fold_id
                )
            except Exception as e:
                print(f"ERROR recuperando features seleccionadas: {e}")
                communicator.disconnect()
                continue

            X_train_fs = X_train[:, features_selected]
            X_test, y_test = X_global[test_idx], y_global[test_idx]
            X_test_fs = X_test[:, features_selected]

            # --- Ejecuta todos los clasificadores y guarda los índices de los resultados de este fold
            fold_results_idx = []
            for clf_type in clf_types:
                trainer = ModelTrainer(clf_type=clf_type, random_state=42)
                result = trainer.fit_predict(X_train_fs, labels_train, X_test_fs, y_test)
                acc = result["accuracy"]
                print(f"Fold {fold_id+1} | {clf_type} | Accuracy: {acc:.4f} | F1: {result['f1_score']:.4f} | Recall: {result['recall']:.4f}")
                results.append({
                    "rep": rep_id+1,
                    "fold": fold_id+1,
                    "clf_type": clf_type,
                    "accuracy": acc,
                    "f1_score": result["f1_score"],
                    "recall": result["recall"],
                    "train_time": result["train_time"],
                    "pred_time": result["pred_time"],
                    "features_selected": list(features_selected),
                    "total_time_fs": global_end_time - server_handler.initial_time
                    # (Las métricas de emisiones se añadirán después)
                })
                fold_results_idx.append(len(results) - 1)

            # --- Métricas de tiempo de cada fold (solo una vez por fold)
            t_load_max, t_pre_max, t_compute_max, t_comm_sum = server_handler.get_bench_summary()
            total_elapsed_time = global_end_time - server_handler.initial_time
            summary = None
            if MI_FS_METHOD == "JMI":
                summary = server_handler.current_jmi_orchestrator.get_server_timing_summary()
            t_others = max(0.0, total_elapsed_time - t_compute_max - t_comm_sum - t_pre_max)
            per_fold_bench = {
                "rep": rep_id+1,
                "fold": fold_id+1,
                "t_load_max": t_load_max,
                "t_pre_max": t_pre_max,
                "t_compute_max": t_compute_max,
                "t_comm_sum": t_comm_sum,
                "t_agg_total": summary["T_agg_total"] if summary is not None and "T_agg_total" in summary else None,
                "t_mi_calc_total": summary["T_mi_calc_total"] if summary is not None and "T_mi_calc_total" in summary else None,
                "t_others": t_others,
                "total_elapsed_time": total_elapsed_time,
            }
            print(per_fold_bench)
            benchmarks.append(per_fold_bench)

            # --- Solicita emisiones y añade los datos a todos los clasificadores de este fold
            server_handler.send_emission_request_to_clients()
            max_wait = 10
            waited = 0
            while (server_handler.emissions_manager.last_emissions_summary is None) and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5

            if server_handler.emissions_manager.last_emissions_summary:
                for idx in fold_results_idx:
                    results[idx].update(server_handler.emissions_manager.last_emissions_summary)
            else:
                print(f"ADVERTENCIA: No se recibieron datos de emisiones en el tiempo de espera ({max_wait}s) para este fold.")

            if communicator:
                communicator.disconnect()
            time.sleep(2)
    time_resume(benchmarks)
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    out_csv = f"results/results_{DATASET_TO_LOAD_GLOBALLY}_{MI_FS_METHOD}_federated_crossval.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nResultados de validación cruzada guardados en: {out_csv}")

    # --- Muestra los resultados bien organizados por clasificador ---
    for clf_type in df["clf_type"].unique():
        print(f"\nResultados para {clf_type}:")
        print("Accuracy medio por repetición:")
        print(df[df["clf_type"] == clf_type].groupby("rep")["accuracy"].mean())
        print(f"Accuracy media total: {df[df['clf_type'] == clf_type]['accuracy'].mean():.4f} ± {df[df['clf_type'] == clf_type]['accuracy'].std():.4f}")
        print(f"F1-score media total: {df[df['clf_type'] == clf_type]['f1_score'].mean():.4f} ± {df[df['clf_type'] == clf_type]['f1_score'].std():.4f}")
        print(f"Recall media total: {df[df['clf_type'] == clf_type]['recall'].mean():.4f} ± {df[df['clf_type'] == clf_type]['recall'].std():.4f}")
        print(f"Tiempo medio entrenamiento: {df[df['clf_type'] == clf_type]['train_time'].mean():.4f} s")
        print(f"Tiempo medio predicción: {df[df['clf_type'] == clf_type]['pred_time'].mean():.4f} s")

        if "grand_total_energy" in df.columns:
            print(f"\nConsumo energético (kWh):")
            print(f"Cliente total: {df[df['clf_type'] == clf_type]['total_client_energy_kwh'].mean():.6f} ± {df[df['clf_type'] == clf_type]['total_client_energy_kwh'].std():.6f}")
            print(f"Servidor total: {df[df['clf_type'] == clf_type]['server_energy_kwh'].mean():.6f} ± {df[df['clf_type'] == clf_type]['server_energy_kwh'].std():.6f}")
            print(f"Media total: {df[df['clf_type'] == clf_type]['grand_total_energy'].mean():.6f} ± {df[df['clf_type'] == clf_type]['grand_total_energy'].std():.6f}")
        if "grand_total_emissions" in df.columns:
            print(f"\nEmisiones totales CO2 (kg):")
            print(f"Cliente total: {df[df['clf_type'] == clf_type]['total_client_co2_kg'].mean():.6f} ± {df[df['clf_type'] == clf_type]['total_client_co2_kg'].std():.6f}")
            print(f"Servidor total: {df[df['clf_type'] == clf_type]['server_co2_kg'].mean():.6f} ± {df[df['clf_type'] == clf_type]['server_co2_kg'].std():.6f}")
            print(f"Media total: {df[df['clf_type'] == clf_type]['grand_total_emissions'].mean():.6f} ± {df[df['clf_type'] == clf_type]['grand_total_emissions'].std():.6f}")

    print("\n--- Validación cruzada federada finalizada ---")
    sys.exit(0)

if __name__ == "__main__":
    main()
