import json
import time
import sys
import os
import numpy as np

# --- Configuración de Rutas para Importación ---
SCRIPT_DIR_ORCH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_ORCH = os.path.dirname(SCRIPT_DIR_ORCH)
if PROJECT_ROOT_ORCH not in sys.path: sys.path.insert(0, PROJECT_ROOT_ORCH)

try:
    from mqtt_handlers.mqtt_communicator import MQTTCommunicator
    from server_logic import ServerLogic 
    from server_logic import SERVER_ID_PREFIX_PC
    from utils import load_dataset, build_iid_data, build_noniid_data, build_noniid_uneven_no_loss
except ImportError as e:
    print(f"ERROR crítico importando módulos: {e}. Verifique PYTHONPATH.")
    sys.exit(1)

def load_simulation_config(project_root_path: str, config_filename="config.json"):
    """Carga la configuración de simulación desde un archivo JSON."""
    config_filepath = os.path.join(project_root_path, config_filename)
    default_config = {
        "DATASET_TO_LOAD_GLOBALLY": "mnist",
        "MI_FS_METHOD": "MIM", 
        "NUM_SIMULATED_CLIENTS_TOTAL": 2,
        "DISTRIBUTION_TYPE": "iid",
        "NUM_BINS": 5,
        "TOP_K_FEATURES_TO_SELECT": 15,
        "TIMEOUT_SECONDS_OVERALL": 300,
        "BROKER_ADDRESS_FOR_SERVER": "localhost",
        "PORT": 1883,
        "AGGREGATION_METHOD": "simple",
        "UNEVENNESS_FACTOR_NONIID": 0.0
    }
    try:
        with open(config_filepath, 'r') as f:
            config = json.load(f)
        print(f"Configuración cargada desde '{config_filepath}'.")
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
                print(f"Advertencia: Usando valor por defecto para '{key}': {default_config[key]}")
        return config
    except Exception as e:
        print(f"Error cargando configuración desde '{config_filepath}': {e}. Usando configuración por defecto.")
        return default_config

def main():
    """Punto de entrada principal para el orquestador del servidor."""
    print("Iniciando servidor y orquestación...")
    
    config = load_simulation_config(PROJECT_ROOT_ORCH) # Carga desde config.json por defecto

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
    
    server_handler = ServerLogic(PROJECT_ROOT_ORCH)
    # --- Inicialización del Comunicador MQTT ---
    communicator = MQTTCommunicator(
        BROKER_ADDRESS, PORT, 
        client_id_prefix=SERVER_ID_PREFIX_PC
    )
    server_handler.set_communicator(communicator)

    communicator.set_message_callback(server_handler.on_server_message_received)
    communicator.set_connect_callback(server_handler.on_connected_to_broker)
    communicator.set_disconnect_callback(server_handler.on_disconnected_from_broker)

    if not communicator.connect():
        print("Fallo en conexión MQTT. Abortando.")
        sys.exit(1)
    communicator.start_listening()

    # --- Configurar Parámetros de la Ronda en server_logic ---
    server_handler.set_round_parameters(MI_FS_METHOD, TOP_K_FEATURES_TO_SELECT, AGGREGATION_METHOD, NUM_BINS)
    server_handler.initialize_new_round(NUM_SIMULATED_CLIENTS_TOTAL) 

    # --- Carga de Dataset y Distribución ---
    print(f"Cargando dataset global '{DATASET_TO_LOAD_GLOBALLY}'...")
    try:
        X_global, y_global = load_dataset(DATASET_TO_LOAD_GLOBALLY)
        print(f"Dataset '{DATASET_TO_LOAD_GLOBALLY}' cargado. X:{X_global.shape}, Y:{y_global.shape}")
    except Exception as e_load:
        print(f"ERROR cargando dataset: {e_load}"); communicator.disconnect(); sys.exit(1)

    #Determinar número global de clases
    unique_global_labels = np.unique(y_global)
    NUM_GLOBAL_CLASSES = len(unique_global_labels)
    user_indices_map = None

    if DISTRIBUTION_TYPE == "iid": user_indices_map = build_iid_data(y_global, NUM_SIMULATED_CLIENTS_TOTAL)
    elif DISTRIBUTION_TYPE == "non-iid": 
        if (UNEVENNESS_FACTOR_NONIID>0):
            user_indices_map = build_noniid_uneven_no_loss(y_global, NUM_SIMULATED_CLIENTS_TOTAL, UNEVENNESS_FACTOR_NONIID)
        else:
            user_indices_map = build_noniid_data(X_global, y_global, NUM_SIMULATED_CLIENTS_TOTAL)
    else: 
        print(f"Tipo de distribución '{DISTRIBUTION_TYPE}' no reconocido. Usando IID.")
        user_indices_map = build_iid_data(y_global, NUM_SIMULATED_CLIENTS_TOTAL) # Fallback

    global_start_time = time.time()
    # --- Envío de Comandos Iniciales ---
    active_client_ids_for_round = []
    actual_clients_commanded = 0
    for client_idx in range(NUM_SIMULATED_CLIENTS_TOTAL):
        if client_idx not in user_indices_map or not user_indices_map[client_idx]: continue
        sim_client_id = f"sim_client_{client_idx}"
        client_indices = [int(i) for i in user_indices_map[client_idx]]
        if server_handler.send_processing_command_to_pi(sim_client_id, DATASET_TO_LOAD_GLOBALLY, client_indices, NUM_GLOBAL_CLASSES):
            active_client_ids_for_round.append(sim_client_id)
            actual_clients_commanded += 1
        else: print(f"Fallo al enviar comando a {sim_client_id}.")
    
    if actual_clients_commanded < server_handler.expected_clients_in_round:
        print(f"Ajustando clientes esperados de {server_handler.expected_clients_in_round} a {actual_clients_commanded}.")
        server_handler.expected_clients_in_round = actual_clients_commanded

    
    print(f"Comandos iniciales enviados. Esperando fase inicial de {server_handler.expected_clients_in_round} clientes...")

    # --- Bucle de Espera Principal ---
    start_wait_time = time.time()
    initial_phase_monitor_timeout = 180.0 

    while time.time() - start_wait_time < initial_phase_monitor_timeout:
        if server_handler.expected_clients_in_round == 0: break 
        
        should_break_loop = False
        with server_handler.jmi_lock: 
            initial_phase_done_count = server_handler.clients_reported_initial_XY_count
            jmi_thread_is_running = server_handler.jmi_orchestrator_thread is not None and server_handler.jmi_orchestrator_thread.is_alive()
            initial_phase_done_by_jmi_start = (MI_FS_METHOD == "JMI" and jmi_thread_is_running)
            
            error_count = sum(1 for cid in active_client_ids_for_round 
                              if cid in server_handler.active_sim_clients and server_handler.active_sim_clients[cid].error_message
                                 and not server_handler.active_sim_clients[cid].local_XY_prob_dist_received)

            # Evaluar condiciones de salida
            cond1 = (initial_phase_done_count >= server_handler.expected_clients_in_round and server_handler.expected_clients_in_round > 0)
            cond2 = initial_phase_done_by_jmi_start
            cond3 = (error_count >= server_handler.expected_clients_in_round and server_handler.expected_clients_in_round > 0)

            if cond1 or cond2 or cond3:
                should_break_loop = True
        
        if should_break_loop:
            print("Fase inicial (P(Xi,Y)) completada, JMI iniciado o todos los clientes esperados fallaron antes.")
            break
        
        time.sleep(0.1)
    else: 
        print(f"Timeout esperando finalización de la fase inicial después de {initial_phase_monitor_timeout}s.")
    elapsed_since_global_start = time.time() - global_start_time
    remaining_timeout_for_fs = TIMEOUT_SECONDS_OVERALL - elapsed_since_global_start

    #GESTIONA HILOS JMI
    if MI_FS_METHOD == "JMI":
        jmi_thread_to_join = None
        
        # Esperar un poco a que se cree el objeto del hilo JMI
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

    #GESTION HILOS MIM
    elif MI_FS_METHOD == "MIM" or (MI_FS_METHOD != "JMI"): 
        print(f"Método es {MI_FS_METHOD}. Esperando finalización del hilo de trabajo correspondiente...")
        aggregation_thread_to_join = server_handler.aggregation_thread
        
        if aggregation_thread_to_join is not None:
            if aggregation_thread_to_join.is_alive():
                print(f"Hilo de {MI_FS_METHOD} detectado. Haciendo join (timeout restante: {max(0, remaining_timeout_for_fs):.2f}s)...")
                aggregation_thread_to_join.join(timeout=max(0, remaining_timeout_for_fs))
                if aggregation_thread_to_join.is_alive():
                    print(f"ADVERTENCIA: Timeout esperando al hilo de {MI_FS_METHOD} (join).")

            
    print("Resumen final del estado de los clientes (post-procesamiento):")

    global_end_time = time.time()
    total_elapsed_time = global_end_time - global_start_time
    print(f"---------------------------------------------------------------------")
    print(f"--- TIEMPO TOTAL DE EJECUCIÓN: {total_elapsed_time:.4f} segundos ---")
    print(f"---------------------------------------------------------------------")
    #Parar trackers de consumo
    server_handler.send_emission_request_to_clients()
    
    print("Desconectando comunicador MQTT...")
    time.sleep(1) 
    if communicator: communicator.disconnect()

    print("Servidor y orquestador finalizados.")
    sys.exit(0) 

if __name__ == "__main__":
    main()