import json
import pickle
import base64
import zlib
import numpy as np
import os
import sys
import paho.mqtt.client as mqtt
import threading
from typing import Optional

# --- Configuración de Rutas e Imports ---
SCRIPT_DIR_APP = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_APP = os.path.dirname(SCRIPT_DIR_APP) 

if PROJECT_ROOT_APP not in sys.path: sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR_APP, '..'))) # Ir un nivel arriba para el proyecto raíz
try:
    from jmi_orchestrator import JMIOrchestrator
    from mqtt_handlers.mqtt_communicator import MQTTCommunicator
    from client_sim_state import ClientSimState
    from server_emissions_manager import ServerEmissionsManager 
    from feature_selector import select_features_mim
 
except ImportError as e:
    print(f"ERROR crítico importando módulos: {e}. Verifique PYTHONPATH.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)
    
SERVER_ID_PREFIX_PC = "pc_server_ctrl"

#TOPICS PARA MQTT
COMMAND_TOPIC = "tfg/fl/pi/command"
STATUS_TOPIC = "tfg/fl/pi/status"
LOCAL_EXTREMES_TOPIC = "tfg/fl/pi/local_extremes"
GLOBAL_DISC_PARAMS_TOPIC = "tfg/fl/pi/global_disc_params"
DATA_RESULTS_TOPIC = "tfg/fl/pi/data_results" 
JMI_PAIR_PROB_RESULTS_TOPIC = "tfg/fl/pi/jmi_pair_prob"
EMISSIONS_DATA_TOPIC = "tfg/fl/pi/emissions_data"
TIMER_DATA_TOPIC = "tfg/fl/pi/bench"


class ServerLogic:
    def __init__(self, project_root_path):
        self.project_root_app = project_root_path # Para guardar archivos, etc.

        # Estado de la ronda y configuración
        self.active_sim_clients = {}
        self.mi_method= "MIM"
        self.top_k_features = 15
        self.aggregation_method = "simple"
        self.global_num_bins = 5 
        self.expected_clients_in_round = 0

        # Colecciones de datos de la ronda
        self.collected_local_extremes = {}
        self.clients_reported_extremes_count = 0
        self.collected_initial_XY_prob_tables = {}
        self.bench_per_client = {} 
        self.clients_reported_initial_XY_count = 0
        
        # Componentes de JMI
        self.jmi_lock = threading.Lock() 
        self.current_jmi_orchestrator = None
        self.jmi_orchestrator_thread = None 
        self.aggregation_thread = None

        # Comunicador (se pasará desde server_app.py)
        self.communicator: Optional[MQTTCommunicator] = None
        
        # Clientes emisiones
        self.emissions_manager = ServerEmissionsManager(self.project_root_app, SERVER_ID_PREFIX_PC)

        print("ServerLogic instanciado.")

    def set_communicator(self, comm_instance: MQTTCommunicator):
        """
            Función para inyectar comunicador MQTT
        """
        self.communicator = comm_instance
        self.emissions_manager.set_dependencies(self.communicator, self.jmi_lock, self.active_sim_clients)

    def set_round_parameters(self, mi_method, top_k, aggregation_method_str, num_bins):
        """
            Función para sobreescribir los valores de configuración por defecto
        """
        self.mi_method = mi_method
        self.top_k_features = top_k
        self.aggregation_method = aggregation_method_str
        self.global_num_bins = num_bins

    def initialize_new_round(self, num_clients_expected_param):
        """
            Inicializa una nueva ronda, se ejecuta con una vez en server_app al inicializar el programa
        """

        with self.jmi_lock:
            self.expected_clients_in_round = num_clients_expected_param
            self.clients_reported_extremes_count = 0
            self.collected_local_extremes.clear()
            self.clients_reported_initial_XY_count = 0
            self.collected_initial_XY_prob_tables.clear()
            self.bench_per_client.clear()
            
            #Emisiones
            self.emissions_manager.reset_server_tracking()

            if self.current_jmi_orchestrator:
                self.current_jmi_orchestrator = None
            
            if (self.jmi_orchestrator_thread and self.jmi_orchestrator_thread.is_alive()) or (self.aggregation_thread and self.aggregation_thread.is_alive()):
                print("Advertencia - Hilo JMI o MIM anterior aún podría estar activo al inicializar nueva ronda.")
            self.jmi_orchestrator_thread = None
            self.aggregation_thread = None 

        print(f"Nueva ronda inicializada. Esperando {self.expected_clients_in_round} clientes. Método: {self.mi_method}, TopK: {self.top_k_features}")

    def handle_client_bench_update(self, bench_json):
        """
            Almacena los datos de tiempos tomados del cliente.
        """
        cid = bench_json.get("sim_client_id")
        comp = bench_json.get("compute_s")
        comm = bench_json.get("comm_s")
        
        if not cid or comp is None or comm is None:
            return
        entry = self.bench_per_client.setdefault(cid, {"pre": 0.0, "compute": 0.0, "comm": 0.0})
        entry["pre"]     += float(bench_json.get("pre_s", 0.0))
        entry["compute"] += float(comp)
        entry["comm"]   += float(comm)
    

    def get_bench_summary(self):
        """
            Devuelve los datos de tiempos tomados del cliente.
        """
        if not self.bench_per_client:
            return 0.0, 0.0, 0.0
        pre_max     = max(d["pre"]     for d in self.bench_per_client.values())
        max_compute = max(d["compute"] for d in self.bench_per_client.values())
        sum_comm    = sum(d["comm"]   for d in self.bench_per_client.values())
        return pre_max, max_compute, sum_comm
    
    
    def add_or_update_active_client(self, sim_client_id, dataset_name):
        """Añade un nuevo cliente o actualiza uno existente para la ronda."""
        if sim_client_id not in self.active_sim_clients:
            self.active_sim_clients[sim_client_id] = ClientSimState(sim_client_id, dataset_name_assigned=dataset_name)
        else:
            self.active_sim_clients[sim_client_id].dataset_name = dataset_name
            self.active_sim_clients[sim_client_id].error_message = None 

    def send_processing_command_to_pi(self, sim_client_id, dataset_name, indices_list, num_global_classes_param):
        """
            Comunicación inicial con los dispositivos del borde, se envia la parte del dataset que manejan, llamado una vez en server_app
            inicia el bucle de comunicación
        """
        if not self.communicator:
            print("ERROR: Comunicador no establecido en ServerLogic.")
            return False

        # La gestión de active_sim_clients (crear ClientSimState) se hace aquí o antes
        self.add_or_update_active_client(sim_client_id, dataset_name)
        self.emissions_manager.start_server_tracking()
        print(f"ServerLogic: Enviando comando a {sim_client_id} con num_global_classes: {num_global_classes_param}")
        command_payload = {
            "action": "PROCESS_SIMULATED_CLIENT",
            "sim_client_id": sim_client_id,
            "dataset_name": dataset_name,
            "data_indices": indices_list,
            "num_global_classes": num_global_classes_param
        }
        msg_info = self.communicator.publish(COMMAND_TOPIC, command_payload, qos=1)
        if not (msg_info and msg_info.rc == mqtt.MQTT_ERR_SUCCESS):
            if sim_client_id in self.active_sim_clients:
                self.active_sim_clients[sim_client_id].error_message = "Fallo al publicar comando inicial."
            print(f"ERROR al publicar comando inicial para {sim_client_id}. RC: {msg_info.rc if msg_info else 'N/A'}")
            return False
        return True

    def _process_and_dispatch_global_parameters(self):
        """
            Tras recibir los max/min locales de todos los clientes, esta función calcula los min/max globales y los envia al borde
        """
        if not self.communicator or not self.collected_local_extremes: 
            print("Comunicador no disponible para enviar parámetros globales o no hay extremos locales recolectados para procesar.")
            return
        
        num_features = 0
        valid_extremes_sources = [ext_val for client_id, ext_val in self.collected_local_extremes.items() if client_id in self.active_sim_clients and self.active_sim_clients[client_id]]
        
        first_valid_extremes = next((ext for ext in valid_extremes_sources if ext and isinstance(ext, list) and len(ext) > 0), None)
        if first_valid_extremes: num_features = len(first_valid_extremes)
        
        if num_features == 0:
            for cid_err in self.collected_local_extremes: 
                if cid_err in self.active_sim_clients and self.active_sim_clients.get(cid_err): 
                    self.active_sim_clients[cid_err].error_message = "Fallo servidor: num_features para params globales."
            return
        
        aggregated_global_ranges = []
        for i in range(num_features):
            mins_vals = [ext[i][0] for ext in valid_extremes_sources if ext and i < len(ext) and isinstance(ext[i], list) and len(ext[i])==2 and ext[i][0] is not None]
            maxs_vals = [ext[i][1] for ext in valid_extremes_sources if ext and i < len(ext) and isinstance(ext[i], list) and len(ext[i])==2 and ext[i][1] is not None]
            if not mins_vals or not maxs_vals: 
                print(f"Característica {i} sin valores min/max válidos. Usando [0.0, 0.0].")
                aggregated_global_ranges.append([0.0, 0.0])
            else: aggregated_global_ranges.append([min(mins_vals), max(maxs_vals)])
                
        clients_to_send_params = set(self.collected_local_extremes.keys()).intersection(set(self.active_sim_clients.keys()))
        for sid_target in clients_to_send_params:
            client_state_target = self.active_sim_clients.get(sid_target)
            if client_state_target: 
                payload = {
                    "sim_client_id": sid_target, 
                    "dataset_name": client_state_target.dataset_name, 
                    "num_bins": self.global_num_bins, 
                    "global_feature_ranges": aggregated_global_ranges
                }
                msg_info = self.communicator.publish(GLOBAL_DISC_PARAMS_TOPIC, payload, qos=1)
                if msg_info and msg_info.rc == mqtt.MQTT_ERR_SUCCESS: 
                    client_state_target.global_params_sent_to_client = True
                else: 
                    client_state_target.error_message = f"Fallo al pub params globales. RC: {msg_info.rc if msg_info else 'N/A'}"
                    print(f"ERROR publicando params globales. RC: {msg_info.rc if msg_info else 'N/A'}")

    def _aggregate_prob_tables_common(self, collected_data_dict, expected_shape_dims=None):
        """
            Agrega las tablas de probabilidad calculadas en el borde al resto de dispositivos
        """
        aggregated_p = None
        ref_shape = None
        dataset_name_log = "N/A"
        count_valid_contributions = 0
        total_samples_for_weighting = 0

        active_sim_clients_ref = self.active_sim_clients # Usar el de la instancia

        if active_sim_clients_ref:
            first_client_id_for_log = next((cid for cid in collected_data_dict if cid in active_sim_clients_ref and active_sim_clients_ref.get(cid)), None)
            if first_client_id_for_log:
                client_state_log = active_sim_clients_ref.get(first_client_id_for_log)
                if client_state_log:
                    dataset_name_log = client_state_log.dataset_name

        for client_key, data_tuple in collected_data_dict.items():            
            prob_array, num_samples = data_tuple
            
            if num_samples >= 0:
                if expected_shape_dims and prob_array.ndim != expected_shape_dims:
                    print(f"ERROR - Dimensión tabla: {prob_array.ndim} (esperada {expected_shape_dims}) de {client_key}.")
                    continue
                
                if ref_shape is None: # Primera tabla válida
                    ref_shape = prob_array.shape
                    aggregated_p = np.zeros(ref_shape, dtype=float)
                
                if prob_array.shape == ref_shape:
                    if self.aggregation_method == "weighted": # Usar self.aggregation_method
                        if num_samples > 0:
                            aggregated_p += (prob_array * num_samples)
                        total_samples_for_weighting += num_samples
                    else: # Promedio simple
                        aggregated_p += prob_array
                    count_valid_contributions += 1
                else:
                    print(f"ERROR - Discrepancia formas de {client_key}. Esperada: {ref_shape}, Obtenida: {prob_array.shape}.")
            else:
                print(f"ERROR - num_samples inválidos de {client_key}.")

        if aggregated_p is None or count_valid_contributions == 0:
            return None, None, dataset_name_log, 0

        if self.aggregation_method == "weighted":
            if total_samples_for_weighting > 0:
                aggregated_p /= total_samples_for_weighting
            elif count_valid_contributions > 0: # Fallback a promedio simple si todas las muestras válidas son 0
                print(f"AGREGACIÓN ({dataset_name_log}): Ponderada fallback a simple (0 muestras para ponderar). Contribuciones: {count_valid_contributions}.")
                # Necesitamos recalcular aggregated_p si solo se sumaron contribuciones con peso 0
                aggregated_p_simple_fallback = np.zeros(ref_shape, dtype=float)
                actual_contrib_for_simple_fallback = 0
                for _ , data_tuple_fb in collected_data_dict.items(): # Iterar de nuevo para el fallback
                    prob_array_fb, num_samples_fb = data_tuple_fb
                    if prob_array_fb is not None and prob_array_fb.shape == ref_shape and isinstance(num_samples_fb, int) and num_samples_fb >=0: # Solo válidas
                        aggregated_p_simple_fallback += prob_array_fb
                        actual_contrib_for_simple_fallback +=1
                if actual_contrib_for_simple_fallback > 0:
                    aggregated_p = aggregated_p_simple_fallback / actual_contrib_for_simple_fallback
                else: # No debería ocurrir si count_valid_contributions > 0
                    return None, None, dataset_name_log, 0
            else:
                print(f"AGREGACIÓN ({dataset_name_log}): Error en ponderación (0 muestras y 0 contribuciones).")
                return None, None, dataset_name_log, 0
        else: # Promedio simple
            aggregated_p /= count_valid_contributions
                    
        return aggregated_p, ref_shape, dataset_name_log, count_valid_contributions


    def aggregate_initial_XY_prob_tables_and_trigger_selection(self):
        """
            Agrega la primera tabla generada en el borde de la red y decide el siguiente paso del algoritmo según el metodo de FS seleccionado
            si es JMI inicia el JMI_Orchestrator y si es MIM lista y ordena las caracteristicas según su relevancia
        """
        num_reporting_clients = len(self.collected_initial_XY_prob_tables)
        if not self.collected_initial_XY_prob_tables or num_reporting_clients == 0:
            print(f"No hay tablas P(Xi,Y) iniciales ({num_reporting_clients}/{self.expected_clients_in_round}). No se procede.")
            return
        if num_reporting_clients < self.expected_clients_in_round:
            print(f"No todas las tablas P(Xi,Y) recibidas ({num_reporting_clients}/{self.expected_clients_in_round}). Procediendo con disponibles.")

        aggregated_p_xy_all, _, ds_name, valid_count = self._aggregate_prob_tables_common(
            self.collected_initial_XY_prob_tables,
            expected_shape_dims=3
        )
        
        if aggregated_p_xy_all is None or valid_count == 0:
            print("Fallo en agregación de P(Xi,Y) iniciales o ninguna tabla válida.")
            return

        print(f"Agregación P(Xi,Y) completada ({valid_count} clientes). Dataset: {ds_name}, Forma: {aggregated_p_xy_all.shape}")
        
        if self.mi_method == "MIM":
            selected_indices, _ = select_features_mim(aggregated_p_xy_all, self.top_k_features)
            if selected_indices is not None and len(selected_indices) > 0 : 
                print(f"Características (MIM): {selected_indices}")
                self.save_selected_federated_features_txt(
                    selected_indices,
                    ds_name,
                    len(selected_indices),
                    "MIM_federated"
                )
            else: 
                print(f"MIM no seleccionó características o devolvió una lista vacía.")
            with self.jmi_lock:
                self.collected_initial_XY_prob_tables.clear()
                #self.clients_reported_initial_XY_count = 0
            

        elif self.mi_method == "JMI":
            if self.jmi_orchestrator_thread and self.jmi_orchestrator_thread.is_alive():
                print("Hilo JMI anterior aún activo. No se iniciará uno nuevo.")
            else:
                print("Iniciando JMIOrchestrator en hilo...")
                num_clients_for_jmi = 0
                with self.jmi_lock: # Proteger acceso a active_sim_clients
                    num_clients_for_jmi = len([cid for cid in self.active_sim_clients 
                                               if self.active_sim_clients[cid] and not self.active_sim_clients[cid].error_message])
                
                if num_clients_for_jmi == 0 and self.expected_clients_in_round > 0:
                    print("No hay clientes activos para iniciar JMI.")
                    with self.jmi_lock:
                        self.collected_initial_XY_prob_tables.clear()
                        self.clients_reported_initial_XY_count = 0 
                    return

                # Crear el orquestador JMI y el hilo dentro del lock para consistencia
                with self.jmi_lock:
                    self.current_jmi_orchestrator = JMIOrchestrator(
                        comm_instance=self.communicator,
                        active_clients_dict=self.active_sim_clients.copy(), 
                        num_expected_clients=num_clients_for_jmi, # Usar el recuento de activos reales
                        global_lock=self.jmi_lock,
                        aggregate_func_ref=self._aggregate_prob_tables_common, # Pasar el método de instancia
                        dataset_name_for_save=ds_name,
                        save_function_ref=self.save_selected_federated_features_txt, # Pasar el método de instancia
                    )
                
                    self.jmi_orchestrator_thread = threading.Thread(
                        target=self.current_jmi_orchestrator.start_selection, 
                        args=(aggregated_p_xy_all, self.top_k_features,), 
                        name="JMIOrchestrationThread_Cls"
                    )
                    self.jmi_orchestrator_thread.daemon = True # Para que termine si el programa principal termina
                    self.jmi_orchestrator_thread.start()
        else: 
            print(f"Método '{self.mi_method}' no reconocido. Usando MIM por defecto.")
            selected_indices, exec_time = select_features_mim(aggregated_p_xy_all, self.top_k_features)
            if selected_indices is not None and len(selected_indices) > 0:
                print(f"Características (MIM por defecto): {selected_indices} (Tiempo: {exec_time:.4f}s)")
                self.save_selected_federated_features_txt(
                    selected_indices, ds_name, len(selected_indices), 
                    "MIM_default_federated", self.project_root_app
                )
            else:
                print("MIM (por defecto) no seleccionó características.")
            with self.jmi_lock:
                self.collected_initial_XY_prob_tables.clear()
                self.clients_reported_initial_XY_count = 0
        

    # --- Métodos manejadores de mensajes MQTT ---
    def handle_pi_status_update(self, status_data):
        """
            Maneja los mensajes recibidos del borde y cambia el estado del cliente en consecuencia, si la comunicación JMI esta en curso
            maneja la flag que bloquea el bucle para seleccionar una caracteristica nueva, la desbloquea cuando todos los clientes hicieron su tarea
        """
        sim_client_id = status_data.get("sim_client_id")
        status = status_data.get("status") 
        if not sim_client_id:
            return
        
        client_state = None
        with self.jmi_lock: 
            if sim_client_id in self.active_sim_clients:
                 client_state = self.active_sim_clients.get(sim_client_id)
        
        if client_state:
            if status == "RECEIVED_COMMAND_AND_INDICES": client_state.command_acked = True
            elif status == "GLOBAL_PARAMS_RECEIVED_ACK": client_state.global_params_acked_by_client = True
            elif status == "LOCAL_XY_PROB_DIST_PUBLISHED": client_state.local_XY_prob_dist_published = True
            elif status == "ERROR": 
                client_state.error_message = status_data.get("error_message", "Error no especificado de cliente.")
                print(f"[{sim_client_id}]: ¡ERROR reportado por cliente!: {client_state.error_message}")
        
        if status == "JMI_BATCH_TRIPLETS_PUBLISHED":
            request_id_status = status_data.get("request_id") 
            with self.jmi_lock:
                if self.current_jmi_orchestrator:
                    orchestrator = self.current_jmi_orchestrator
                    if orchestrator.current_batch_request_id and request_id_status == orchestrator.current_batch_request_id:
                        if sim_client_id in orchestrator.clients_reported_current_batch_status:
                            orchestrator.clients_reported_current_batch_status[sim_client_id] = True
                        
                        all_reported_for_batch = True
                        if not orchestrator.clients_reported_current_batch_status: # Si el dict está vacío
                            all_reported_for_batch = False
                        else:
                            for reported_status_val in orchestrator.clients_reported_current_batch_status.values():
                                if not reported_status_val:
                                    all_reported_for_batch = False
                                    break
                            
                        if all_reported_for_batch and not orchestrator.all_clients_reported_batch_event.is_set():
                            orchestrator.all_clients_reported_batch_event.set()
            

    def handle_pi_local_extremes(self, extremes_data):
        """
            Maneja la recepcion de los extremos locales, los almacena y cuando llegan todos lanza un thread para generar los máximos y mínimos globales
        """
        sim_client_id = extremes_data.get("sim_client_id")
        local_min_max_list = extremes_data.get("feature_min_max")
        if not sim_client_id:
            return

        with self.jmi_lock: 
            if sim_client_id not in self.active_sim_clients:
                print(f"SERVER_LOGIC: Extremos locales de cliente desconocido {sim_client_id} ignorados.")
                return
            
            client_state = self.active_sim_clients.get(sim_client_id)
            if not client_state: # No debería pasar si está en active_sim_clients
                print(f"SERVER_LOGIC: No se encontró estado para cliente {sim_client_id} al recibir extremos.")
                return
            if client_state.local_extremes_received:
                print(f"SERVER_LOGIC: Extremos locales de {sim_client_id} ya recibidos. Ignorando duplicado.")
                return 

            if local_min_max_list is not None and isinstance(local_min_max_list, list):
                if not all(isinstance(item, list) and len(item) == 2 and \
                           all(isinstance(val, (int, float, np.number, type(None))) or np.isnan(val) for val in item) \
                           for item in local_min_max_list if item is not None): # Permitir np.isnan
                    client_state.error_message = "Formato inválido para feature_min_max."
                    print(f"SERVER_LOGIC: {client_state.error_message} para {sim_client_id}")
                    return
                
                client_state.feature_min_max_local = local_min_max_list
                client_state.local_extremes_received = True
                
                if not local_min_max_list:
                    client_state.error_message = "Lista min/max vacía recibida."
                    print(f"SERVER_LOGIC: {client_state.error_message} de {sim_client_id}")
                else: 
                    if sim_client_id not in self.collected_local_extremes: 
                        self.collected_local_extremes[sim_client_id] = local_min_max_list
                        self.clients_reported_extremes_count += 1
                        print(f"SERVER_LOGIC: Extremos locales de {sim_client_id} recibidos ({self.clients_reported_extremes_count}/{self.expected_clients_in_round}).")
                
                # Comprobar si todos los clientes esperados (que no tienen error previo grave) han reportado
                # El expected_clients_in_round se ajusta en server_app si algunos clientes no tienen datos.
                if self.clients_reported_extremes_count >= self.expected_clients_in_round and self.expected_clients_in_round > 0:
                    print(f"SERVER_LOGIC [EXTREMES]: Todos los {self.expected_clients_in_round} clientes esperados han reportado extremos. Iniciando envío de params globales.")
                    # Usar el método _process_and_dispatch_global_parameters de la instancia
                    dispatch_thread = threading.Thread(target=self._process_and_dispatch_global_parameters)
                    dispatch_thread.daemon = True 
                    dispatch_thread.start()
            else:
                client_state.error_message = "Payload 'local_extremes' inválido o lista feature_min_max es None."
                print(f"SERVER_LOGIC: {client_state.error_message} de {sim_client_id}")

    def handle_initial_XY_prob_results(self, data_payload):
        """
            Maneja el almacenamiento de la primera recepcion de tablas de probabilidad.
        """
        sim_client_id = data_payload.get("sim_client_id")
        pickled_b64_str = data_payload.get("pickled_data_base64")
        prob_table_type = data_payload.get("prob_table_type")
        is_compressed = data_payload.get("is_compressed", False)
        num_samples_from_client = data_payload.get("num_samples")

        if not sim_client_id:
            return

        with self.jmi_lock:            
            client_state = self.active_sim_clients.get(sim_client_id)
            if sim_client_id not in self.active_sim_clients or not client_state or client_state.local_XY_prob_dist_received or prob_table_type != "XY":
                print(f"Ha surgido un error con el cliente {sim_client_id} o las tablas recibidas.")
                return

            if pickled_b64_str:
                try:
                    decoded_data_bytes = base64.b64decode(pickled_b64_str)
                    data_to_unpickle = zlib.decompress(decoded_data_bytes) if is_compressed else decoded_data_bytes
                    
                    arr = pickle.loads(data_to_unpickle)

                    client_state.local_XY_prob_dist_array = arr
                    client_state.local_XY_prob_dist_received = True
                        
                    num_samples_to_store = 0 
                    if isinstance(num_samples_from_client, int) and num_samples_from_client >= 0:
                        num_samples_to_store = num_samples_from_client

                        
                    if sim_client_id not in self.collected_initial_XY_prob_tables: 
                        self.collected_initial_XY_prob_tables[sim_client_id] = (arr, num_samples_to_store)
                        self.clients_reported_initial_XY_count += 1
                        print(f"P(Xi,Y) de {sim_client_id} recibido ({self.clients_reported_initial_XY_count}/{self.expected_clients_in_round}).")
                        
                    if self.clients_reported_initial_XY_count >= self.expected_clients_in_round and self.expected_clients_in_round > 0:
                        if not self.aggregation_thread or not self.aggregation_thread.is_alive(): # Evitar lanzar múltiples
                            print(f"Todos los {self.expected_clients_in_round} clientes esperados han reportado P(Xi,Y). Iniciando agregación y selección.")
                                # Guardar la referencia al hilo
                            self.aggregation_thread = threading.Thread(target=self.aggregate_initial_XY_prob_tables_and_trigger_selection)
                            self.aggregation_thread.daemon = True 
                            self.aggregation_thread.start()

                except Exception as e:
                    client_state.error_message = f"Error deserializando P(Xi,Y) de {sim_client_id}: {type(e).__name__} - {e}"
                    print(f"{client_state.error_message}")
            else:
                client_state.error_message = "Payload P(Xi,Y) sin datos (pickled_b64_str es None o vacío)."
                print(f"{client_state.error_message} de {sim_client_id}")

    def handle_jmi_pair_prob_result(self, data_payload):
        """
            Procesa un lote de tablas de probabilidad de tripletas P(Xk,Xj,Y) enviado por un cliente para una iteración JMI específica.
        """
        sim_client_id = data_payload.get("sim_client_id")
        request_id_rcv = data_payload.get("request_id")
        num_samples_client = data_payload.get("num_samples_client") # num_samples del cliente que generó este batch
        
        pickled_batch_b64_str = data_payload.get("pickled_batch_data_base64")
        prob_table_type = data_payload.get("prob_table_type") # Debería ser "XkXjY_BATCH"
        is_compressed = data_payload.get("is_compressed", False)

        if not sim_client_id:
            return
        
        if prob_table_type != "XkXjY_BATCH":
            print(f"Recibido tipo de tabla inesperado '{prob_table_type}' de {sim_client_id} para JMI. Esperando 'XkXjY_BATCH'.")
            return 
        if not pickled_batch_b64_str:
            with self.jmi_lock:
                if sim_client_id in self.active_sim_clients and self.active_sim_clients.get(sim_client_id):
                    self.active_sim_clients[sim_client_id].error_message = f"Payload de LOTE JMI ({request_id_rcv}) sin datos."
            print(f"Payload de LOTE JMI de {sim_client_id} (req: {request_id_rcv}) sin datos.")
            return

        batch_triplet_results_list = []
        try:
            decoded_data_bytes = base64.b64decode(pickled_batch_b64_str)
            data_to_unpickle = zlib.decompress(decoded_data_bytes) if is_compressed else decoded_data_bytes
            batch_triplet_results_list = pickle.loads(data_to_unpickle)
            if not isinstance(batch_triplet_results_list, list):
                raise ValueError("El payload del lote JMI deserializado no es una lista.")

        except Exception as e:
            print(f"Error deserializando LOTE JMI de {sim_client_id} (req: {request_id_rcv}): {type(e).__name__} - {e}")
            return
        
        processed_count_for_log = 0
        with self.jmi_lock:
            if sim_client_id not in self.active_sim_clients: # Cliente ya no activo
                return
            if not self.current_jmi_orchestrator: # JMI no está activo
                print(f"Recibido lote JMI de {sim_client_id} pero JMIOrchestrator no está activo. Lote ignorado.")
                return

            orchestrator = self.current_jmi_orchestrator
            
            for item in batch_triplet_results_list:
                feature_pair_rcv = item.get("feature_pair")
                triplet_table_l = item.get("triplet_table")
                pair_key = tuple(sorted(feature_pair_rcv))
                
                if pair_key not in orchestrator.current_iter_triplet_tables_from_clients:
                    orchestrator.current_iter_triplet_tables_from_clients[pair_key] = {
                        'tables_list': [], 
                        'received_from_clients_count': 0,
                        'aggregated_table': None, 
                        'client_sources': set()
                    }
                
                entry = orchestrator.current_iter_triplet_tables_from_clients[pair_key]
                if sim_client_id not in entry['client_sources']:
                    # num_samples_client es el número de muestras del cliente que generó este batch
                    # Es importante para la agregación ponderada de estas tablas de tripletas
                    if not isinstance(num_samples_client, int) or num_samples_client < 0:
                        num_samples_for_entry = 0
                    else:
                        num_samples_for_entry = num_samples_client

                    entry['tables_list'].append((triplet_table_l, num_samples_for_entry))
                    entry['received_from_clients_count'] += 1 
                    entry['client_sources'].add(sim_client_id)
                    processed_count_for_log +=1
        
        if processed_count_for_log > 0:
             print(f"Procesadas {processed_count_for_log}/{len(batch_triplet_results_list)} tablas del lote JMI de {sim_client_id}.")
        else: # batch_triplet_results_list no estaba vacío, pero no se procesó nada (ej. todo inválido)
             print(f"Lote JMI de {sim_client_id} para req_id {request_id_rcv} no contenía tablas válidas o ya procesadas.")
        

    def on_server_message_received(self, topic, payload_bytes):
        """
            Maneja la recepcion de mensajes de los dispositivos del borde de la red.
        """
        payload_str = ""
        try:
            payload_str = payload_bytes.decode('utf-8') 
            data = json.loads(payload_str)
            if topic == STATUS_TOPIC: self.handle_pi_status_update(data)
            elif topic == DATA_RESULTS_TOPIC: self.handle_initial_XY_prob_results(data)
            elif topic == LOCAL_EXTREMES_TOPIC: self.handle_pi_local_extremes(data)
            elif topic == JMI_PAIR_PROB_RESULTS_TOPIC:
                if data.get("prob_table_type") == "XkXjY_BATCH":
                    self.handle_jmi_pair_prob_result(data)
            elif topic == EMISSIONS_DATA_TOPIC:
                self.emissions_manager.process_client_emission_report(data)
            elif topic == TIMER_DATA_TOPIC:
                self.handle_client_bench_update(data)
        except Exception as e: 
            print(f"Error general procesando mensaje en '{topic}': {type(e).__name__} - {e}.")


    def save_selected_federated_features_txt(self, selected_feature_indices, dataset_name_str, actual_k_selected, technique_name): 
        """
        Guarda los índices de las características seleccionadas por el proceso federado en un archivo .txt.
        """
        main_datasets_folder = "selected_features"
        output_dir = os.path.join(self.project_root_app, main_datasets_folder)
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"{dataset_name_str}_federated_selected_top{actual_k_selected}_{technique_name}_feature_indices.txt"
        output_filepath = os.path.join(output_dir, output_filename)

        try:
            with open(output_filepath, 'w') as f:
                for feature_index in selected_feature_indices:
                    f.write(f"{int(feature_index)}\n") 
            print(f"Índices de características federadas ({technique_name}) guardados en: {output_filepath}")
        except Exception as e:
            print(f"Error guardando el archivo .txt de índices de características en '{output_filepath}': {e}")

    def send_emission_request_to_clients(self):
        """
            Petición a los clientes para que envien sus datos de consumo y emisiones.
        """
        self.emissions_manager.request_emissions_from_clients()

    def on_connected_to_broker(self):
        """
            Topics a los que esta suscrito el servidor.
        """
        print("Conectado al broker MQTT.")
        if self.communicator:
            self.communicator.subscribe(STATUS_TOPIC, qos=1)
            self.communicator.subscribe(LOCAL_EXTREMES_TOPIC, qos=1)
            self.communicator.subscribe(DATA_RESULTS_TOPIC, qos=1) 
            self.communicator.subscribe(JMI_PAIR_PROB_RESULTS_TOPIC, qos=1) 
            self.communicator.subscribe(EMISSIONS_DATA_TOPIC, qos=0)
            self.communicator.subscribe(TIMER_DATA_TOPIC, qos=0) 
            print(f"Suscrito a topics relevantes.")

    def on_disconnected_from_broker(self, rc):
        print(f"Desconectado del broker MQTT. Código: {rc}.")