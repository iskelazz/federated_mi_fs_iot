import json
import logging
import pickle
import base64
import zlib
import numpy as np
import os
import sys
import argparse



# --- Configuración de Rutas para Importación ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MQTT_HANDLERS_DIR = os.path.join(PROJECT_ROOT, 'mqtt_handlers')

if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
if MQTT_HANDLERS_DIR not in sys.path: sys.path.insert(0, MQTT_HANDLERS_DIR)

try:
    from timer import Timer
    from utils import load_dataset, discretize_equalwidth
    from client_utils import calculate_local_prob_dist_array, calculate_local_triplet_prob_dist
    from mqtt_handlers.mqtt_communicator import MQTTCommunicator
    from client_emissions_manager import ClientEmissionsManager
except ImportError as e:
    print(f"CLIENT_APP: ERROR crítico importando módulos: {e}. Verifique PYTHONPATH y la estructura de directorios.")
    sys.exit(1)
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("client_app")
   
CLIENT_ID_PREFIX_PI = "pi_fs_node" # Prefijo para el ID de cliente MQTT

# Topics a los que el cliente se suscribe o publica
COMMAND_TOPIC = "tfg/fl/pi/command" # Cliente se suscribe para recibir comandos del servidor
GLOBAL_DISC_PARAMS_TOPIC = "tfg/fl/pi/global_disc_params" # Cliente se suscribe para params de discretización

STATUS_TOPIC = "tfg/fl/pi/status" # Cliente publica actualizaciones de estado
LOCAL_EXTREMES_TOPIC = "tfg/fl/pi/local_extremes" # Cliente publica sus min/max locales
DATA_RESULTS_TOPIC = "tfg/fl/pi/data_results" # Cliente publica tablas P(Xi,Y) iniciales
JMI_PAIR_PROB_RESULTS_TOPIC = "tfg/fl/pi/jmi_pair_prob" # Cliente publica tablas P(Xk,Xj,Y) para JMI
EMISSIONS_DATA_TOPIC = "tfg/fl/pi/emissions_data" #Cliente publica los datos de consumo al servidor

BENCH_TOPIC = "tfg/fl/pi/bench"

class ClientApp:
    def __init__(self, sim_id):
        
        config = ClientApp._load_simulation_config(PROJECT_ROOT)
        
        self.sim_id = sim_id 
        self.broker_address = config["BROKER_ADDRESS_FOR_CLIENT"]
        self.port = config["PORT"]
        self.tracker = None
        # Atributos para el estado interno
        self.current_job_data = self._initialize_job_state() 
        
        # Atributo para el comunicador MQTT
        safe_sim_id_part = self.sim_id.replace('sim_client_','').replace('_','-')[:20]
        mqtt_instance_client_id = f"{CLIENT_ID_PREFIX_PI}_{safe_sim_id_part}"
        self.communicator = MQTTCommunicator(self.broker_address, self.port, client_id_prefix=mqtt_instance_client_id)
        
        self._setup_mqtt_callbacks()
    
    @staticmethod
    def _load_simulation_config(project_root_path, config_filename="config.json"):
        """Carga la configuración desde un archivo JSON."""
        config_filepath = os.path.join(project_root_path, config_filename)
        default_config = { 
        "BROKER_ADDRESS_FOR_CLIENT": "localhost",
        "PORT": 1883,
        }
        try:
            with open(config_filepath, 'r') as f:
                all_config = json.load(f)
            print(f"Configuración cargada desde '{config_filepath}'.")
            config = all_config.get("FS_FEDERATED")
            if config is None:
                print(f"Advertencia: La clave 'FS_FEDERATED' no se encontró en '{config_filepath}'. "
                  f"Usando la configuración por defecto completa para 'FS_FEDERATED'.")
                # Si "FS_FEDERATED" no está, devolvemos el default completo para esta sección.
                return default_config
            
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
                    print(f"Advertencia: Usando valor por defecto para '{key}': {default_config[key]}")
            return config
        except Exception as e:
            print(f"Error cargando configuración desde '{config_filepath}': {e}. Usando configuración por defecto.")
            return default_config 

    def _initialize_job_state(self):
        """Devuelve un diccionario con el estado inicial para un job."""
        return {
            "sim_client_id": None,
            "dataset_name": None,
            "X_client_partition": None,
            "y_client_partition": None,
            "X_client_discretized": None,
            "num_bins_global": None,
            "global_feature_ranges": None,
            "num_classes": None,
            "current_jmi_request_id": None
        }

    def _reset_current_job(self, job_sim_id):
        """Resetea el estado del job actual con el nuevo ID de trabajo."""
        
        self.current_job_data = self._initialize_job_state()
        self.current_job_data["sim_client_id"] = job_sim_id

    def _setup_mqtt_callbacks(self):
        """Configura los callbacks del cliente MQTT a los métodos de esta instancia."""
        self.communicator.set_message_callback(self.on_message_received)
        self.communicator.set_connect_callback(self.on_connected)
        self.communicator.set_disconnect_callback(self.on_disconnected)

    # --- Métodos para el ciclo de vida y procesamiento ---
    def start(self):
        """Conecta al cliente con el broker MQTT"""
        print(f"[{self.sim_id}]: Intentando conectar al broker MQTT...")
        if self.communicator.connect():
            try:
                print(f"[{self.sim_id}]: Conexión MQTT establecida. Escuchando mensajes...")
                self.communicator.loop_forever()
            except Exception as e_loop:
                print(f"[{self.sim_id}]: Error inesperado en el bucle principal: {type(e_loop).__name__} - {e_loop}")
            except KeyboardInterrupt as e:
                print(f"[{self.sim_id}]: Saliendo del cliente")
            finally:
                self.cleanup()
        else:
            print(f"[{self.sim_id}]: No se pudo conectar al broker MQTT. El cliente no se iniciará.")

    def cleanup(self):
        if self.tracker and self.tracker.is_tracking():
            print(f"CLIENT_APP [{self.sim_id}]: Deteniendo tracker de emisiones en cleanup.")
            log_id_cleanup = self.current_job_data.get("sim_client_id") or self.sim_id
            self.tracker.stop_tracking(log_id_cleanup) # No enviamos, solo paramos y obtenemos datos
        
        print(f"CLIENT_APP [{self.sim_id}]: Procediendo a limpieza y desconexión...")
        if self.communicator:
            self.communicator.disconnect()
        print(f"CLIENT_APP [{self.sim_id}]: Cliente finalizado.")


     # --- Callbacks de MQTT (convertidos a métodos) ---
    def on_connected(self):
        print(f"[{self.sim_id}]: Conectado al broker MQTT.")
        self.communicator.subscribe(COMMAND_TOPIC, qos=1)
        self.communicator.subscribe(GLOBAL_DISC_PARAMS_TOPIC, qos=1)
        print(f"CLIENT_APP [{self.sim_id}]: Suscrito a topics: {COMMAND_TOPIC}, {GLOBAL_DISC_PARAMS_TOPIC}")

    def on_disconnected(self, rc):
        print(f"CLIENT_APP [{self.sim_id}]: Desconectado del broker MQTT. Código de resultado: {rc}.")

    def on_message_received(self, topic, payload_bytes):
        """
            Gestiona los mensajes recibidos a traves de MQTT, los evalua y llama a las funciones adecuadas en consecuencia
        """
        log_id = self.current_job_data.get("sim_client_id") or self.sim_id
        try:
            payload_str = payload_bytes.decode('utf-8')
            message_data = json.loads(payload_str)

            msg_sim_client_id = message_data.get("sim_client_id")
            # Solo procesar mensajes destinados a esta instancia de cliente
            if msg_sim_client_id and msg_sim_client_id != self.sim_id:
                return 

            if topic == COMMAND_TOPIC:
                action = message_data.get("action")
                if action == "PROCESS_SIMULATED_CLIENT":
                    self.process_initial_command(message_data)
                elif action == "REQUEST_JMI_BATCH_TRIPLET_PROBS":
                    self.process_jmi_batch_triplet_request(message_data)
                elif action == "SEND_EMISSIONS_DATA": # Nuevo
                    self.stop_save_and_send_emissions(EMISSIONS_DATA_TOPIC) 
                else:
                    print(f"CLIENT_APP [{log_id}]: Acción '{action}' desconocida en {COMMAND_TOPIC}.")
            
            elif topic == GLOBAL_DISC_PARAMS_TOPIC:
                # El filtro por sim_client_id ya se aplicó arriba
                self.process_global_disc_params(message_data)

        except Exception as e:
            print(f"CLIENT_APP [{log_id}]: Error en on_message_received ('{topic}'): {type(e).__name__} - {e}. Payload: {payload_bytes[:200]}...")

    def _setup_job_and_load_data(self, command_data):
        """
        Configura el estado para un nuevo job basado en el comando del servidor
        y carga la partición de datos correspondiente.
        Inicia el tracker de emisiones para este job.
        Devuelve True si todo fue exitoso, False en caso contrario.
        """
        payload_sim_client_id_from_cmd = command_data.get("sim_client_id")
        self._reset_current_job(payload_sim_client_id_from_cmd) 
        
        log_id = self.current_job_data["sim_client_id"] # Ya establecido por _reset_current_job
        
        self.current_job_data["dataset_name"] = command_data.get("dataset_name")
        client_indices = command_data.get("data_indices")
        num_global_classes_from_server = command_data.get("num_global_classes")

        client_output_dir = os.path.join(PROJECT_ROOT, "emissions_output", self.sim_id) # Carpeta por cliente
        os.makedirs(client_output_dir, exist_ok=True)
        

        try:
            X_global, y_global = load_dataset(self.current_job_data["dataset_name"])
            self.current_job_data["X_client_partition"] = X_global[client_indices, :]
            self.current_job_data["y_client_partition"] = y_global[client_indices]

            #Recibe el numero de clases globales del servidor, sino lo consulta en el dataset
            if num_global_classes_from_server is not None and isinstance(num_global_classes_from_server, int) and num_global_classes_from_server > 0:
                self.current_job_data["num_classes"] = num_global_classes_from_server
            else:
                y_unique_labels_init = np.unique(self.current_job_data["y_client_partition"])
                self.current_job_data["num_classes"] = len(y_unique_labels_init) if y_unique_labels_init.size > 0 else 0
            
            print(f"CLIENT_APP [{log_id}]: Partición cargada. X_shape: {self.current_job_data['X_client_partition'].shape}, "
                  f"y_shape: {self.current_job_data['y_client_partition'].shape}, Num Clases (job): {self.current_job_data['num_classes']}")
            
            #Iniciar tracker despues de la carga de datos para no medir procesos artificiales para nuestro trabajo
            self.tracker = ClientEmissionsManager( # Aquí usas ClientEmissionsManager
            project_name=f"client_job_{log_id}_{self.current_job_data['dataset_name']}", # Nombre de proyecto más específico
            output_dir=client_output_dir,
            )
            self.tracker.start_tracking()
            print(f"CLIENT_APP [{log_id}]: EmissionsTracker iniciado para job '{log_id}'.")
            return True # Configuración y carga de datos exitosa

        except Exception as e_load:
            error_msg = f"Error cargando o particionando datos para el job: {type(e_load).__name__} - {e_load}"
            self.communicator.publish(STATUS_TOPIC, {"sim_client_id": log_id, "status": "ERROR", "error_message": error_msg}, qos=1)
            if self.tracker and self.tracker.is_tracking():
                self.tracker.stop_tracking() # Detener tracker si falla la carga
            return False

    def _calculate_and_send_local_extremes(self):
        """
        Calcula los mínimos y máximos locales de la partición de datos actual
        y los envía al servidor.
        """
        log_id = self.current_job_data.get("sim_client_id")

        try:
            local_mins = np.min(self.current_job_data["X_client_partition"], axis=0).tolist()
            local_maxs = np.max(self.current_job_data["X_client_partition"], axis=0).tolist()
            feature_min_max_list = [[float(min_val), float(max_val)] for min_val, max_val in zip(local_mins, local_maxs)]
            
            extremes_payload = {
                "sim_client_id": log_id, 
                "dataset_name": self.current_job_data["dataset_name"],
                "feature_min_max": feature_min_max_list
            }
            self.communicator.publish(LOCAL_EXTREMES_TOPIC, extremes_payload, qos=1)
            print(f"CLIENT_APP [{log_id}]: Mínimos/máximos locales enviados.")

        except Exception as e_extremes:
            error_msg = f"Error calculando o enviando extremos locales: {type(e_extremes).__name__} - {e_extremes}"
            self.communicator.publish(STATUS_TOPIC, {"sim_client_id": log_id, "status": "ERROR", "error_message": error_msg}, qos=1)
            if self.tracker and self.tracker.is_tracking():
                self.tracker.stop_tracking() # Detener tracker si falla la carga
                
    def process_initial_command(self, command_data):
        """
        Función que procesa la orden del servidor para iniciar la selección de características.
        Primero configura el job y carga los datos, luego calcula y envía los extremos locales.
        """
        action_from_command = command_data.get("action", "N/A") # Para el log
        current_sim_id_for_log = command_data.get("sim_client_id", self.sim_id) # Usar el ID del comando para el log si está disponible
        

        print(f"CLIENT_APP [{current_sim_id_for_log}]: Comando '{action_from_command}' recibido.")
        self.load_dts = Timer("load_dataset")
        self.load_dts.__enter__()
        if self._setup_job_and_load_data(command_data):
            log_id = self.current_job_data["sim_client_id"]
            self.communicator.publish(STATUS_TOPIC, {"sim_client_id": log_id, "status": "RECEIVED_COMMAND_AND_INDICES"}, qos=1)
            
            self.load_dts.__exit__(None, None, None)
            self.pre_timer = Timer("cli_preproc")
            self.pre_timer.__enter__()
            self._calculate_and_send_local_extremes()
        else:
            # El error ya fue logueado y el tracker (si se inició) manejado por _setup_job_and_load_data
            print(f"CLIENT_APP [{current_sim_id_for_log}]: Fallo en la configuración inicial del job. No se procede con el cálculo de extremos.")

    def process_global_disc_params(self, params_data):
        """
        Funcion que gestiona la recepcion de los min/max globales del servidor y lanza el calculo para las tablas de probabilidad de I(Xk, Y)
        """
        active_job_id = self.current_job_data.get("sim_client_id")

        print(f"[{active_job_id}]: Parámetros de discretización globales recibidos.")
        self.current_job_data["num_bins_global"] = params_data.get("num_bins")
        self.current_job_data["global_feature_ranges"] = params_data.get("global_feature_ranges")

        # PROCESO DE DISCRETIZACION
        try:
            
            if self.current_job_data["X_client_partition"] is not None and self.current_job_data["X_client_partition"].size > 0:
                self.current_job_data["X_client_discretized"] = discretize_equalwidth(
                    self.current_job_data["X_client_partition"],
                    bins=self.current_job_data["num_bins_global"],
                    feature_ranges=self.current_job_data["global_feature_ranges"]
                )
                print(f"[{active_job_id}]: Discretización local completada.")
                self.communicator.publish(STATUS_TOPIC, {"sim_client_id": active_job_id, "status": "GLOBAL_PARAMS_RECEIVED_ACK"}, qos=1)
                #Tras la discretizacion calculamos tablas de probabilidad
                self._proceed_to_calculate_and_send_initial_probabilities()
            else:
                error_msg = "No hay datos locales (X_client_partition) para discretizar o está vacío."
                self.communicator.publish(STATUS_TOPIC, {"sim_client_id": active_job_id, "status": "ERROR", "error_message": error_msg}, qos=1)
        except Exception as e_disc:
            error_msg = f"Error durante la discretización local: {type(e_disc).__name__} - {e_disc}"
            self.communicator.publish(STATUS_TOPIC, {"sim_client_id": active_job_id, "status": "ERROR", "error_message": error_msg}, qos=1)


    def _proceed_to_calculate_and_send_initial_probabilities(self):
        """
        Realiza el calculo inicial de las tablas de probabilidad, el paso MIM I(Xk, Y).
        """
        sim_client_id = self.current_job_data.get("sim_client_id")
        if not sim_client_id: return

        print(f"[{sim_client_id}]: Calculando y enviando P(Xi,Y) iniciales.")
        if self.current_job_data.get("X_client_discretized") is None or \
           self.current_job_data.get("y_client_partition") is None or \
           self.current_job_data.get("num_bins_global") is None or \
           self.current_job_data.get("num_classes") is None or \
           self.current_job_data.get("num_classes", 0) == 0: # Revisar num_classes > 0
            error_msg = "Datos del cliente no listos/válidos para calcular P(Xi,Y) (X_discretized, y_labels, num_bins, o num_classes faltan o son inválidos)."
            self.communicator.publish(STATUS_TOPIC, {"sim_client_id": sim_client_id, "status": "ERROR", "error_message": error_msg}, qos=1)
            return
        
        try:
            with Timer("compute_xy") as t_compute:
                if hasattr(self, "pre_timer"):
                    self.pre_timer.__exit__(None, None, None)
                    t_pre = self.pre_timer.elapsed
                else:
                    t_pre = 0.0
                if hasattr(self, "load_dts"):
                    t_load = self.load_dts.elapsed
                else:
                    t_load = 0.0
                local_prob_array_XY = calculate_local_prob_dist_array(
                    self.current_job_data["X_client_discretized"],
                    self.current_job_data["y_client_partition"],
                    self.current_job_data["num_bins_global"],
                    self.current_job_data["num_classes"] 
                )

            #Se comprime el resultado para aligerar la carga de la red y del broker MQTT
            with Timer("comm_xy") as t_comm:
                pickled_array = pickle.dumps(local_prob_array_XY)
                original_size = len(pickled_array)
                compressed_pickled_array = zlib.compress(pickled_array)
                compressed_size = len(compressed_pickled_array)
                print(f"CLIENT_APP [{sim_client_id}]: Tablas P(Xi,Y) - Tamaño Pickled: {original_size} B, Comprimido (zlib): {compressed_size} B (Reducción: {((original_size - compressed_size) / original_size) * 100:.2f}%)")

                
                b64_array_str = base64.b64encode(compressed_pickled_array).decode('utf-8')
                data_payload = {
                    "sim_client_id": sim_client_id,
                    "dataset_name": self.current_job_data["dataset_name"],
                    "prob_table_type": "XY", 
                    "pickled_data_base64": b64_array_str,
                    "is_compressed": True,
                    "num_samples": len(self.current_job_data["y_client_partition"])
                }
                self.communicator.publish(DATA_RESULTS_TOPIC, data_payload, qos=1)
                self.communicator.publish(STATUS_TOPIC, {"sim_client_id": sim_client_id, "status": "LOCAL_XY_PROB_DIST_PUBLISHED"}, qos=1)
            bench_payload = {
                "sim_client_id": sim_client_id,
                "phase": "XY",
                "compute_s": round(t_compute.elapsed, 6),
                "comm_s": round(t_comm.elapsed, 6),
                "pre_s": round(t_pre, 6),
                "load_s": round(t_load, 6),
            }
            self.communicator.publish(BENCH_TOPIC, bench_payload, qos=0)
        except Exception as e_prob:
            error_msg = f"Error calculando/enviando P(Xi,Y) iniciales: {type(e_prob).__name__} - {e_prob}"
            self.communicator.publish(STATUS_TOPIC, {"sim_client_id": sim_client_id, "status": "ERROR", "error_message": error_msg}, qos=1)
        

    def process_jmi_batch_triplet_request(self, command_data):
        """
        Recibiendo la última caracteristica seleccionada y las candidatas, procesa un batch del algoritmo JMI I((Xk,Xj), Y) envia los resultados al servidor
        """
        active_job_id = self.current_job_data.get("sim_client_id")

        request_id = command_data.get("request_id")
        fixed_selected_idx = command_data.get("fixed_selected_feature_idx")
        candidate_k_indices = command_data.get("candidate_features_indices")

        self.current_job_data["current_jmi_request_id"] = request_id
    
        #Si no se reciben los datos necesarios se reporta un error
        if self.current_job_data.get("X_client_discretized") is None or \
           self.current_job_data.get("y_client_partition") is None or \
           self.current_job_data.get("num_bins_global") is None or \
           self.current_job_data.get("num_classes", 0) == 0: 
            error_msg = f"Datos del cliente no listos/válidos para JMI Batch (ID: {request_id})."
            self.communicator.publish(STATUS_TOPIC, {"sim_client_id": active_job_id, "status": "ERROR", "error_message": error_msg, "request_id": request_id}, qos=1)
            return

        with Timer("compute_jmi_batch") as t_compute:
            batch_triplet_results = [] # Lista para almacenar todas las tripletas del batch
            for k_idx in candidate_k_indices:
                try:
                    p_xyz_table = calculate_local_triplet_prob_dist(
                        self.current_job_data["X_client_discretized"],
                        self.current_job_data["y_client_partition"],
                        k_idx, fixed_selected_idx,
                        self.current_job_data["num_bins_global"],
                        self.current_job_data["num_classes"] # Usar num_classes del job
                    )
                    if p_xyz_table.size == 0 and self.current_job_data["num_bins_global"] > 0:
                        print(f"[{active_job_id}]: Cálculo de P(Xk,Xj,Y) para par ({k_idx},{fixed_selected_idx}) en batch {request_id} resultó vacío. Omitiendo este par.")
                        continue
                    batch_triplet_results.append({
                        "feature_pair": [k_idx, fixed_selected_idx],
                        "triplet_table": p_xyz_table
                    })
                except Exception as e_jmi_batch_pair:
                    print(f"[{active_job_id}]: Error calculando P(Xk,Xj,Y) para par ({k_idx},{fixed_selected_idx}) en batch {request_id}: {e_jmi_batch_pair}. Omitiendo.")
                    continue
        with Timer("comm_jmi_batch") as t_comm:
            if batch_triplet_results:
                try:
                    #Se comprime el resultado para aligerar la carga de la red y del broker MQTT
                    pickled_batch_data = pickle.dumps(batch_triplet_results)
                    original_size = len(pickled_batch_data)
                    compressed_pickled_data = zlib.compress(pickled_batch_data)
                    compressed_size = len(compressed_pickled_data)
                    print(f"CLIENT_APP [{active_job_id}]: Lote JMI (req: {request_id}) - Tamaño Pickled: {original_size} B, Comprimido (zlib): {compressed_size} B (Reducción: {((original_size - compressed_size) / original_size) * 100:.2f}%)")

                    b64_batch_data_str = base64.b64encode(compressed_pickled_data).decode('utf-8')
                    jmi_batch_payload = {
                        "sim_client_id": active_job_id, 
                        "request_id": request_id,
                        "prob_table_type": "XkXjY_BATCH", 
                        "pickled_batch_data_base64": b64_batch_data_str,
                        "num_triplets_in_payload": len(batch_triplet_results), 
                        "is_compressed": True,
                        "num_samples_client": len(self.current_job_data["y_client_partition"])
                    }
                    self.communicator.publish(JMI_PAIR_PROB_RESULTS_TOPIC, jmi_batch_payload, qos=1)

                except Exception as e_batch_send:
                    error_msg = f"Error serializando/enviando LOTE de tripletas JMI (ID: {request_id}): {type(e_batch_send).__name__} - {e_batch_send}"
                    self.communicator.publish(STATUS_TOPIC, {"sim_client_id": active_job_id, "status": "ERROR", "error_message": error_msg, "request_id": request_id}, qos=1)
                    return
            else:
                print(f"[{active_job_id}]: No se generaron tablas de tripletas válidas en el batch {request_id} para enviar.")
        try:
            status_payload_jmi_batch_done = {
                "sim_client_id": active_job_id, 
                "status": "JMI_BATCH_TRIPLETS_PUBLISHED",
                "request_id": request_id,
                "num_triplets_expected_in_batch_from_client": len(batch_triplet_results)
            }
            self.communicator.publish(STATUS_TOPIC, status_payload_jmi_batch_done, qos=1)
            bench_payload = {
                "sim_client_id": active_job_id,
                "phase": f"JMI_batch_{request_id}",
                "compute_s": round(t_compute.elapsed, 6),
                "comm_s": round(t_comm.elapsed, 6),
            }
            self.communicator.publish(BENCH_TOPIC, bench_payload, qos=0)
        except Exception as e_bench_send:
            print("No se enviaron las medidas de tiempos")
            
        print(f"[{active_job_id}]: Finalizado envío de LOTE JMI (ID: {request_id}). {len(batch_triplet_results)} tablas en lote.")

    def stop_save_and_send_emissions(self, emissions_topic):
            """
            Para el tracker de emisiones, obtiene los datos y los envía a un topic MQTT.
            """
            # Usamos el sim_id del job actual si está disponible, sino el sim_id de la instancia
            log_id = self.current_job_data.get("sim_client_id") if self.current_job_data and self.current_job_data.get("sim_client_id") else self.sim_id
            
            emissions_data_dict = self.tracker.stop_tracking_and_get_data(log_id)

            #Crear el payload
            payload = emissions_data_dict
            #Publicar el payload a un topic MQTT
            msg_info = self.communicator.publish(emissions_topic, payload, qos=1)

            if msg_info and msg_info.rc == 0: # 0 es MQTT_ERR_SUCCESS
                print(f"CLIENT_APP [{log_id}]: Datos de emisiones enviados exitosamente al topic '{emissions_topic}'.")
            else:
                rc_value = msg_info.rc if msg_info else "N/A (msg_info es None)"
                print(f"CLIENT_APP [{log_id}]: Fallo al enviar datos de emisiones al topic '{emissions_topic}'. RC: {rc_value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cliente para selección de características federada.")
    parser.add_argument("--sim-id", type=str, required=True, help="ID único para esta instancia de cliente (ej: sim_client_iid_0_dataset).")

    # Opcionalmente podemos agregar el broker y el puerto por parametro
    # parser.add_argument("--broker", type=str, default=DEFAULT_BROKER_ADDRESS, help="Dirección del broker MQTT.")
    # parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Puerto del broker MQTT.")
    args = parser.parse_args()

    client = ClientApp(sim_id=args.sim_id) # , broker_address=args.broker, port=args.port)
    client.start()