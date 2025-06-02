import sys
import numpy as np
import threading
import os

try:
    from feature_selector import calculate_mi_for_feature, calculate_mi_for_triplet
except ImportError as e:
    print(f"ERROR crítico importando módulos de cálculo: {e}. Verifique PYTHONPATH.")
    sys.exit(1)

COMMAND_TOPIC = "tfg/fl/pi/command" 
TIMEOUT_JMI_BATCH_RECEPTION_SECONDS = 300.0 

class JMIOrchestrator:
    def __init__(self, comm_instance, active_clients_dict, num_expected_clients, global_lock, aggregate_func_ref,dataset_name_for_save,
                 save_function_ref):
        self.comm_instance = comm_instance
        self.active_clients_dict = active_clients_dict 
        self.num_expected_clients = num_expected_clients # Número de clientes que se espera que participen activamente
        self.lock = global_lock # Bloqueo para concurrencia
        self.aggregate_func = aggregate_func_ref #Funcion de agregación

        self.S_selected_indices = [] #Conjunto de caracteristicas seleccionadas por indice
        self.Q_candidate_indices = [] #Conjunto de caracteristicas candidatas por indice
        self.aggregated_XY_initial_tables = None #Tablas de probabilidad entre las caracteristicas y las clases 
        self.current_iter_candidate_scores = {}

        #Datos para almacenar las caracteristicas en un .txt
        self.save_features_func = save_function_ref #Funcion de guardado
        self.dataset_name_for_save = dataset_name_for_save #Nombre del dataset evaluado

        #Almacenes temporales de tablas de probabilidad
        self.global_triplet_scores_cache = {}
        self.current_iter_triplet_tables_from_clients = {}
        self.clients_reported_current_batch_status = {}

        self.all_clients_reported_batch_event = threading.Event() #Mantiene en espera el calculo de nuevas caracteristicas a seleccionar mientras los clientes no pasan las tablas necesarias       
        self.current_batch_request_id = None

    def _initialize_first_feature(self):
        """
            Inicia el algoritmo JMI seleccionando la primera caracteristica con MIM
        """
        if self.aggregated_XY_initial_tables is None or self.aggregated_XY_initial_tables.shape[0] == 0:
            print("Tablas P(Xi,Y) iniciales no disponibles o vacías.")
            return False
        
        num_total_features = self.aggregated_XY_initial_tables.shape[0]
        initial_mi_scores = np.zeros(num_total_features)
        for i in range(num_total_features):
            try:
                initial_mi_scores[i] = calculate_mi_for_feature(self.aggregated_XY_initial_tables[i, :, :])
            except Exception: # Captura genérica para errores de cálculo
                initial_mi_scores[i] = -float('inf') 

        if not np.any(initial_mi_scores > 1e-12): # Si ningún score es significativamente positivo
            print("ERROR: No se calcularon scores MI iniciales válidos.")
            return False

        first_selected_idx = int(np.argmax(initial_mi_scores))
        self.S_selected_indices.append(first_selected_idx)
        if first_selected_idx in self.Q_candidate_indices:
            self.Q_candidate_indices.remove(first_selected_idx)
        print(f"Primera característica seleccionada (MIM): {first_selected_idx} (MI: {initial_mi_scores[first_selected_idx]:.4f})")
        return True

    def start_selection(self, aggregated_XY_tables, top_k_to_select):
        """
            Función que gestiona de forma completa el loop JMI hasta tener el conjunto de caracteristicas. En cada iteración realiza las peticiones al cliente, espera a recibir el resultado de todos ellos para seleccionar una nueva caracteristica
        """
        print("Iniciando JMI.")
        self.aggregated_XY_initial_tables = aggregated_XY_tables
        
        num_total_features = self.aggregated_XY_initial_tables.shape[0]
        with self.lock: # Proteger inicialización de listas y caché
            self.Q_candidate_indices = list(range(num_total_features))
            self.S_selected_indices = [] 
            self.global_triplet_scores_cache.clear()

        self._initialize_first_feature()


        while len(self.S_selected_indices) < top_k_to_select and self.Q_candidate_indices:
            iter_num_for_log = len(self.S_selected_indices)
            print(f"Seleccionando característica JMI #{iter_num_for_log + 1}.")
            
            X_last_selected_idx = -1
            current_candidate_indices_for_iter = []
            # Determinar clientes válidos para esta ronda de batch
            clients_to_command_this_round = [
                cid for cid, state in self.active_clients_dict.items() 
                if state and not state.error_message # Solo clientes activos y sin error previo
            ]
            
            with self.lock:
                if not self.S_selected_indices: break
                X_last_selected_idx = int(self.S_selected_indices[-1])
                current_candidate_indices_for_iter = list(self.Q_candidate_indices)
                
                self.current_iter_candidate_scores.clear()
                self.current_iter_triplet_tables_from_clients.clear()
                # Aquí se inicializa clients_reported_current_batch_status
                self.clients_reported_current_batch_status = {cid: False for cid in clients_to_command_this_round}
                self.all_clients_reported_batch_event.clear()
                self.current_batch_request_id = f"jmi_iter_{iter_num_for_log}_lastsel_{X_last_selected_idx}_cand_{len(current_candidate_indices_for_iter)}"

            if not current_candidate_indices_for_iter: break # No más candidatas

            # --- Envío de Solicitud de Batch ---
            payload_base_to_clients = {
                "action": "REQUEST_JMI_BATCH_TRIPLET_PROBS",
                "request_id": self.current_batch_request_id,
                "fixed_selected_feature_idx": X_last_selected_idx,
                "candidate_features_indices": [int(k) for k in current_candidate_indices_for_iter]
            }
            
            num_clients_commanded_this_batch = 0
            for client_id_target in clients_to_command_this_round:
                if not self.comm_instance: continue # Guarda contra comunicador no disponible
                payload_for_client = {**payload_base_to_clients, "sim_client_id": client_id_target}
                msg_info = self.comm_instance.publish(COMMAND_TOPIC, payload_for_client, qos=1)
                if msg_info and msg_info.rc == 0: num_clients_commanded_this_batch +=1
            

            # --- Espera del Batch ---
            print(f"Esperando {TIMEOUT_JMI_BATCH_RECEPTION_SECONDS}s para batches de {len(clients_to_command_this_round)} clientes.")
            self.all_clients_reported_batch_event.wait(timeout=TIMEOUT_JMI_BATCH_RECEPTION_SECONDS)

            # --- Agregación y Cacheo de Scores del Batch Actual ---
            num_new_scores_cached = 0
            with self.lock: 
                for pair_key, data_for_pair in list(self.current_iter_triplet_tables_from_clients.items()):
                    if pair_key in self.global_triplet_scores_cache and self.global_triplet_scores_cache[pair_key] > -float('inf'):
                        continue # Ya calculado y cacheado previamente
                    
                    # Usar el número de clientes a los que se les envió comando para este batch
                    if data_for_pair['received_from_clients_count'] >= len(clients_to_command_this_round):
                        if data_for_pair.get('aggregated_table') is None: 
                            temp_dict_for_agg = {i: tbl for i, tbl in enumerate(data_for_pair['tables_list'])}
                            aggregated_table, _, _, _ = self.aggregate_func(temp_dict_for_agg, expected_shape_dims=3)
                        else: 
                            aggregated_table = data_for_pair['aggregated_table']

                        if aggregated_table is not None:
                            self.global_triplet_scores_cache[pair_key] = calculate_mi_for_triplet(aggregated_table)
                            num_new_scores_cached += 1
                        else: 
                            self.global_triplet_scores_cache[pair_key] = -float('inf')
                    else: 
                        self.global_triplet_scores_cache[pair_key] = -float('inf')
            if num_new_scores_cached > 0:
                print(f"{num_new_scores_cached} nuevos scores de tripletas (con X_last_selected={X_last_selected_idx}) cacheados.")

            # --- Cálculo de Scores JMI Totales ---
            for k_cand_idx in current_candidate_indices_for_iter:
                current_Jk_score = 0.0
                all_terms_valid = True
                for j_sel_idx in self.S_selected_indices:
                    pair_key_lookup = tuple(sorted((k_cand_idx, j_sel_idx)))
                    with self.lock: 
                        term_score = self.global_triplet_scores_cache.get(pair_key_lookup, -float('inf'))
                    
                    if term_score > -float('inf'):
                        current_Jk_score += term_score
                    else: # Si algún término falta o es inválido, el score JMI total para esta k_cand es inválido
                        all_terms_valid = False
                        break 
                with self.lock: 
                    self.current_iter_candidate_scores[k_cand_idx] = current_Jk_score if all_terms_valid else -float('inf')
            
            # --- Selección de la Mejor Característica ---
            best_k_for_this_iter = -1
            with self.lock:
                
                valid_scores = {k: s for k, s in self.current_iter_candidate_scores.items() if s > -float('inf')}
                if not valid_scores:
                    print("No hay candidatas con scores JMI válidos. Abortando."); break
                
                best_k_for_this_iter = max(valid_scores, key=valid_scores.get)
                best_score_val = valid_scores[best_k_for_this_iter]

            if best_k_for_this_iter != -1 :
                with self.lock: # Proteger modificación de S y Q
                    if best_k_for_this_iter in self.Q_candidate_indices: 
                        self.S_selected_indices.append(best_k_for_this_iter)
                        self.Q_candidate_indices.remove(best_k_for_this_iter)
                        print(f"Característica JMI #{len(self.S_selected_indices)} seleccionada: {best_k_for_this_iter} (Score JMI: {best_score_val:.4f})")
                    else: # Error de lógica si la mejor candidata ya no está en Q
                        print(f"Candidata {best_k_for_this_iter} no en Q. Abortando."); break 
            else: # No se pudo seleccionar ninguna (ej. todos los scores JMI fueron -inf)
                print(f"No se pudo seleccionar nueva característica. Finalizando JMI."); break
            
        print("Proceso JMI finalizado.")
        with self.lock: 
            final_selection = list(self.S_selected_indices) 
        print(f"Características seleccionadas ({len(final_selection)}): {final_selection}")
        if final_selection:
            if self.save_features_func:
                self.save_features_func(
                    final_selection,
                    self.dataset_name_for_save,
                    len(final_selection), # K real
                    "JMI_federated"
                )
            else: 
                 print("No se proporcionó función de guardado a JMIOrchestrator.")
        return final_selection
