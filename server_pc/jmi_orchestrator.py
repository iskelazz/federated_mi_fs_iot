import sys
import numpy as np
import threading
from timer import Timer

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
        self.num_expected_clients = num_expected_clients
        self.lock = global_lock 
        self.aggregate_func = aggregate_func_ref
        
        self.total_agg_time = 0.0       # Agregación de tablas
        self.total_mi_calc_time = 0.0   # Cálculo de MI con las tablas agregadas

        self.S_selected_indices = [] 
        self.Q_candidate_indices = [] 
        self.aggregated_XY_initial_tables = None 
        self.current_iter_candidate_scores = {}
        self.triplet_cache = dict()  

        self.save_features_func = save_function_ref 
        self.dataset_name_for_save = dataset_name_for_save 

        self.current_iter_triplet_tables_from_clients = {}
        self.clients_reported_current_batch_status = {}

        self.all_clients_reported_batch_event = threading.Event()     
        self.current_batch_request_id = None

    def _initialize_first_feature(self):
        if self.aggregated_XY_initial_tables is None or self.aggregated_XY_initial_tables.shape[0] == 0:
            print("Tablas P(Xi,Y) iniciales no disponibles o vacías.")
            return False

        num_total_features = self.aggregated_XY_initial_tables.shape[0]
        initial_mi_scores = np.zeros(num_total_features)
        for i in range(num_total_features):
            try:
                with Timer("mim") as t_mi:
                    initial_mi_scores[i] = calculate_mi_for_feature(self.aggregated_XY_initial_tables[i, :, :])
                self.total_mi_calc_time += t_mi.elapsed
            except Exception:
                initial_mi_scores[i] = -float('inf')

        if not np.any(initial_mi_scores > 1e-12):
            print("ERROR: No se calcularon scores MI iniciales válidos.")
            return False

        first_selected_idx = int(np.argmax(initial_mi_scores))
        self.S_selected_indices.append(first_selected_idx)
        if first_selected_idx in self.Q_candidate_indices:
            self.Q_candidate_indices.remove(first_selected_idx)
        print(f"Primera característica seleccionada (MIM): {first_selected_idx} (MI: {initial_mi_scores[first_selected_idx]:.4f})")
        return True

    def start_selection(self, aggregated_XY_tables, top_k_to_select):
        print("Iniciando JMI.")
        self.aggregated_XY_initial_tables = aggregated_XY_tables

        # --- Inicialización de estructuras ---
        num_total_features = self.aggregated_XY_initial_tables.shape[0]
        with self.lock:
            self.Q_candidate_indices = list(range(num_total_features))
            self.S_selected_indices = [] 
            self.triplet_cache.clear()
        with self.lock:
            self._initialize_first_feature()

        while len(self.S_selected_indices) < top_k_to_select and self.Q_candidate_indices:
            iter_num_for_log = len(self.S_selected_indices)
            print(f"Seleccionando característica JMI #{iter_num_for_log + 1}.")

            # --- Preparación de la iteración ---
            X_last_selected_idx = -1
            current_candidate_indices_for_iter = []
            clients_to_command_this_round = [
                cid for cid, state in self.active_clients_dict.items() 
                if state and not state.error_message
            ]

            with self.lock:
                if not self.S_selected_indices: break
                X_last_selected_idx = int(self.S_selected_indices[-1])
                current_candidate_indices_for_iter = list(self.Q_candidate_indices)

                filtered_candidates = []
                for fk in current_candidate_indices_for_iter:
                    key = tuple(sorted((fk, X_last_selected_idx)))
                    if key not in self.triplet_cache:
                        filtered_candidates.append(fk)
                current_candidate_indices_for_iter = filtered_candidates

                if not current_candidate_indices_for_iter:
                    print(f"[JMI] Todos los pares (Xk={fk}, Xj={X_last_selected_idx}) ya estaban en cache. Iteración {iter_num_for_log + 1} se salta envío.")
                    continue

                self.current_iter_candidate_scores.clear()
                self.current_iter_triplet_tables_from_clients.clear()
                self.clients_reported_current_batch_status = {cid: False for cid in clients_to_command_this_round}
                self.all_clients_reported_batch_event.clear()
                self.current_batch_request_id = f"jmi_iter_{iter_num_for_log}_lastsel_{X_last_selected_idx}_cand_{len(current_candidate_indices_for_iter)}"

            if not current_candidate_indices_for_iter: break

            # --- Envío de solicitud de cálculo de tripletas ---
            payload_base_to_clients = {
                "action": "REQUEST_JMI_BATCH_TRIPLET_PROBS",
                "request_id": self.current_batch_request_id,
                "fixed_selected_feature_idx": X_last_selected_idx,
                "candidate_features_indices": [int(k) for k in current_candidate_indices_for_iter]
            }

            for client_id_target in clients_to_command_this_round:
                if not self.comm_instance: continue
                payload_for_client = {**payload_base_to_clients, "sim_client_id": client_id_target}
                self.comm_instance.publish(COMMAND_TOPIC, payload_for_client, qos=1)

            print(f"Esperando {TIMEOUT_JMI_BATCH_RECEPTION_SECONDS}s para batches de {len(clients_to_command_this_round)} clientes.")
            self.all_clients_reported_batch_event.wait(timeout=TIMEOUT_JMI_BATCH_RECEPTION_SECONDS)

            # --- Agregación y almacenamiento de MI en la caché ---
            with self.lock: 
                for pair_key, data_for_pair in list(self.current_iter_triplet_tables_from_clients.items()):
                    if pair_key in self.triplet_cache and self.triplet_cache[pair_key].get("mi", -float('inf')) > -float('inf'):
                        continue

                    if data_for_pair['received_from_clients_count'] >= len(clients_to_command_this_round):
                        if data_for_pair.get('aggregated_table') is None:
                            temp_dict_for_agg = {i: tbl for i, tbl in enumerate(data_for_pair['tables_list'])}
                            with Timer("agg") as t_agg:
                                aggregated_table, _, _, _ = self.aggregate_func(temp_dict_for_agg, expected_shape_dims=3)
                            self.total_agg_time += t_agg.elapsed
                        else:
                            aggregated_table = data_for_pair['aggregated_table']

                        if aggregated_table is not None:
                            with Timer("mi") as t_mi:
                                mi_val = calculate_mi_for_triplet(aggregated_table)
                            self.total_mi_calc_time += t_mi.elapsed
                            self.triplet_cache[pair_key] = {"mi": mi_val}  # Solo guardamos el score
                        else:
                            self.triplet_cache[pair_key] = {"mi": -float('inf')}
                    else:
                        self.triplet_cache[pair_key] = {"mi": -float('inf')}

            # --- Construcción de matriz de scores y vectorización ---
            score_matrix = np.full((len(current_candidate_indices_for_iter), len(self.S_selected_indices)), -np.inf)
            k_index_map = {k: i for i, k in enumerate(current_candidate_indices_for_iter)}
            j_index_map = {j: i for i, j in enumerate(self.S_selected_indices)}

            for k_cand_idx in current_candidate_indices_for_iter:
                for j_sel_idx in self.S_selected_indices:
                    pair_key_lookup = tuple(sorted((k_cand_idx, j_sel_idx)))
                    term_score = self.triplet_cache.get(pair_key_lookup, {}).get("mi", -float('inf'))
                    score_matrix[k_index_map[k_cand_idx], j_index_map[j_sel_idx]] = term_score

            summed_scores = np.where(score_matrix > -np.inf, score_matrix, 0.0).sum(axis=1)
            valid_mask = (score_matrix > -np.inf).all(axis=1)

            for i, k_cand_idx in enumerate(current_candidate_indices_for_iter):
                score = summed_scores[i] if valid_mask[i] else -float('inf')
                self.current_iter_candidate_scores[k_cand_idx] = score

            # --- Selección de la mejor característica ---
            best_k_for_this_iter = -1
            with self.lock:
                valid_scores = {k: s for k, s in self.current_iter_candidate_scores.items() if s > -float('inf')}
                if not valid_scores:
                    print("No hay candidatas con scores JMI válidos. Abortando."); break

                best_k_for_this_iter = max(valid_scores, key=valid_scores.get)
                best_score_val = valid_scores[best_k_for_this_iter]

            if best_k_for_this_iter != -1:
                with self.lock:
                    if best_k_for_this_iter in self.Q_candidate_indices:
                        self.S_selected_indices.append(best_k_for_this_iter)
                        self.Q_candidate_indices.remove(best_k_for_this_iter)
                        print(f"Característica JMI #{len(self.S_selected_indices)} seleccionada: {best_k_for_this_iter} (Score JMI: {best_score_val:.4f})")
                    else:
                        print(f"Candidata {best_k_for_this_iter} no en Q. Abortando."); break
            else:
                print(f"No se pudo seleccionar nueva característica. Finalizando JMI."); break

        # --- Finalización del proceso ---
        print("Proceso JMI finalizado.")
        with self.lock:
            final_selection = list(self.S_selected_indices)
        print(f"Características seleccionadas ({len(final_selection)}): {final_selection}")
        if final_selection:
            if self.save_features_func:
                self.save_features_func(
                    final_selection,
                    self.dataset_name_for_save,
                    len(final_selection),
                    "JMI_federated"
                )
            else:
                print("No se proporcionó función de guardado a JMIOrchestrator.")
        return final_selection


    def get_server_timing_summary(self):
        return {
            "T_agg_total": round(self.total_agg_time, 3),
            "T_mi_calc_total": round(self.total_mi_calc_time, 3)
        }