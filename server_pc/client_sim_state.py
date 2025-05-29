# --- Estado del Cliente y Globales de Ronda ---
class ClientSimState:
    """Almacena el estado de un cliente simulado durante una ronda."""
    def __init__(self, client_id, dataset_name_assigned=""):
        self.client_id = client_id
        self.dataset_name = dataset_name_assigned
        self.command_acked = False
        self.local_extremes_received = False
        self.feature_min_max_local = None
        self.global_params_sent_to_client = False
        self.global_params_acked_by_client = False
        self.local_XY_prob_dist_published = False
        self.local_XY_prob_dist_received = False
        self.num_local_samples = None
        self.local_XY_prob_dist_array = None
        self.error_message = None