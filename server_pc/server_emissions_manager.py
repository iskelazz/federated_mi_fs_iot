import os
import sys
from typing import Optional
from codecarbon import EmissionsTracker
SCRIPT_DIR_APP = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_APP = os.path.dirname(SCRIPT_DIR_APP) 

if PROJECT_ROOT_APP not in sys.path: sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR_APP, '..'))) # Ir un nivel arriba para el proyecto raíz
try:
    from mqtt_handlers.mqtt_communicator import MQTTCommunicator
 
except ImportError as e:
    print(f"ERROR crítico importando módulos: {e}. Verifique PYTHONPATH.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

#Clase que maneja las emisiones en el servidor    
class ServerEmissionsManager:
    def __init__(self, project_root_path, server_id_for_log = "server"):
        emissions_dir = os.path.join(project_root_path, "emissions_output")
        
        try:
            os.makedirs(emissions_dir, exist_ok=True)
            print(f"ServerEmissionsManager: Ensured 'emissions_output' directory exists at: {emissions_dir}")
        except OSError as e:
            print(f"ServerEmissionsManager: CRITICAL ERROR creating 'emissions_output' directory at {emissions_dir}: {e}")
        self.server_tracker = EmissionsTracker(
            project_name=f"server_{server_id_for_log}", 
            output_dir=os.path.join(project_root_path, "emissions_output")
        )
        self.collected_client_energy = {}
        self.client_reports_expected = 0
        self.client_reports_received = 0
        
        self.communicator: Optional[MQTTCommunicator] = None
        self.jmi_lock = None # O un lock específico para emisiones
        self.active_sim_clients_ref = None # Referencia al dict de ServerLogic

    def set_dependencies(self, communicator, lock, active_clients_ref):
        """
        Inyecta las dependencias necesarias para manejar las comunicaciones entre cliente y servidor.
        """
        self.communicator = communicator
        self.jmi_lock = lock 
        self.active_sim_clients_ref = active_clients_ref

    def start_server_tracking(self):
        """
        Inicia la toma de mediciones de consumo y emisiones.
        """
        self.server_tracker.start()
        
    def reset_server_tracking(self):
        """
        Devuelve el tracker a su estado inicial.
        """
        self.collected_client_energy.clear()
        self.clients_reported_emissions_count = 0
        self.num_clients_commanded_for_emissions = 0
       
    def _handle_server_emissions_data(self):
        """
        Para el tracker y devuelve los datos de consumo obtenidos del servidor.
        """
        self.server_tracker.stop()
        # 2. Acceder a los datos finales del tracker
        if hasattr(self.server_tracker, 'final_emissions_data') and self.server_tracker.final_emissions_data is not None:
            emissions_details = self.server_tracker.final_emissions_data
        co2_emissions_value_kg = emissions_details.emissions # en kg
        energy_consumed_value_kwh = emissions_details.energy_consumed # en kWh
        print(f"---------------------------------------------------------------------")
        print(f"SERVIDOR: Datos detallados de CodeCarbon -> "
                f"CO2: {co2_emissions_value_kg:.6f} kg, "
                f"Energía Consumida: {energy_consumed_value_kwh:.6f} kWh, s")
        print(f"---------------------------------------------------------------------")
        return emissions_details

    def prepare_for_new_client_emissions_round(self, num_clients_to_command):
        """
            Establece el número de clientes para una ronda de medición de emisiones
        """
        with self.jmi_lock: # Asumiendo que usamos el lock global
            self.collected_client_energy.clear()
            self.client_reports_received = 0
            self.client_reports_expected = num_clients_to_command


    def request_emissions_from_clients(self, client_id_target = None):
        """
            Solicita a los clientes los datos de emisiones a traves de un topic del protocolo de comunicación MQTT
        """
        if not self.communicator or self.active_sim_clients_ref is None:
            print("SERVER_EMISSIONS_MANAGER: Comunicador o referencia a clientes activos no disponible.")
            return

        clients_to_command = []
        # Acceder a active_sim_clients_ref bajo el lock si es compartido con ServerLogic
        with self.jmi_lock:
            if client_id_target:
                if client_id_target in self.active_sim_clients_ref and \
                   self.active_sim_clients_ref[client_id_target] and \
                   not self.active_sim_clients_ref[client_id_target].error_message:
                    clients_to_command.append(client_id_target)
                else:
                    self.prepare_for_new_client_emissions_round(0) # No se comandará a nadie
                    return
            else:
                clients_to_command = [cid for cid, state in self.active_sim_clients_ref.items() if state and not state.error_message]
        
        self.prepare_for_new_client_emissions_round(len(clients_to_command))

        print(f"SERVER_EMISSIONS_MANAGER: Solicitando datos de emisiones a {len(clients_to_command)} cliente(s).")
        for client_id in clients_to_command:
            payload = {"action": "SEND_EMISSIONS_DATA", "sim_client_id": client_id}
            # COMMAND_TOPIC necesitaría ser una constante accesible aquí o pasada
            msg_info = self.communicator.publish("tfg/fl/pi/command", payload, qos=1) 
            if not (msg_info and msg_info.rc == 0): # Asumiendo MQTT_ERR_SUCCESS es 0
                print(f"SERVER_EMISSIONS_MANAGER: ERROR solicitando emisiones a {client_id}.")
            else:
                print(f"SERVER_EMISSIONS_MANAGER: Solicitud de emisiones enviada a {client_id}.")


    def process_client_emission_report(self, data_payload):
        client_id = data_payload.get("sim_client_id")
        energy_kwh = data_payload.get("energy_consumed_kwh")
        co2_kg = data_payload.get("co2_emissions_kg") 
        if isinstance(energy_kwh, (float, int)) and client_id:
            with self.jmi_lock: # Proteger acceso
                first_report_from_this_client_this_round = client_id not in self.collected_client_energy 
                self.collected_client_energy[client_id] = (energy_kwh, co2_kg)
                if first_report_from_this_client_this_round:
                    self.client_reports_received += 1                
            self.check_and_print_aggregated_emissions()
            
        
    def check_and_print_aggregated_emissions(self):
        with self.jmi_lock: # Asegurar consistencia al leer contadores y datos
            if self.client_reports_expected > 0 and self.client_reports_received >= self.client_reports_expected:
                print(f"SERVER_EMISSIONS_MANAGER: Todos los {self.client_reports_received}/{self.client_reports_expected} clientes han reportado emisiones.")
                
                server_emissions_details = self._handle_server_emissions_data()
                server_energy_kwh = server_emissions_details.energy_consumed if server_emissions_details else 0.0
                server_co2_kg = server_emissions_details.emissions if server_emissions_details else 0.0

                total_client_energy_kwh = 0.0
                total_client_co2_kg = 0.0
                valid_client_reports_for_sum = 0
                for energy_val, co2_val in self.collected_client_energy.values():
                    total_client_energy_kwh += energy_val
                    total_client_co2_kg += co2_val
                    valid_client_reports_for_sum += 1
                
                grand_total_energy = total_client_energy_kwh + server_energy_kwh
                grand_total_emissions = total_client_co2_kg + server_co2_kg

                print(f"---------------------------------------------------------------------")
                print(f"SERVER_EMISSIONS_MANAGER: CONSUMO ENERGÉTICO AGREGADO TOTAL:")
                print(f"    Energía Servidor: {server_energy_kwh:.6f} kWh")
                print(f"    Energía Clientes (agregada): {total_client_energy_kwh:.6f} kWh (de {valid_client_reports_for_sum} reportes)")
                print(f"    TOTAL GENERAL: {grand_total_energy:.6f} kWh")
                print(f"---------------------------------------------------------------------")
                print(f"---------------------------------------------------------------------")
                print(f"SERVER_EMISSIONS_MANAGER: EMISIONES CO2 AGREGADO TOTAL:")
                print(f"    Emisiones Servidor: {server_co2_kg:.6f} kg")
                print(f"    Emisiones Clientes (agregada): {total_client_co2_kg:.6f} kg (de {valid_client_reports_for_sum} reportes)")
                print(f"    TOTAL GENERAL: {grand_total_emissions:.6f} kg")
                print(f"---------------------------------------------------------------------")
                
                # Reset para la próxima posible ronda de solicitud de emisiones
                self.client_reports_expected = 0 
                self.client_reports_received = 0
                self.collected_client_energy.clear()