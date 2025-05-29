# Nombre de archivo: client_emissions_manager.py
from codecarbon import EmissionsTracker




class ClientEmissionsManager:
    def __init__(self, project_name, output_dir):
        """
        Inicializa el gestor de emisiones para un cliente.
        """
        self.project_name = project_name
        self.output_dir = output_dir
        self.tracker: EmissionsTracker = EmissionsTracker(
            project_name=self.project_name,
            output_dir=self.output_dir,
            force_cpu_power=11, #Estimacion basada en los 12W de consumo de la rasp pi, repartidas en 11W CPU
            force_ram_power=1   # Y 1W la ram. Quitar si se quiere simular el consumo del cliente en otro equipo
        )
        self._is_tracking = False
        print(f"Cliente [{self.project_name}]: Instanciado.")

    def start_tracking(self):
        """Inicia el seguimiento de emisiones para el job actual."""
        if not self._is_tracking:
            try:
                self.tracker.start()
                self._is_tracking = True
                print(f"[{self.project_name}]: Tracker iniciado.")
            except Exception as e:
                print(f"[{self.project_name}]: Error al iniciar tracker: {e}")
        else:
            print(f"[{self.project_name}]: Tracker ya estaba iniciado.")

    def stop_tracking_and_get_data(self, log_id):
        """
        Detiene el seguimiento de emisiones y devuelve los datos recolectados.
        """        
        emissions_data_dict = None
        try:
            # stop() finaliza la medición. La información detallada está en final_emissions_data.
            self.stop_tracking()

            details = self.tracker.final_emissions_data
            emissions_data_dict = {
                "sim_client_id": log_id,
                "co2_emissions_kg": details.emissions,
                "energy_consumed_kwh": details.energy_consumed,
                "duration_seconds": details.duration
            }
            print(f"Cliente [{self.project_name}]: Datos finales -> "
                f"CO2: {emissions_data_dict['co2_emissions_kg']:.6f} kg, "
                f"Energía: {emissions_data_dict['energy_consumed_kwh']:.6f} kWh, "
                f"Duración: {emissions_data_dict['duration_seconds']:.2f} s")
        
        except Exception as e:
            print(f"[{self.project_name}]: Error al detener tracker o acceder a datos: {e}")
            self._is_tracking = False # Asegurar que se marca como detenido incluso con error
        
        return emissions_data_dict

    def is_tracking(self):
        """Devuelve True si el tracker está actualmente activo."""
        return self._is_tracking
    
    def stop_tracking(self):
        self.tracker.stop()
        self._is_tracking = False 
        print(f"[{self.project_name}]: Tracker detenido.")
