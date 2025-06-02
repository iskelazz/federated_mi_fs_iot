# Seleccion de caracteristicas basada en información mutua para un entorno federado IoT

> **Python version:** >= 3.8

Esto es un herramienta de python para medir la relevancia de las caracteristicas de distintos conjuntos de datos. Para lograrlo se hace uso de tecnicas de información mutua, se usan las tecnicas de MIM y JMI
La realización de este trabajo esta pensada para ser utilizada en un entorno IoT, el objetivo es medir el rendimiento, consumo y compararlo con soluciones centralizadas. Para ello vamos hacer uso de varias raspberry pi 5, el protocolo de comunicación usado es MQTT

---

## Archivos

  ## /client_pi

  Carpeta que integra la logica que va utilizar el cliente (Raspberry pi 5), esta compuesto por:

  - **ClientApp**
  Archivo: [`client_app.py`]
    - Es el nucleo del cliente, se encarga de gestionar las operaciones que le corresponden. 
    - Gestión de operaciones MQTT (`_setup_mqtt_callbacks`, `start`, `cleanup`, `on_connected`, `on_disconnected`, `on_message_received`)
    - Reseteo de estado del cliente (`_initialize_job_state`, `_reset_current_job`)
    - `process_initial_command(self, command_data)`: Función de inicio del proceso de selección de caracteristicas, llama a las funciones _setup_job_and_load_data y _calculate_and_send_local_extremes.
    - `_setup_job_and_load_data(self, command_data)`: Configuración inicial del cliente, carga la parte del dataset que le corresponde al cliente e inicia el tracker de emisiones.
    - `_calculate_and_send_local_extremes(self)`: Calcula los maximos y minimos locales de cada caracteristica y los envia al servidor. 
    - `process_global_disc_params(self, params_data)`: Recibe del servidor los maximos y minimos globales y realiza el proceso de discretación de su parte del dataset con un numero de bins recibido en el payload, llama a la función proceed_to_calculate_and_send_initial_probabilities para calcular las tablas de probabilidad y las envia al servidor.
    - `_proceed_to_calculate_and_send_initial_probabilities(self)`: Realiza el calculo de tablas de probabilidad MIM inicial I(Xk, Y).
    - `process_jmi_batch_triplet_request(self, command_data)`: Parte del algoritmo JMI, realiza el calculo parcial de las tablas de probabilidad de la última caracteristica seleccionada con las candidatas, I((Xk,Xj), Y), envia los resultados al servidor.
    - `stop_save_and_send_emissions(self, emissions_topic)`: Para el tracker de emisiones, extrae los resultados y los envia al servidor.

  - **ClientEmissionsManager**
  Archivo: [`client_emissions_manager.py`]
    - Gestiona las operaciones del tacker de emisiones.
    - `start_tracking(self)`: Inicia el tracker.
    - `stop_tracking_and_get_data(self, log_id)`: Detiene el tracker y devuelve la información recolectada en el proceso (duración de trackeo, energia consumida y CO2 emitido).
    - `is_tracking(self)`: Devuelve un booleano que es true si el tracker esta en funcionamiento.
    - `stop_tracking(self)`: Detiene el tracker. 
  

  - **client_utils**
  Archivo: [`clients_utils.py`]
    - `calculate_local_prob_dist_array(X_discretized, y_labels, num_bins, num_classes, sim_client_id_for_log="N/A_CLIENT")`: Calcula P(Xi, Y) usando np.histogram2d.
    - `calculate_local_triplet_prob_dist(X_client_discretized, y_client_partition, k_idx, j_idx, num_bins, num_classes, sim_client_id_for_log="N/A_CLIENT")`:  "Calcula P_l(X_k, X_j, Y) usando np.histogramdd.

  ## /mqtt_handlers

  - **MQTTCommunicator**
  Archivo: [`mqtt_communicator.py`]
    - Clase envoltorio que tiene el objetivo de simplificar las funciones de comunicación usando el protocolo MQTT.
    - Gestión de la comunicación (`_on_connect`, `_on_message`, `_on_disconnect`, `_on_publish`) 
    - Callbacks (`set_message_callback`, `set_connect_callback`, `set_disconnect_callback`, `set_publish_callback`)
    - Gestión de bucle de red (`start_listening`, `stop_listening`, `loop_forever`)
    - `connect(self)`: Intenta establecer una conexión con el broker MQTT. Devuelve True si tiene éxito, False en caso contrario.
    - `publish(self, topic, payload_data, qos=0, retain=False)`: Publica un mensaje en el topic especificado. Convierte automáticamente diccionarios y listas a formato JSON string antes de enviar.  Admite payloads de tipo string, bytes o los convierte a string. Devuelve el resultado de la publicación de Paho-MQTT o None en caso de error.
    - `subscribe(self, topic, qos=0)`: Suscribe al cliente al topic especificado con el nivel de QoS dado. Almacena el topic y QoS para posibles reconexiones.

  ## /server_pc

  Carpeta que integra la lógica del servidor, compuesto por:

  - **ClientSimState**
  Archivo [`client_sim_state.py`]
    - Almacena en el servidor el estado de un cliente especifico.

  - **feature_selector**
  Archivo [`feature_selector.py`]
    - `calculate_mi_for_feature(p_XY_2D_table)`: Calcula la Información Mutua I(X;Y) para una característica X y la clase Y.
    - `select_features_mim(p_XY_data_array_3D, top_k=15)`: Realiza la selección de características utilizando el método MIM.
    - `calculate_mi_for_triplet(p_XkXjY_table_3D)`:  Calcula la Información Mutua I( (Xk,Xj); Y ) para un par de características (Xk,Xj) y la clase Y.
  
  - **JMIOrchestrator**
  Archivo [´jmi_orchestrator.py´]
    - Dirige el algoritmo de selección de características JMI en el entorno federado. Opera en su propio hilo.
    - `__init__((self, comm_instance, active_clients_dict, num_expected_clients, global_lock, aggregate_func_ref,dataset_name_for_save,save_function_ref))`: Recibe y almacena referencias a la instancia de comunicación MQTT, el diccionario de clientes activos, el número esperado de clientes, un bloqueo global para concurrencia, la función de agregación de tablas de probabilidad, el nombre del dataset y la función para guardar las características seleccionadas.
    - `_initialize_first_feature(self)`: Inicia el algoritmo JMI seleccionando la primera caracteristica con MIM.
    - `start_selection(self, aggregated_XY_tables, top_k_to_select)`: Función que gestiona de forma completa el loop JMI hasta tener el conjunto de caracteristicas. En cada iteración realiza las peticiones al cliente, espera a recibir el resultado de todos ellos para seleccionar una nueva caracteristica.
  
  - **server_app**
  Archivo [`server_app.py`] 
    - Se encarga de gestionar el hilo principal del servidor, recibe los datos de configuración de la seleccion de caracteristicas y los gestiona, tambien gestiona el bucle principal y da conclusión al proceso.

  - **ServerEmissionsManager**
  Archivo [`server_emissions_manager.py`]
    - Se encarga del soporte a las operaciones con codecarbon para estimar el consumo del proceso de selección de caracteristicas asi como su emision de gases de CO2.
    - `__init__(self, project_root_path, server_id_for_log = "server")`: Función para construir un objeto ServerEmissionsManager.
    - `set_dependencies(self, communicator, lock, active_clients_ref)`: Función para inyectar las dependencias más importantes desde server_logic, el comunicador MQTT, lock para bloquear los hilos y active_clients_ref para conocer el numero de clientes.
    - `start_server_tracking(self)`: Inicia la toma de mediciones de consumo y emisiones.
    - `reset_server_tracking(self)`: Resetea el tracker a su estado original.
    - `_handle_server_emissions_data(self)`: Para el tracker y devuelve los datos de consumo obtenidos del servidor.
    - `prepare_for_new_client_emissions_round(self, num_clients_to_command)`: Establece el número de clientes para una ronda de medición de emisiones.
    - `request_emissions_from_clients(self, client_id_target = None)`: Solicita a los clientes los datos de emisiones a traves de un topic del protocolo de comunicación MQTT.
    - `process_client_emission_report(self, data_payload)`: Recibe cada uno de los reportes de los clientes en lo referente a las emisiones.
    - `check_and_print_aggregated_emissions(self)`: Comprueba si todos los clientes reportaron sus datos de emisiones y consumo, en caso afirmativo, imprime por pantalla los datos combinados de todos los clientes y el servidor.

  - **ServerLogic**
  Archivo [server_logic.py]
    - Es el núcleo lógico del programa, se encarga de inicializar el proceso, capturar y enviar todas las comunicaciones con los clientes, guia los procesos de calculo de máximos/mínimos globales y de selección de caracteristicas, tanto MIM como JMI. Gestiona ClientSimState, JMIOrchestrator y ServerEmissionsManager.
    - `set_communicator(self, comm_instance: MQTTCommunicator)`: Función para inyectar comunicador MQTT.
    - `initialize_new_round(self, num_clients_expected_param)`: Inicializa una nueva ronda, se ejecuta con una vez en server_app al inicializar el programa.
    - `add_or_update_active_client(self, sim_client_id, dataset_name)`: Añade un nuevo cliente o actualiza uno existente para la ronda.
    - `send_processing_command_to_pi(self, sim_client_id, dataset_name, indices_list, num_global_classes_param)`: Comunicación inicial con los dispositivos del borde, se envia la parte del dataset que manejan, llamado una vez en server_app inicia el bucle de comunicación.
    - `_process_and_dispatch_global_parameters(self)`: Tras recibir los max/min locales de todos los clientes, esta función calcula los min/max globales y los envia a los dispositivos del borde de la red.
    - `_aggregate_prob_tables_common(self, collected_data_dict, expected_shape_dims=None)`: Añade de forma conjunta y con el peso adecuado, todas las tablas de probabilidad calculadas en el borde de la red.
    - `aggregate_initial_XY_prob_tables_and_trigger_selection(self)`: Agrega las primeras tablas generada en los dispositivos del borde de la red y decide el siguiente paso del algoritmo según el metodo de FS seleccionado, si es JMI inicia el JMI_Orchestrator y si es MIM lista y ordena las caracteristicas según su relevancia.
    - `handle_pi_status_update(self, status_data)`: Maneja los mensajes recibidos del borde y cambia el estado del cliente en consecuencia, si la comunicación JMI esta en curso maneja la flag que bloquea el bucle para seleccionar una caracteristica nueva, la desbloquea cuando todos los clientes hicieron su tarea.
    - `handle_pi_local_extremes(self, extremes_data)`:  Maneja la recepcion de los extremos locales, los almacena y cuando llegan todos lanza un thread para generar los globales.
    - `handle_initial_XY_prob_results(self, data_payload)`: Maneja el almacenamiento de la primera recepcion de tablas de probabilidad.
    - `handle_jmi_pair_prob_result(self, data_payload)`: Procesa un lote de tablas de probabilidad de tripletas P(Xk,Xj,Y) enviado por un cliente para una iteración JMI específica.
    - `on_server_message_received(self, topic, payload_bytes)`: Maneja la recepcion de mensajes de los dispositivos del borde de la red.
    - `save_selected_federated_features_txt(self, selected_feature_indices, dataset_name_str, actual_k_selected, technique_name)`: Guarda las caracteristicas seleccionadas en un .txt en la carpeta /selected_features en la raíz del proyecto.
    - `send_emission_request_to_clients(self)`: Petición a los clientes para que envien sus datos de consumo y emisiones.
    - `on_connected_to_broker(self)`: Topics a los que esta suscrito el servidor.
    - `on_disconnected_from_broker(self, rc)`: Desconecta al servidor del broker MQTT. 

  ## /centralized

  Proceso de selección de caracteristicas centralizado, su objetivo es proporcionar datos para la comparación con el algoritmo federado.

  - **mutual_information**
  Archivo [mutual_information.py]
    - Contiene los algoritmos de selección mutua.
    - `mi(X, Y)`: Información mutua entre una caracteristica y sus etiquetas. 
    - `MIM(X_data, Y_labels, topK)`: Selecciona topK caracteristicas usando el algoritmo de información mutua MIM.
    - `JMI(X_data, Y_labels, topK)`: Selecciona topK caracteristicas usando el algoritmo de información mutua JMI.

  - **feature_selection_centralized**
  Archivo [feature_selection_centralized.py]
    - Script que da uso a los algoritmos de selección mutua de mutual_information.py, contiene la configuración del proceso en constantes en las primeras lineas del archivo. Contiene las siguientes funciones auxiliares.
    - `load_and_prepare_data(dataset_name_to_load, n_bins_for_discretization)`: Carga el dataset, asegura la forma correcta de X y lo discretiza.
    - `select_features_centralized(X_discrete_data, y_labels, mi_function, top_k, n_original_features)`: Se encarga de llamar a las funciones de selección de caracteristicas.
    - `save_selected_features_txt(selected_feature_indices, dataset_name_str, top_k_val, technique_name, project_root_path)`: Guarda las caracteristicas seleccionadas en un .txt en la carpeta /selected_features en la raíz del proyecto.

- **calculate_TPR**
Archivo [calculate_TPR.py](calculate_TPR.py)
  - Calcula la Tasa de Verdaderos Positivos (TPR) comparando dos listas de características seleccionadas. Las listas son .txt y se pasan como argumento a este script.
  - `load_feature_indices(filepath)`: Carga los índices de características desde un archivo .txt a un conjunto.
  - `calculate_tpr(ideal_features_set, evaluated_features_set, k_value = 0)`:  Calcula la Tasa de Verdaderos Positivos (TPR) entre dos conjuntos de características. TPR = TP / k donde TP es el número de características comunes, y k es el número de características seleccionadas en el conjunto centralizado (o el k especificado).

- **classifier_evaluator**
Archivo [classifier_evaluator.py](classifier_evaluator.py)
  - Se encarga de lanzar un clasificador con el objetivo de evaluar el funcionamiento de la selección, la configuración se realiza en constantes en las primeras lineas del archivo, por lo tanto hay que editar para cambiar el funcionamiento, puede hacer uso de caracteristicas especificas o de todo el dataset.
  - `load_predefined_test_set(dataset_base_name, project_root_path)`: Carga los dataset con conjunto de test independiente.
  - `load_selected_feature_indices(filepath)`: Carga los dataset sin conjunto de test independiente.
  - `evaluate_classifier(X_train, X_test, y_train, y_test, classifier_name, dataset_name_log)`: Ejecuta el proceso de clasificación, posteriormente muestra los resultados para un conjunto de métricas.
  

- **Utilidades**
Archivo: [`utils.py`](utils.py) 
  - Cargado de datasets (`load_dataset`, `load_dataset_bin`, `load_dataset_mat`).  
  - Particionado de información: IID (`build_iid_data`) y non‑IID (`build_noniid_data`).  
  - Separación de datos para los diferentes clientes (`partition`).  
  - Discretización de un dataset con un número de bins dado (`disc_equalwidth`).
  - Calculo de cuotas de muestras (enteros) para cada usuario de forma desbalanceada (`calculate_uneven_quotas`)
  - Reparto desbalanceado de muestras entre los diferentes clientes (`build_noniid_uneven_no_loss(labels, num_users, unevenness_factor = 0.5)`)


> **Aviso:** Se deben instalar las siguientes librerias para ejecutar este proyecto:
> numpy,
> scipy,
> codecarbon,
> mqtt.paho,
> sklearn,
> zlib

---

## Uso

### 1. Configurar selección de caracteristicas

  En el archivo config.json de la raiz del proyecto debemos ajustar la selección de caracteristicas con los parametros deseados.
  ```json
  {
    "DATASET_TO_LOAD_GLOBALLY": "mnist",
    "MI_FS_METHOD": "JMI",
    "NUM_SIMULATED_CLIENTS_TOTAL": 1,
    "DISTRIBUTION_TYPE": "iid",
    "NUM_BINS": 5,
    "TOP_K_FEATURES_TO_SELECT": 75,
    "TIMEOUT_SECONDS_OVERALL": 600,
    "BROKER_ADDRESS_FOR_SERVER": "localhost",
    "BROKER_ADDRESS_FOR_CLIENT": "localhost",
    "PORT": 1883,
    "AGGREGATION_METHOD": "simple",
    "UNEVENNESS_FACTOR_NONIID": 0.0
}
  ```
  **Parametros**:
    - `DATASET_TO_LOAD_GLOBALLY`: Dataset sobre el cual se realiza la selección de caracteristicas, debe ser un dataset valido para cargar en `load_dataset` de utils.py
    - `MI_FS_METHOD`: Algoritmo de IM, puede ser MIM o JMI.
    - `NUM_SIMULATED_CLIENTS_TOTAL`: El número de clientes que usara la selección de caracteristicas, en nuestro caso igual al número de raspberrys usadas.
    - `DISTRIBUTION_TYPE`: Tipo de distribución entre los clientes se puede seleccionar iid o non-iid.
    - `NUM_BINS`: Número de bins para la discretación de los datos de los datasets.
    - `TOP_K_FEATURES_TO_SELECT`: Número de caracteristicas a seleccionar.
    - `TIMEOUT_SECONDS_OVERALL`: Timeout en el proceso de comunicación con los clientes en segundos, si se supera se aborta el proceso.
    - `BROKER_ADDRESS_FOR_SERVER`: Dirección del broker MQTT para el servidor.
    - `BROKER_ADDRESS_FOR_CLIENT`: Dirección del broker MQTT para el cliente.
    - `PORT`: Puerto del broker MQTT.
    - `AGGREGATION_METHOD`: "Simple", si todos los clientes tienen el mismo peso o "weighted" si el peso del cliente lo determinan sus muestras con respecto al total.
    - `UNEVENNESS_FACTOR_NONIID`: Si la distribución es non-iid el valor de este factor es un float, entre 0 y 1, determina el desbalanceo de muestras entre los clientes, siendo 0 un número de muestras identico entre los clientes y 1 un fuerte desbalanceo.

### 2. Iniciar clientes

  En cada raspberry pi acceder a la carpeta /client_pi y lanzar el siguiente comando en una terminal:
  ```bash
  # En raspberry Pi 1
  python3 .\client_pi.py --sim-id sim_client_0
  ```
  Para sucesivas raspberry pi debemos cambiar i, por números sucesivos (0,1,2...n), en el argumento de --sim-id donde i es: sim_client_{i}, por ejemplo:
  ```bash
  # En raspberry Pi 2
  python3 .\client_pi.py --sim-id sim_client_1
  ```
  ```bash
  # En raspberry Pi 3
  python3 .\client_pi.py --sim-id sim_client_2
  ```

### 3. Selección de caracteristicas federado

  Para iniciar la selección de caracteristicas hay que ejecutar, en la carpeta /server_pc el siguiente comando:
  ```bash
  # En servidor
  python3 .\server_app
  ```

  Usara la configuración de config.json de la raiz del proyecto, imprimira los resultados por pantalla y guardara las caracteristicas seleccionadas en la carpeta /selected_features de la raiz del proyecto. También almacenara los resultados de emisiones en la carpeta /emissions_output

### 4. Selección de caracteristicas centralizado

  Es un paso autónomo a los tres primeros, la configuración se realiza en las constantes de las primeras lineas de código del archivo /centralized/feature_selection_centralized.py

  ```python
    # --- Parámetros de Configuración ---
    DATASET_NAME = "mnist"
    TOP_K_FEATURES = 75
    N_BINS_DISCRETIZATION = 5
    MI_TECHNIQUE_FUNCTION = JMI # Puedes cambiar esto a MIM 
  ```

  **Parametros**:
    - `DATASET_NAME`: Dataset sobre el cual se realiza la selección de caracteristicas, debe ser un dataset valido para cargar en `load_dataset` de utils.py
    - `TOP_K_FEATURES`: Número de caracteristicas a seleccionar.
    - `N_BINS_DISCRETIZATION`: Número de bins para la discretación de los datos de los datasets.
    - `MI_TECHNIQUE_FUNCTION`: Algoritmo de IM, puede ser MIM o JMI.

  Los resultados tambien se almacenaran en la carpeta /selected_features del mismo modo que el caso federado.

### 5. Calculo de TPR

  Este es el calculo de la Tasa de Verdaderos Positivos TPR = TP / k donde TP es el número de características comunes, y k es el número de características seleccionadas. Requiere haber completado los pasos 1, 2, 3 y 4 con la misma configuración. La instrucción en el terminal se lanza con la ruta del caso centralizado y federado `python3 .\calculate_TPR.py {dirección 1} {dirección 2}` el resultado se muestra en la terminal.

  ```bash
  python3 .\calculate_TPR.py selected_features\arcene_centralized_selected_top75_JMI_feature_indices.txt selected_features/arcene_federated_selected_top75_JMI_federated_feature_indices.txt
  ```

### 6. Proceso de clasificación

  En este paso, usaremos algoritmos de clasificación (knn, random forest y regresión logística) para valorar su rendimiento con las caracteristicas seleccionadas y compararlo. {En construcción}

## Datasets Folder

Los datasets estan localizados en la ruta `datasets/<name>/`, las rutas estan hardcodeadas en `utils.py`:

- **Datasets binarios** (`.data` & `.labels`):  
  - `datasets/gisette/gisette_train.data`  
  - `datasets/gisette/gisette_train.labels`  
  - `datasets/gisette/gisette_valid.data`  
  - `datasets/gisette/gisette_valid.labels`

- **MAT files**:  
  - `datasets/MNIST_et.mat`  
  - `datasets/humanActivity.mat`

Nombres permitidos `load_dataset()`: `"mnist"`, `"human"`, `"gisette"`, `"arcene"`, `"madelon"`, `"gas_sensor"`, `"internet_ads"`.
