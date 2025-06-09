# Seleccion de caracteristicas basada en información mutua para un entorno federado IoT

> **Python version:** >= 3.8

Esto es un herramienta de Python para medir la relevancia de las características de distintos conjuntos de datos. Para lograrlo se hace uso de técnicas de información mutua; se usan las técnicas de MIM y JMI.
La realización de este trabajo está pensada para ser utilizada en un entorno IoT. El objetivo es medir el rendimiento y consumo, y compararlo con soluciones centralizadas. Para ello vamos hacer uso de varias Raspberry Pi 5, el protocolo de comunicación usado es MQTT.

---

## Archivos

  ## /client_pi

  Carpeta que integra la lógica que va utilizar el cliente (Raspberry Pi 5). Está compuesto por:

  - **ClientApp**
  Archivo: [`client_app.py`](client_pi/client_app.py)
    - Es el núcleo del cliente, se encarga de gestionar las operaciones que le corresponden. 
    - Gestión de operaciones MQTT (`_setup_mqtt_callbacks`, `start`, `cleanup`, `on_connected`, `on_disconnected`, `on_message_received`)
    - Reseteo de estado del cliente (`_initialize_job_state`, `_reset_current_job`)
    - Carga de configuración de config.json (`_load_simulation_config`)
    - `process_initial_command(self, command_data)`: Función de inicio del proceso de selección de características, llama a las funciones _setup_job_and_load_data y _calculate_and_send_local_extremes.
    - `_setup_job_and_load_data(self, command_data)`: Configuración inicial del cliente, carga la parte del dataset que le corresponde al cliente e inicia el tracker de emisiones.
    - `_calculate_and_send_local_extremes(self)`: Calcula los máximos y mínimos locales de cada característica y los envía al servidor. 
    - `process_global_disc_params(self, params_data)`: Recibe del servidor los máximos y mínimos globales y realiza el proceso de discretización de su parte del dataset con un número de bins recibido en el payload; llama a la función proceed_to_calculate_and_send_initial_probabilities para calcular las tablas de probabilidad y las envía al servidor.
    - `_proceed_to_calculate_and_send_initial_probabilities(self)`: Realiza el cálculo de tablas de probabilidad MIM inicial I(Xk, Y).
    - `process_jmi_batch_triplet_request(self, command_data)`: Parte del algoritmo JMI, realiza el cálculo parcial de las tablas de probabilidad de la última caracteristica seleccionada con las candidatas, I((Xk,Xj), Y), envía los resultados al servidor.
    - `stop_save_and_send_emissions(self, emissions_topic)`: Para el tracker de emisiones, extrae los resultados y los envía al servidor.

  - **ClientEmissionsManager**
  Archivo: [`client_emissions_manager.py`](client_pi/client_emissions_manager.py)
    - Gestiona las operaciones del tacker de emisiones.
    - `start_tracking(self)`: Inicia el tracker.
    - `stop_tracking_and_get_data(self, log_id)`: Detiene el tracker y devuelve la información recolectada en el proceso (duración de trackeo, energía consumida y CO2 emitido).
    - `is_tracking(self)`: Devuelve un booleano que es true si el tracker está en funcionamiento.
    - `stop_tracking(self)`: Detiene el tracker. 
  

  - **client_utils**
  Archivo: [`clients_utils.py`](client_pi/client_utils.py)
    - `calculate_local_prob_dist_array(X_discretized, y_labels, num_bins, num_classes, sim_client_id_for_log="N/A_CLIENT")`: Calcula P(Xi, Y) usando np.histogram2d.
    - `calculate_local_triplet_prob_dist(X_client_discretized, y_client_partition, k_idx, j_idx, num_bins, num_classes, sim_client_id_for_log="N/A_CLIENT")`:  "Calcula P_l(X_k, X_j, Y) usando np.histogramdd.

  ## /mqtt_handlers

  - **MQTTCommunicator**
  Archivo: [`mqtt_communicator.py`](mqtt_handlers/mqtt_communicator.py)
    - Clase envoltorio que tiene el objetivo de simplificar las funciones de comunicación usando el protocolo MQTT.
    - Gestión de la comunicación (`_on_connect`, `_on_message`, `_on_disconnect`, `_on_publish`) 
    - Callbacks (`set_message_callback`, `set_connect_callback`, `set_disconnect_callback`, `set_publish_callback`)
    - Gestión de bucle de red (`start_listening`, `stop_listening`, `loop_forever`)
    - `connect(self)`: Intenta establecer una conexión con el broker MQTT. Devuelve True si tiene éxito, False en caso contrario.
    - `publish(self, topic, payload_data, qos=0, retain=False)`: Publica un mensaje en el topic especificado. Convierte automáticamente diccionarios y listas a formato JSON antes de enviar.  Admite payloads de tipo string, bytes o los convierte a string. Devuelve el resultado de la publicación de Paho-MQTT o None en caso de error.
    - `subscribe(self, topic, qos=0)`: Suscribe al cliente al topic especificado con el nivel de QoS dado. Almacena el topic y QoS para posibles reconexiones.

  ## /server_pc

  Carpeta que integra la lógica del servidor. Compuesto por:

  - **ClientSimState**
  Archivo [`client_sim_state.py`](server_pc/client_sim_state.py)
    - Almacena en el servidor el estado de un cliente específico.

  - **feature_selector**
  Archivo [`feature_selector.py`](server_pc/feature_selector.py)
    - `calculate_mi_for_feature(p_XY_2D_table)`: Calcula la Información Mutua I(X;Y) para una característica X y la clase Y.
    - `select_features_mim(p_XY_data_array_3D, top_k=15)`: Realiza la selección de características utilizando el método MIM.
    - `calculate_mi_for_triplet(p_XkXjY_table_3D)`:  Calcula la Información Mutua I( (Xk,Xj); Y ) para un par de características (Xk,Xj) y la clase Y.
  
  - **JMIOrchestrator**
  Archivo [´jmi_orchestrator.py´](server_pc/jmi_orchestrator.py)
    - Dirige el algoritmo de selección de características JMI en el entorno federado. Opera en su propio hilo.
    - `__init__((self, comm_instance, active_clients_dict, num_expected_clients, global_lock, aggregate_func_ref,dataset_name_for_save,save_function_ref))`: Recibe y almacena referencias a la instancia de comunicación MQTT, el diccionario de clientes activos, el número esperado de clientes, un bloqueo global para concurrencia, la función de agregación de tablas de probabilidad, el nombre del dataset y la función para guardar las características seleccionadas.
    - `_initialize_first_feature(self)`: Inicia el algoritmo JMI seleccionando la primera característica con MIM.
    - `start_selection(self, aggregated_XY_tables, top_k_to_select)`: Función que gestiona de forma completa el loop JMI hasta tener el conjunto de características. En cada iteración realiza las peticiones al cliente y espera a recibir el resultado de todos ellos para seleccionar una nueva característica.
  
  - **server_app**
  Archivo [`server_app.py`](server_pc/server_app.py)
    - Se encarga de gestionar el hilo principal del servidor, recibe los datos de configuración de la selección de características y los gestiona, tambien gestiona el bucle principal y da conclusión al proceso.
    - Carga de configuración de config.json (`_load_simulation_config`).
    - `generate_and_display_label_dispersion(config,unique_global_labels,num_total_clients,client_data_indices_map,global_labels_array,dataset_name_global,distribution_type_global)`: Crea un gráfico para poder visualizar la dispersión de los datos entre los clientes.
  - **ServerEmissionsManager**
  Archivo [`server_emissions_manager.py`](server_pc/server_emissions_manager.py)
    - Se encarga del soporte a las operaciones con codecarbon para estimar el consumo del proceso de selección de características asi como su emisión de gases de CO2.
    - `__init__(self, project_root_path, server_id_for_log = "server")`: Función para construir un objeto ServerEmissionsManager.
    - `set_dependencies(self, communicator, lock, active_clients_ref)`: Función para inyectar las dependencias más importantes desde server_logic, el comunicador MQTT, lock para bloquear los hilos y active_clients_ref para conocer el número de clientes.
    - `start_server_tracking(self)`: Inicia la toma de mediciones de consumo y emisiones.
    - `reset_server_tracking(self)`: Resetea el tracker a su estado original.
    - `_handle_server_emissions_data(self)`: Para el tracker y devuelve los datos de consumo obtenidos del servidor.
    - `prepare_for_new_client_emissions_round(self, num_clients_to_command)`: Establece el número de clientes para una ronda de medición de emisiones.
    - `request_emissions_from_clients(self, client_id_target = None)`: Solicita a los clientes los datos de emisiones a través de un topic del protocolo de comunicación MQTT.
    - `process_client_emission_report(self, data_payload)`: Recibe cada uno de los reportes de los clientes en lo referente a las emisiones.
    - `check_and_print_aggregated_emissions(self)`: Comprueba si todos los clientes reportaron sus datos de emisiones y consumo; en caso afirmativo, imprime por pantalla los datos combinados de todos los clientes y el servidor.

  - **ServerLogic**
  Archivo [server_logic.py](server_pc/server_logic.py)
    - Es el núcleo lógico del programa. Se encarga de inicializar el proceso, capturar y enviar todas las comunicaciones con los clientes, guía los procesos de cálculo de máximos/mínimos globales y de selección de características, tanto MIM como JMI. Gestiona ClientSimState, JMIOrchestrator y ServerEmissionsManager.
    - `set_communicator(self, comm_instance: MQTTCommunicator)`: Función para inyectar comunicador MQTT.
    - `initialize_new_round(self, num_clients_expected_param)`: Inicializa una nueva ronda; se ejecuta con una vez en server_app al inicializar el programa.
    - `handle_client_bench_update(self, bench_json)`: Almacena los datos de tiempos tomados del cliente.
    - `get_bench_summary(self)`: Devuelve los datos de tiempos tomados del cliente.
    - `add_or_update_active_client(self, sim_client_id, dataset_name)`: Añade un nuevo cliente o actualiza uno existente para la ronda.
    - `send_processing_command_to_pi(self, sim_client_id, dataset_name, indices_list, num_global_classes_param)`: Comunicación inicial con los dispositivos del borde; se envía la parte del dataset que manejan, llamado una vez en server_app inicia el bucle de comunicación.
    - `_process_and_dispatch_global_parameters(self)`: Tras recibir los máx./mín. locales de todos los clientes, esta función calcula los máx./mín. globales y los envía a los dispositivos del borde de la red.
    - `_aggregate_prob_tables_common(self, collected_data_dict, expected_shape_dims=None)`: Añade de forma conjunta y con el peso adecuado, todas las tablas de probabilidad calculadas en el borde de la red.
    - `aggregate_initial_XY_prob_tables_and_trigger_selection(self)`: Agrega las primeras tablas generada en los dispositivos del borde de la red y decide el siguiente paso del algoritmo según el metodo de FS seleccionado: si es JMI, inicia el JMI_Orchestrator; si es MIM, lista y ordena las caracteristicas según su relevancia.
    - `handle_pi_status_update(self, status_data)`: Maneja los mensajes recibidos del borde y cambia el estado del cliente en consecuencia. Si la comunicación JMI está en curso, maneja la flag que bloquea el bucle para seleccionar una característica nueva; la desbloquea cuando todos los clientes hicieron su tarea.
    - `handle_pi_local_extremes(self, extremes_data)`:  Maneja la recepción de los extremos locales, los almacena y, cuando llegan todos, lanza un thread para generar los globales.
    - `handle_initial_XY_prob_results(self, data_payload)`: Maneja el almacenamiento de la primera recepcion de tablas de probabilidad.
    - `handle_jmi_pair_prob_result(self, data_payload)`: Procesa un lote de tablas de probabilidad de tripletas P(Xk,Xj,Y) enviado por un cliente para una iteración JMI específica.
    - `on_server_message_received(self, topic, payload_bytes)`: Maneja la recepción de mensajes de los dispositivos del borde de la red.
    - `save_selected_federated_features_txt(self, selected_feature_indices, dataset_name_str, actual_k_selected, technique_name)`: Guarda las características seleccionadas en un .txt en la carpeta /selected_features en la raíz del proyecto.
    - `send_emission_request_to_clients(self)`: Petición a los clientes para que envien sus datos de consumo y emisiones.
    - `on_connected_to_broker(self)`: Topics a los que está suscrito el servidor.
    - `on_disconnected_from_broker(self, rc)`: Desconecta al servidor del broker MQTT. 

  ## /centralized

  Proceso de selección de características centralizado, su objetivo es proporcionar datos para la comparación con el algoritmo federado.

  - **mutual_information**
  Archivo [mutual_information.py](centralized/mutual_information.py)
    - Contiene los algoritmos de selección mutua.
    - `mi(X, Y)`: Información mutua entre una característica y sus etiquetas. 
    - `MIM(X_data, Y_labels, topK)`: Selecciona topK características usando el algoritmo de información mutua MIM.
    - `JMI(X_data, Y_labels, topK)`: Selecciona topK características usando el algoritmo de información mutua JMI.

  - **feature_selection_centralized**
  Archivo [feature_selection_centralized.py](centralized/feature_selection_centralized.py)
    - Script que da uso a los algoritmos de selección mutua de mutual_information.py 
    Contiene la configuración del proceso en constantes en las primeras líneas del archivo. Contiene las siguientes funciones auxiliares.
    - `load_and_prepare_data(dataset_name_to_load, n_bins_for_discretization)`: Carga el dataset, asegura la forma correcta de X y lo discretiza.
    - `select_features_centralized(X_discrete_data, y_labels, mi_function, top_k, n_original_features)`: Se encarga de llamar a las funciones de selección de características.
    - `save_selected_features_txt(selected_feature_indices, dataset_name_str, top_k_val, technique_name, project_root_path)`: Guarda las características seleccionadas en un .txt en la carpeta /selected_features en la raíz del proyecto.
    - Carga de configuración de config.json (`load_simulation_config`).

- **calculate_TPR**
Archivo [calculate_TPR.py](calculate_TPR.py)
  - Calcula la Tasa de Verdaderos Positivos (TPR) comparando dos listas de características seleccionadas. Las listas son .txt y se pasan como argumento a este script.
  - `load_feature_indices(filepath)`: Carga los índices de características desde un archivo .txt a un conjunto.
  - `calculate_tpr(ideal_features_set, evaluated_features_set, k_value = 0)`:  Calcula la Tasa de Verdaderos Positivos (TPR) entre dos conjuntos de características. TPR = TP/k, donde TP es el número de características comunes, y k es el número de características seleccionadas en el conjunto centralizado (o el k especificado).

- **classifier_evaluator**
Archivo [classifier_evaluator.py](classifier_evaluator.py)
  - Se encarga de lanzar un clasificador con el objetivo de evaluar el funcionamiento de la selección, la configuración se realiza en constantes en las primeras lineas del archivo, por lo tanto hay que editar para cambiar el funcionamiento, puede hacer uso de caracteristicas especificas o de todo el dataset.
  - `load_predefined_test_set(dataset_base_name, project_root_path)`: Carga los dataset con conjunto de test independiente.
  - `load_selected_feature_indices(filepath)`: Carga los dataset sin conjunto de test independiente.
  - `evaluate_classifier(X_train, X_test, y_train, y_test, classifier_name, dataset_name_log)`: Ejecuta el proceso de clasificación, posteriormente muestra los resultados para un conjunto de métricas.
  - - Carga de configuración de config.json (`load_simulation_config`).
  

- **Utilidades**
Archivo: [`utils.py`](utils.py) 
  - Cargado de datasets (`load_dataset`, `load_dataset_bin`, `load_dataset_mat`).  
  - Particionado de información: IID (`build_iid_data`) y non‑IID (`build_noniid_data`).  
  - Separación de datos para los diferentes clientes (`partition`).  
  - Discretización de un dataset con un número de bins dado (`disc_equalwidth`).
  - Calculo de cuotas de muestras (enteros) para cada usuario de forma desbalanceada (`calculate_uneven_quotas`).
  - Reparto desbalanceado de muestras entre los diferentes clientes (`build_noniid_uneven_no_loss(labels, num_users, unevenness_factor = 0.5)`).
  - `plot_label_dispersion_matplotlib_only(device_label_counts, client_order, label_order, title="Distribución de Etiquetas por Cliente")`:  Genera y muestra un gráfico de barras apiladas utilizando Matplotlib.


> **Aviso:** Se deben instalar las siguientes librerías para ejecutar este proyecto:
> numpy,
> scipy,
> codecarbon,
> mqtt.paho,
> scikit-learn,
> matplotlib

---

## Uso

### 1. Configurar selección de caracteristicas

  En el archivo config.json de la raíz del proyecto debemos ajustar la selección de características con los parametros deseados.
  ```json
  {
    "FS_FEDERATED": {
        "DATASET_TO_LOAD_GLOBALLY": "arcene",
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
        "UNEVENNESS_FACTOR_NONIID": 0.0,
        "PLOT_DISPERSION": false
    },
    "FS_CENTRALIZED":{
        "DATASET_TO_LOAD_GLOBALLY": "arcene",
        "TOP_K_FEATURES_TO_SELECT": 75,
        "NUM_BINS": 5,
        "MI_FS_METHOD": "JMI"
    },
    "CLASSIFIER": {
        "DATASET_NAME": "mnist",
        "CLASSIFIER_CHOICE": "KNN",
        "TEST_SPLIT_RATIO":0.3,
        "SCALE_FEATURES": true,
        "USE_ALL_FEATURES": true,
        "ITERATIONS": 5,
        "FILE_NAME_FS": "mnist_federated_selected_top75_JMI_federated_feature_indices.txt"
    }
}
  ```
  ## FS_FEDERATED
  Configuración necesaria para el proceso de selección de caracteristicas federado.

  **Parametros**:
  - `DATASET_TO_LOAD_GLOBALLY`: Dataset sobre el cual se realiza la selección de caracteristicas, debe ser un dataset válido para cargar en `load_dataset` de utils.py
  - `MI_FS_METHOD`: Algoritmo de IM, puede ser MIM o JMI.
  - `NUM_SIMULATED_CLIENTS_TOTAL`: El número de clientes que usara la selección de caracteristicas, en nuestro caso igual al número de Raspberry Pi usadas.
  - `DISTRIBUTION_TYPE`: Tipo de distribución entre los clientes; se puede seleccionar iid o non-iid.
  - `NUM_BINS`: Número de bins para la discretación de los datos de los datasets.
  - `TOP_K_FEATURES_TO_SELECT`: Número de caracteristicas a seleccionar.
  - `TIMEOUT_SECONDS_OVERALL`: Timeout en el proceso de comunicación con los clientes en segundos; si se supera, se aborta el proceso.
  - `BROKER_ADDRESS_FOR_SERVER`: Dirección del broker MQTT para el servidor.
  - `BROKER_ADDRESS_FOR_CLIENT`: Dirección del broker MQTT para el cliente.
  - `PORT`: Puerto del broker MQTT.
  - `AGGREGATION_METHOD`: "Simple", si todos los clientes tienen el mismo peso o "weighted" si el peso del cliente lo determinan sus muestras con respecto al total.
  - `UNEVENNESS_FACTOR_NONIID`: Si la distribución es non-iid, el valor de este factor es un float entre 0 y 1 determina el desbalanceo de muestras entre los clientes, siendo 0 un número identico de muestras entre los clientes y 1 un fuerte desbalanceo.
  - `PLOT_DISPERSION`: Si es true, devuelve un gráfico de barras apiladas que representa la dispersión del dataset entre los clientes.

  ## FS_CENTRALIZED
  Configuración necesaria para el proceso de selección de caracteristicas centralizado.

  **Parametros**:
  - `DATASET_TO_LOAD_GLOBALLY`: Dataset sobre el cual se realiza la selección de caracteristicas, debe ser un dataset válido para cargar en `load_dataset` de utils.py
  - `TOP_K_FEATURES_TO_SELECT`: Número de caracteristicas a seleccionar.
  - `NUM_BINS`: Número de bins para la discretación de los datos de los datasets.
  - `MI_FS_METHOD`: Algoritmo de IM, puede ser MIM o JMI.

  ## CLASSIFIER
  Configuración necesaria para el proceso de clasificación.

  **Parametros**:
  - `DATASET_NAME`: Dataset sobre el cual se realiza el proceso de clasificación, debe ser un dataset válido para cargar en `load_dataset` de utils.py
  - `CLASSIFIER_CHOICE`: Técnica de clasificación seleccionada, estan soportadas KNN (es kNN 5 vecinos), RF (Random Forest), NAIVE_BAYES (Método Naive bayes), LOGISTIC_REGRESSION (regresión logistica con kernel liblinear).
  - `TEST_SPLIT_RATIO`: Partición entre el conjunto de entrenamiento y el de test, el valor (representado entre 0 y 1) es el porcentaje del conjunto de test, el resto se usara para entrenamiento.
  - `SCALE_FEATURES`: Indica si se escalan o no las caracteristicas.
  - `USE_ALL_FEATURES`: Si es true se entrenara el modelo con todas las características, si es falso usara el valor de FILE_NAME_FS, para seleccionar las caracteristicas indicadas en ese .txt.
  - `ITERATIONS`: Iteraciones de entrenamientos y métricas con un dataset, tiene el objetivo de obtener un valor más consistente para la precisión asi como una desviación tipica.
  - `FILE_NAME_FS`: Valor para indicar el nombre del archivo que contiene las características seleccionadas, se encontra por defecto en la carpeta /selected_features. 

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

  Para iniciar la selección de características hay que ejecutar, en la carpeta /server_pc, el siguiente comando:
  ```bash
  # En servidor
  python3 .\server_app
  ```

  Usará la configuración de config.json de la raíz del proyecto (El bloque FS_FEDERATED), imprimirá los resultados por pantalla y guardará las características seleccionadas en la carpeta /selected_features de la raíz del proyecto. También almacenará los resultados de emisiones en la carpeta /emissions_output

### 4. Selección de caracteristicas centralizado

  Es un paso autónomo a los tres primeros, la configuración se realiza en el archivo de la raíz del proyecto config.json (el bloque FS_CENTRALIZED). El archivo ejecutable es /centralized/feature_selection_centralized.py

  ```bash
  python3 .\feature_selection_centralized.py
  ```


  Los resultados también se almacenarán en la carpeta /selected_features del mismo modo que el caso federado.

### 5. Calculo de TPR

  Este es el cálculo de la Tasa de Verdaderos Positivos (TPR = TP/k) donde TP es el número de características comunes, y k es el número de características seleccionadas. Requiere haber completado los pasos 1, 2, 3 y 4 con la misma configuración. La instrucción en el terminal se lanza con la ruta del caso centralizado y federado `python3 .\calculate_TPR.py {dirección 1} {dirección 2}` el resultado se muestra en la terminal.

  ```bash
  python3 .\calculate_TPR.py selected_features\arcene_centralized_selected_top75_JMI_feature_indices.txt selected_features/arcene_federated_selected_top75_JMI_federated_feature_indices.txt
  ```

### 6. Proceso de clasificación

  En este paso, usaremos algoritmos de clasificación (KNN, Random Forest y Regresión Logística) para valorar su rendimiento con las características seleccionadas y compararlo. La configuración de la clasificación se realizará en el archivo config.json de la raíz del proyecto (El bloque CLASSIFIER), devolverá métricas para cada iteración asi como una evaluación final más robusta con todas las iteraciones. Se debe ejecutar el archivo classifier_evaluator.py de la raíz del proyecto.

  ```bash
  python3 .\classifier_evaluator.py
  ```

## Datasets Folder

Los datasets están localizados en la ruta `datasets/<name>/`, las rutas están definidas (hardcodeadas) en `utils.py`:

- **Datasets binarios** (`.data` & `.labels`):  
  - `datasets/gisette/gisette_train.data`  
  - `datasets/gisette/gisette_train.labels`  
  - `datasets/gisette/gisette_valid.data`  
  - `datasets/gisette/gisette_valid.labels`

- **MAT files**:  
  - `datasets/MNIST_et.mat`  
  - `datasets/humanActivity.mat`

Nombres permitidos `load_dataset()`: `"mnist"`, `"human"`, `"gisette"`, `"arcene"`, `"madelon"`, `"gas_sensor"`, `"internet_ads"`.
