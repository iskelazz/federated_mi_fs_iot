# Seleccion de caracteristicas basada en información mutua para un entorno federado IoT

> **Python version:** >= 3.8

Esto es un herramienta de python para medir la relevancia de las caracteristicas de distintos conjuntos de datos. Para lograrlo se hace uso de tecnicas de información mutua, se usan las tecnicas de MIM y JMI
La realización de este trabajo esta pensada para ser utilizada en un entorno IoT, el objetivo es medir el rendimiento, consumo y compararlo con soluciones centralizadas. Para ello vamos hacer uso de varias raspberry pi 5, el protocolo de comunicación usado es MQTT

---

## Archivos

  ## Client_pi

  Carpeta que integra la logica que va utilizar el cliente (raspberry pi 5), esta compuesto por:
  - **client_app**
  Archivo: [`client_app.py`]
    - Es el nucleo del cliente, se encarga de gestionar las operaciones que le corresponden:
    - Gestión de operaciones MQTT (`_setup_mqtt_callbacks`, `start`, `cleanup`, `on_connected`, `on_disconnected`, `on_message_received`)
    - Reseteo de estado del cliente (`_initialize_job_state`, `_reset_current_job`)
    - `process_initial_command(self, command_data)`: Función de inicio del proceso de selección de caracteristicas, recibe los indices de la partición del servidor y el dataset, incia el calculo de maximos y minimos locales y envia los resultados al servidor.
  File: [`mutual_information.py`](mutual_information.py) 
    - `mi(X, Y)`: Mutual information between a single feature and labels.  
    - `MIM(X_data, Y_labels, topK)`: Selects top-K features based on individual MI scores.
    - `JMI(X_data, Y_labels, topK)`: Iteratively selects features that maximize joint MI with previously selected features.

- **Federated MI** 
File: [`mutual_information_fed.py`](mutual_information_fed.py) 
  - `MIM_fed(dataXY, topK)`: Federated version of MIM.  
    - Computes local joint distributions per client.  
    - Aggregates them to estimate global MI.  
    - Returns selected features, total elapsed time, and CO₂ emissions estimate.  
  - `JMI_fed(dataXY, topK)`: Federated JMI.  
    - Builds on federated MIM to iteratively pick features with joint MI.  
    - Tracks client‑side and aggregation timings, plus emissions.

- **Utilities**
File: [`utils.py`](utils.py) 
  - Dataset loading (`load_dataset`, `load_dataset_bin`, `load_dataset_mat`).  
  - Data partitioning: IID (`build_iid_data`) and non‑IID (`build_noniid_data`).  
  - Splitting into per‑client lists (`partition`).  
  - Equal‑width discretization (`disc_equalwidth`).


> **Note:** `requirements.txt` should list at least:
> ```txt
> numpy
> scipy
> codecarbon
> ```

---

## Usage



## Datasets Folder

Datasets are located in the folders `datasets/<name>/`, matching the loaders in `utils.py`:

- **Binary datasets** (`.data` & `.labels`):  
  - `datasets/gisette/gisette_train.data`  
  - `datasets/gisette/gisette_train.labels`  
  - `datasets/gisette/gisette_valid.data`  
  - `datasets/gisette/gisette_valid.labels`

- **MAT files**:  
  - `datasets/MNIST_et.mat`  
  - `datasets/humanActivity.mat`

Supported names in `load_dataset()`: `"mnist"`, `"human"`, `"gisette"`, `"arcene"`.
