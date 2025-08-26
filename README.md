# Seleccion de caracteristicas basada en información mutua para un entorno federado IoT

> **Python version:** >= 3.8

## Descripción

Este proyecto implementa un sistema de selección de características (Feature Selection) utilizando técnicas de Información Mutua (MIM y JMI) en un entorno de Aprendizaje Federado. El objetivo es identificar las características más relevantes de un conjunto de datos distribuido en dispositivos IoT (como Raspberry Pi) de manera eficiente y sin centralizar los datos.

## Motivación

En el Internet de las Cosas (IoT), los dispositivos generan enormes cantidades de datos. Analizar estos datos de forma centralizada es costoso en términos de ancho de banda, latencia, privaciadad y consumo de energía. Este proyecto aborda ese desafío aplicando selección de características directamente en los dispositivos, permitiendo entrenar modelos de Machine Learning más ligeros y rápidos.

## Características Principales

* **Selección de Características Federada:** Utiliza algoritmos como JMI y MIM.
* **Comunicación Eficiente:** Implementado sobre el protocolo MQTT.
* **Análisis Comparativo:** Permite comparar el rendimiento y consumo con un enfoque centralizado.
* **Configuración Flexible:** Parametriza fácilmente el entorno a través de un fichero `config.json`.

## Instalación

1.  Clona este repositorio:
    ```bash
    git clone [https://github.com/iskelazz/federated_mi_fs_iot.git](https://github.com/iskelazz/federated_mi_fs_iot.git)
    cd federated_mi_fs_iot
    ```

2.  **(Opcional pero recomendado) Crea y activa un entorno virtual**
    ```bash
    python3 -m venv venv
    source venv/bin/activate 
    ```

3.  Instala las dependencias. Se deben instalar las siguientes librerías para ejecutar este proyecto:
> numpy,
> scipy,
> codecarbon,
> mqtt.paho,
> scikit-learn,
> matplotlib,
> pandas

```bash
    pip install numpy scipy codecarbon paho-mqtt scikit-learn matplotlib pandas
```

---

## Guia de uso

### 1. Hacer los cortes para la validación cruzada (OBLIGATORIO)
El sistema esta pensado para aplicar selección de carecteristicas sobre la partición de entrenamiento y luego la clasificación con los clasificadores seleccionados.
  Editar los valores de la validación cruzada y el dataset en el archivo make_splits.py, ejecutar el script:
  ```bash
  # En servidor
  python3 .\make_splits.py
  ```
  Colocar el archivo con las particiones en /datasets/splits, este archivo aparecera en la raiz del proyecto con el nombre splits_<nombre_dataset>.json. Ejem. splits_arcene.json  

### 2. Configurar selección de caracteristicas

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
        "PLOT_DISPERSION": false,
        "CLASSIFIER_TYPE": ["knn", "rf"],
        "OPPORTUNITY_CROSS_SILO": true 
    },
    "FS_CENTRALIZED":{
        "DATASET_TO_LOAD_GLOBALLY": "arcene",
        "TOP_K_FEATURES_TO_SELECT": 75,
        "NUM_BINS": 5,
        "MI_FS_METHOD": "JMI",
        "CLASSIFIER_METHOD": ["knn", "rf"]
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
  - `CLASSIFIER_TYPE`: Clasificadores que se usaran para probar la selección de características.
  - `OPPORTUNITY_CROSS_SILO`: Si es true, el dataset es "opportunity" y el número de clientes = 4, aplica división por sujeto, ignorara el valor de DISTRIBUTION_TYPE y UNEVENNESS_FACTOR_NONIID. Es un parametro experimental, puede no funcionar de forma adecuada, por lo que por defecto esta a false.

  ## FS_CENTRALIZED
  Configuración necesaria para el proceso de selección de caracteristicas centralizado.

  **Parametros**:
  - `DATASET_TO_LOAD_GLOBALLY`: Dataset sobre el cual se realiza la selección de caracteristicas, debe ser un dataset válido para cargar en `load_dataset` de utils.py
  - `TOP_K_FEATURES_TO_SELECT`: Número de caracteristicas a seleccionar.
  - `NUM_BINS`: Número de bins para la discretación de los datos de los datasets.
  - `MI_FS_METHOD`: Algoritmo de IM, puede ser MIM o JMI.
  - `CLASSIFIER_TYPE`: Clasificadores que se usaran para probar la selección de características.


### 3. Iniciar clientes

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

### 4. Selección de caracteristicas federado

  Para iniciar la selección de características hay que ejecutar, en la carpeta /server_pc, el siguiente comando:
  ```bash
  # En servidor
  python3 .\server_app
  ```

  Usará la configuración de config.json de la raíz del proyecto (El bloque FS_FEDERATED), imprimirá los resultados por pantalla y guardará las características seleccionadas en la carpeta /selected_features de la raíz del proyecto. También almacenará los resultados de emisiones en la carpeta /emissions_output. Realizara la selección de caracteristicas tal y como estean configuradas las particiones de la validación cruzada, ese número de veces y en cada selección realizara la clasificación con los métodos configurados y devolvera la media de los valores y su desviación tipica.

### 5. Selección de caracteristicas centralizado

  Es un paso autónomo a los tres primeros, la configuración se realiza en el archivo de la raíz del proyecto config.json (el bloque FS_CENTRALIZED). El archivo ejecutable es /centralized/feature_selection_centralized.py

  ```bash
  python3 .\feature_selection_centralized.py
  ```


  Los resultados también se almacenarán en la carpeta /selected_features del mismo modo que el caso federado.

### 6. Calculo de TPR

  Este es el cálculo de la Tasa de Verdaderos Positivos (TPR = TP/k) donde TP es el número de características comunes, y k es el número de características seleccionadas. Requiere haber completado los pasos 1, 2, 3, 4 y 5 con la misma configuración. La instrucción en el terminal se lanza con la ruta del caso centralizado y federado `python3 .\calculate_TPR.py --fed_dir {dirección 1} --cent_dir {dirección 2} --dataset {nombre} --method {metodo de IM: MIM o JMI} --k {Numero de caracteristicas seleccionadas}`. Comparara 1 a 1 todos los resultados de la validación cruzada, almacenadas en el directorio y devolvera la media y la desviacón tipica. El resultado se muestra en la terminal.

  ```bash
  python3 .\compare_TPR.py --fed_dir .\selected_features\ --cent_dir .\selected_features\centralized\madelon\ --dataset madelon --method JMI --k 75
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

- **Caso Opportunity**:
  - Dataset muy pesado, no incluido en /datasets, para descargarlo ya preprocesado, ejecutar el script /datasets/opportunityUCI.py
  - Preprocesado: Se eliminan muestras con clase nula, y caracteristicas con más de 50% de sus celdas nulas, se aplica mediana al resto de nulos.
```bash
  python3 .\opportunityUCI.py
  ```

Nombres permitidos `load_dataset()`: `"mnist"`, `"human"`, `"gisette"`, `"arcene"`, `"madelon"`, `"gas_sensor"`, `"internet_ads"`, `opportunity`.
