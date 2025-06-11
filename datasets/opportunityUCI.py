#!/usr/bin/env python3
"""
make_opportunity_mat.py  (versión «drop-50 %, drop-Null, median fill»)

Genera opportunity.mat con:
    - data      → ndarray float32  [n_samples, n_features]
    - labels    → ndarray int16    [n_samples]
    - subjects  → ndarray int8     [n_samples]
"""

import os, zipfile, glob, argparse
import numpy as np
import pandas as pd
import scipy.io as sp
from urllib.request import urlretrieve

# ---------------------------------------------------------------------
URL_ZIP   = ("https://archive.ics.uci.edu/static/public/226/"
             "opportunity%2Bactivity%2Brecognition.zip")
RAW_DIR   = "OpportunityUCIDataset/dataset"
SAVE_PATH = "datasets/opportunity.mat"

DOWNSAMPLE_FACTOR = 1            # 300 Hz → 30 Hz
KEEP_COLUMNS      = slice(1, 243) # 242 sensores, sin MILLISEC
LABEL_COL         = 243           # 0-index → col. 244 = Locomotion
DROP_THRESHOLD    = 0.50          # >50 % NaN en cualquier sujeto ⇒ fuera
# ---------------------------------------------------------------------


def download_and_extract(zip_path: str):
    if not os.path.exists(zip_path):
        print("Descargando ZIP (~292 MB)…")
        urlretrieve(URL_ZIP, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        print("Extrayendo .dat…")
        zf.extractall()
    print("Extracción completada.")


def build_numpy_arrays():
    files = sorted(glob.glob(f"{RAW_DIR}/S*-*.dat"))
    assert len(files) == 24, "Se esperaban 24 ficheros .dat"

    X_list, y_list, subj_list = [], [], []

    for f in files:
        df = pd.read_csv(f, sep=r"\s+", header=None)
        df.replace(-1, np.nan, inplace=True)

        sensors = df.iloc[:, KEEP_COLUMNS]               # 242 columnas
        labels  = df.iloc[:, LABEL_COL]

        mask = labels.notna() & (labels != 0)            # descarta Null
        sensors, labels = sensors[mask], labels[mask]

        sensors = sensors.iloc[::DOWNSAMPLE_FACTOR].to_numpy(np.float32)
        labels  = labels.iloc[::DOWNSAMPLE_FACTOR].to_numpy(np.int16)

        subj_id = int(os.path.basename(f).split('-')[0][1])
        X_list.append(sensors)
        y_list.append(labels)
        subj_list.append(np.full(labels.shape, subj_id, np.int8))

        print(f"{os.path.basename(f):12} → {sensors.shape[0]:5} muestras,"
              f" {sensors.shape[1]} features")

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    subjects = np.hstack(subj_list)
    print(f"Apilado global: {X.shape}")

    # -------- 1) DROP columnas >50 % NaN en algún sujeto --------
    cols_to_drop = np.zeros(X.shape[1], dtype=bool)
    for s in np.unique(subjects):
        frac_nan = np.isnan(X[subjects == s]).mean(axis=0)
        cols_to_drop |= (frac_nan > DROP_THRESHOLD)
    X = X[:, ~cols_to_drop]
    print(f"Eliminadas {cols_to_drop.sum()} columnas; quedan {X.shape[1]}")

    # -------- 2) Imputación mediana global --------
    med = np.nanmedian(X, axis=0)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = med[nan_idx[1]]
    print("Imputación completada (mediana por feature)")

    print("Etiquetas finales:", np.unique(y))
    return X.astype(np.float32), y, subjects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="opportunity_raw.zip",
                        help="Archivo ZIP descargado")
    args = parser.parse_args()

    download_and_extract(args.zip)
    X, y, subj = build_numpy_arrays()

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    sp.savemat(SAVE_PATH, {"data": X, "labels": y, "subjects": subj})
    print(f"Guardado {SAVE_PATH} ✓")


if __name__ == "__main__":
    main()
