"""Carga la misma versión champion para la API y la inferencia diaria."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from Infraestructure.flows.flow_pipeline_prefect import FEATURES


def cargar_champion(config):
    registro = config["registro"]
    nombre, alias = registro["nombre_modelo"], registro.get("alias", "champion")
    uri = os.environ.get("MLFLOW_TRACKING_URI") or config["mlflow_tracking_uri"]
    mlflow.set_tracking_uri(uri)
    cliente = MlflowClient(tracking_uri=uri)
    try:
        version = cliente.get_model_version_by_alias(nombre, alias)
    except MlflowException as exc:
        raise RuntimeError(
            f"No se pudo obtener {nombre}@{alias}. Comprueba MLflow y ejecuta primero el entrenamiento."
        ) from exc
    if not version.run_id:
        raise RuntimeError("La versión registrada debe estar vinculada a su run de entrenamiento.")
    ruta_metadatos = config.get("inferencia", {}).get("metadatos_artefacto", "resultados/metadatos.json")
    with TemporaryDirectory(prefix="seoul-bike-champion-") as carpeta:
        ruta = cliente.download_artifacts(version.run_id, ruta_metadatos, carpeta)
        with Path(ruta).open(encoding="utf-8") as archivo:
            metadatos = json.load(archivo)
    if metadatos.get("features") != FEATURES or metadatos.get("horizonte_horas") != 24:
        raise RuntimeError("El champion no coincide con el pipeline del día siguiente.")
    origen = pd.Timestamp(metadatos["origen_fecha"])
    if pd.isna(origen) or origen.tzinfo is not None or origen != origen.normalize():
        raise RuntimeError("El origen del modelo debe ser una fecha válida sin zona horaria.")
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}/{version.version}")
    if list(getattr(modelo, "feature_names_in_", [])) != FEATURES:
        raise RuntimeError("El esquema de entrada del champion no coincide con el pipeline.")
    return {"modelo": modelo, "nombre": nombre, "alias": alias,
            "version": str(version.version), "run_id": version.run_id, "origen": origen}
