"""Registra un modelo entrenado y asigna el alias champion en MLflow.

Desde la raíz: poetry run python -m Infraestructure.flows.registrar_champion
"""

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from Infraestructure.flows.flow_pipeline_prefect import CONFIG, FEATURES, resolver_ruta


def registrar_champion(artefactos, reemplazar=False):
    with CONFIG.open(encoding="utf-8") as archivo:
        cfg = yaml.safe_load(archivo)
    uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow_tracking_uri"]
    mlflow.set_tracking_uri(uri)
    cliente = MlflowClient(tracking_uri=uri)
    nombre = cfg["registro"]["nombre_modelo"]
    alias = cfg["registro"].get("alias", "champion")
    # Evita reemplazar un champion existente sin indicarlo explícitamente.
    try:
        actual = cliente.get_model_version_by_alias(nombre, alias)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        actual = None
    if actual is not None and not reemplazar:
        print(f"Ya existe {nombre}@{alias}, versión {actual.version}; no se modificó.")
        return str(actual.version)

    carpeta = resolver_ruta(artefactos)
    with (carpeta / "metadatos.json").open(encoding="utf-8") as archivo:
        metadatos = json.load(archivo)
    if metadatos["versiones"]["scikit-learn"] != version("scikit-learn"):
        raise ValueError("Usa la misma versión de scikit-learn que figura en los metadatos.")
    if metadatos["features"] != FEATURES or metadatos["horizonte_horas"] != 24:
        raise ValueError("El modelo no corresponde al pipeline del día siguiente.")
    ruta_datos = resolver_ruta(cfg["datos"]["ruta"])
    if hashlib.sha256(ruta_datos.read_bytes()).hexdigest() != metadatos["dataset_sha256"]:
        raise ValueError("El dataset actual es distinto al usado en el entrenamiento guardado.")

    metricas = pd.read_csv(carpeta / "metricas_diarias.csv", index_col=0)
    candidato = metricas.loc[list(cfg["candidatos"])].mae.idxmin()
    # Solo se cargan los artefactos de entrenamiento propios del proyecto.
    modelo = joblib.load(carpeta / f"{candidato}.joblib")
    if list(modelo.feature_names_in_) != FEATURES:
        raise ValueError("Las columnas del modelo no coinciden con la API.")
    ejemplo = pd.read_csv(carpeta / "features_train.csv")[FEATURES].dropna().head(3)
    if ejemplo.empty:
        raise ValueError("No hay filas completas para documentar la firma del modelo.")
    pred = modelo.predict(ejemplo)
    mlflow.set_experiment(cfg["experimento_mlflow"])
    with mlflow.start_run(run_name=f"registro-champion-{candidato}") as run:
        mlflow.log_params({"modelo": candidato, "horizonte_horas": 24,
                           "origen": "artefacto entrenado; sin reentrenamiento",
                           **cfg["candidatos"][candidato]})
        mlflow.set_tags({"dataset_sha256": metadatos["dataset_sha256"],
                         "postprocesamiento": "maximum(pred, 0)",
                         "criterio": "menor MAE diario entre los tres regresores"})
        for escala in ["diarias", "horarias"]:
            tabla = pd.read_csv(carpeta / f"metricas_{escala}.csv", index_col=0)
            mlflow.log_metrics({f"{escala}_{m}": float(tabla.loc[candidato, m])
                                for m in ["mae", "rmse", "r2"]})
        for archivo in ["metadatos.json", "config.json", "metricas_diarias.csv", "metricas_horarias.csv"]:
            mlflow.log_artifact(str(carpeta / archivo), artifact_path="resultados")
        info = mlflow.sklearn.log_model(
            modelo, name="model", input_example=ejemplo,
            signature=infer_signature(ejemplo, pred),
            skops_trusted_types=["sklearn.tree._tree.Tree", "numpy.dtype"],
        )
        registrada = mlflow.register_model(info.model_uri, nombre)
        # Verifica la versión registrada antes de mover el alias.
        recuperado = mlflow.sklearn.load_model(f"models:/{nombre}/{registrada.version}")
        np.testing.assert_allclose(recuperado.predict(ejemplo), pred)
        cliente.set_model_version_tag(nombre, registrada.version, "modelo", candidato)
        cliente.set_registered_model_alias(nombre, alias, registrada.version)
        print(json.dumps({"modelo_registrado": nombre, "alias": alias,
                          "version": str(registrada.version), "candidato": candidato,
                          "run_id": run.info.run_id, "tracking_uri": uri}, indent=2))
    return str(registrada.version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artefactos", default=(
        "notebook/artifacts/02_Artifacts_prediction_by_next_day/prefect_runs/validacion_pipeline"))
    parser.add_argument("--reemplazar", action="store_true", help="Permite reemplazar un alias ya existente")
    args = parser.parse_args()
    registrar_champion(args.artefactos, args.reemplazar)
