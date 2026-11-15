"""Inferencia diaria: poetry run python -m Infraestructure.flows.inferencia_flow"""

import argparse
import json
import os
from datetime import datetime, timezone
from uuid import uuid4

import mlflow
import numpy as np
import pandas as pd
import yaml
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from prefect.settings import PREFECT_API_URL, temporary_settings

from Infraestructure.flows.flow_pipeline_prefect import (
    CONFIG, cargar_datos, crear_features, guardar_json, resolver_ruta,
)
from Infraestructure.servicios.champion import cargar_champion as leer_champion


@task
def cargar_configuracion(config):
    with resolver_ruta(config).open(encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


@task
def cargar_champion(cfg):
    champion = leer_champion(cfg)
    print(f"Champion: {champion['nombre']} v{champion['version']}; run {champion['run_id']}")
    return champion


@task
def ingesta(cfg, fecha=None):
    df, huella = cargar_datos.fn(resolver_ruta(cfg["datos"]["ruta"]))
    dia = df.fecha.max() + pd.Timedelta(days=1)
    if fecha is not None and pd.Timestamp(fecha) != dia:
        raise ValueError("Solo se pronostica el día siguiente al último día completo del CSV.")
    historial = df.set_index("timestamp").alquileres.mask(df.set_index("timestamp").registro_ausente)
    # El CSV histórico siempre produce el mismo día hasta que se ingesten datos nuevos.
    print(f"Último dato: {historial.index.max()}; día a predecir: {dia.date()}")
    return {"historial": historial, "fecha": dia, "dataset_sha256": huella}


@task
def predecir(champion, lote, cfg):
    if lote["fecha"] <= champion["origen"]:
        raise ValueError("El día objetivo debe ser posterior al origen del modelo.")
    timestamps = pd.date_range(lote["fecha"], periods=24, freq="h")
    entradas = crear_features(timestamps, lote["historial"], champion["origen"])
    pred = np.asarray(champion["modelo"].predict(entradas), dtype=float)
    if pred.shape != (24,) or not np.isfinite(pred).all():
        raise ValueError("El champion no devolvió 24 predicciones finitas.")
    pred = np.maximum(pred, 0)
    identificador = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    carpeta = resolver_ruta(cfg["inferencia"]["carpeta_salida"]) / identificador
    carpeta.mkdir(parents=True, exist_ok=False)
    resultado = pd.DataFrame({"timestamp": timestamps, "rentas_estimadas": pred,
                              "modelo": champion["nombre"], "modelo_version": champion["version"],
                              "run_id_entrenamiento": champion["run_id"]})
    resultado.to_csv(carpeta / "predicciones.csv", index=False)
    entradas.to_csv(carpeta / "features.csv", index=False)
    resumen = {"fecha": lote["fecha"].date().isoformat(), "horas": 24,
               "total_rentas_estimadas": float(pred.sum()), "modelo": champion["nombre"],
               "alias": champion["alias"], "version_modelo": champion["version"],
               "run_id_entrenamiento": champion["run_id"],
               "historial_hasta": lote["historial"].index.max().isoformat(),
               "dataset_sha256": lote["dataset_sha256"], "carpeta": str(carpeta)}
    guardar_json(carpeta / "resumen.json", resumen)
    return resumen


@task
def registrar_inferencia(resumen, cfg):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["experimento_mlflow"])
    with mlflow.start_run(run_name=f"inferencia-{resumen['fecha']}") as run:
        mlflow.set_tags({"fase": "inferencia", "modelo_nombre": resumen["modelo"],
                         "modelo_version": resumen["version_modelo"], "alias": resumen["alias"],
                         "modelo_run_id": resumen["run_id_entrenamiento"],
                         "dataset_sha256": resumen["dataset_sha256"]})
        mlflow.log_params({"fecha_objetivo": resumen["fecha"], "historial_hasta": resumen["historial_hasta"]})
        mlflow.log_metrics({"horas_predichas": resumen["horas"],
                            "total_rentas_estimadas": resumen["total_rentas_estimadas"]})
        mlflow.log_artifacts(resumen["carpeta"], artifact_path="inferencia")
        return run.info.run_id


@flow(name="inferencia-seoul-bike", log_prints=True)
def inferencia_flow(config=str(CONFIG), fecha=None):
    cfg = cargar_configuracion(config)
    champion = cargar_champion(cfg)
    lote = ingesta(cfg, fecha)
    resumen = predecir(champion, lote, cfg)
    resumen["run_id_inferencia"] = registrar_inferencia(resumen, cfg)
    guardar_json(resolver_ruta(resumen["carpeta"]) / "resumen.json", resumen)
    create_markdown_artifact(
        key="inferencia-seoul-bike",
        markdown=(f"# Pronóstico {resumen['fecha']}\n\n"
                  f"Champion v{resumen['version_modelo']}; 24 horas; "
                  f"total estimado: {resumen['total_rentas_estimadas']:.2f} rentas.\n\n"
                  f"Run de entrenamiento: `{resumen['run_id_entrenamiento']}`"),
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    return resumen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--fecha", default=None)
    args = parser.parse_args()
    with resolver_ruta(args.config).open(encoding="utf-8") as archivo:
        configuracion = yaml.safe_load(archivo)
    url = os.environ.get("PREFECT_API_URL") or configuracion["despliegue"]["prefect_api_url"]
    with temporary_settings({PREFECT_API_URL: url}):
        inferencia_flow(config=args.config, fecha=args.fecha)
