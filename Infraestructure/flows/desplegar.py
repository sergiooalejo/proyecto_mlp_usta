"""Despliega los flows: poetry run python -m Infraestructure.flows.desplegar

El proceso debe permanecer activo para ejecutar los trabajos programados.
"""

import argparse
import os

import yaml
from prefect import serve
from prefect.schedules import Cron
from prefect.settings import PREFECT_API_URL, temporary_settings

from Infraestructure.flows.entrenamiento_flow import entrenamiento_flow
from Infraestructure.flows.flow_pipeline_prefect import CONFIG, resolver_ruta
from Infraestructure.flows.inferencia_flow import inferencia_flow


def desplegar(config=str(CONFIG)):
    ruta = str(resolver_ruta(config))
    with open(ruta, encoding="utf-8") as archivo:
        cfg = yaml.safe_load(archivo)
    despliegue = cfg["despliegue"]
    url = os.environ.get("PREFECT_API_URL") or despliegue["prefect_api_url"]
    with temporary_settings({PREFECT_API_URL: url}):
        entrenamiento = entrenamiento_flow.to_deployment(
            name=despliegue["nombre_entrenamiento"], parameters={"config": ruta},
            concurrency_limit=1, tags=["seoul-bike", "entrenamiento"],
        )
        inferencia = inferencia_flow.to_deployment(
            name=despliegue["nombre_inferencia"], parameters={"config": ruta},
            schedules=[Cron(despliegue["cron"], timezone=despliegue["timezone"])],
            concurrency_limit=1, tags=["seoul-bike", "inferencia"],
        )
        # Entrenamiento manual; inferencia diaria. Un solo trabajo a la vez.
        serve(entrenamiento, inferencia, limit=1, pause_on_shutdown=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    desplegar(parser.parse_args().config)
