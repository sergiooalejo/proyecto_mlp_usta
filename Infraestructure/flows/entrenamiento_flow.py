"""Entrenamiento: poetry run python -m Infraestructure.flows.entrenamiento_flow"""

import argparse
import os

import yaml
from prefect.settings import PREFECT_API_URL, temporary_settings

from Infraestructure.flows.flow_pipeline_prefect import (
    CONFIG, flow_pipeline_prefect as entrenamiento_flow, resolver_ruta,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args()
    with resolver_ruta(args.config).open(encoding="utf-8") as archivo:
        cfg = yaml.safe_load(archivo)
    url = os.environ.get("PREFECT_API_URL") or cfg["despliegue"]["prefect_api_url"]
    with temporary_settings({PREFECT_API_URL: url}):
        entrenamiento_flow(config=args.config)
