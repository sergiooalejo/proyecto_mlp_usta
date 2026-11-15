"""API de rentas de bicicletas para las 24 horas del día siguiente.

Desde la raíz del proyecto:
    poetry run uvicorn Infraestructure.api.main:app --host 127.0.0.1 --port 8000

Documentación: http://localhost:8000/docs
Requiere un modelo registrado con alias champion y los metadatos de su run.
El modelo se carga al iniciar; reiniciar la API después de cambiar el alias.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import date, datetime

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from Infraestructure.flows.flow_pipeline_prefect import (
    CONFIG, FEATURES, cargar_datos, crear_features, resolver_ruta,
)
from Infraestructure.servicios.champion import cargar_champion

LOGGER = logging.getLogger(__name__)


class RegistroHora(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    alquileres: int = Field(ge=0, strict=True, description="Cero representa un registro ausente.")

    @field_validator("timestamp")
    @classmethod
    def validar_hora(cls, valor):
        if valor.tzinfo is not None or valor.minute or valor.second or valor.microsecond:
            raise ValueError("Usa una hora exacta, sin zona horaria, como en SeoulBikeData.csv.")
        return valor


class SolicitudPrediccion(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"example": {"fecha": "2018-12-01"}},
    )

    fecha: date | None = Field(default=None, description="Día siguiente al último día del historial.")
    historial: list[RegistroHora] | None = Field(
        default=None, min_length=24, max_length=17520,
        description="Opcional: reemplaza el historial del CSV para esta petición; no se guarda.",
    )


class PrediccionHora(BaseModel):
    timestamp: datetime
    rentas_estimadas: float = Field(ge=0)


class RespuestaPrediccion(BaseModel):
    fecha: date
    dia_semana: str
    modelo: str
    alias: str
    version_modelo: str
    run_id_entrenamiento: str
    historial_hasta: datetime
    total_rentas_estimadas: float = Field(ge=0)
    predicciones: list[PrediccionHora] = Field(min_length=24, max_length=24)


@asynccontextmanager
async def lifespan(application):
    ruta = resolver_ruta(os.environ.get("SEOUL_BIKE_CONFIG", str(CONFIG)))
    with ruta.open(encoding="utf-8") as archivo:
        config = yaml.safe_load(archivo)
    application.state.config = config
    application.state.champion = cargar_champion(config)
    yield
    application.state.champion = None


app = FastAPI(
    title="API de rentas de bicicletas de Seúl",
    description="Pronóstico de las 24 horas del día siguiente con el modelo champion de MLflow.",
    version="1.0.0", lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    champion = request.app.state.champion
    return {"estado": "ok", "modelo": champion["nombre"], "alias": champion["alias"],
            "version": champion["version"], "run_id_entrenamiento": champion["run_id"],
            "horizonte_horas": 24}


def preparar_historial(solicitud, config):
    if solicitud.historial is None:
        try:
            # Ejecuta la función de lectura, sin lanzar tareas ni flows de Prefect.
            df, _ = cargar_datos.fn(resolver_ruta(config["datos"]["ruta"]))
        except (OSError, ValueError, KeyError) as exc:
            LOGGER.exception("No se pudo leer el historial del CSV.")
            raise HTTPException(status_code=503, detail="El CSV de historial no está disponible o no es válido.") from exc
        datos = df[["timestamp", "alquileres"]].copy()
    else:
        datos = pd.DataFrame([r.model_dump() for r in solicitud.historial]).sort_values("timestamp")
    datos = datos.set_index("timestamp").sort_index()
    indice = pd.DatetimeIndex(datos.index)
    if not indice.is_unique:
        raise HTTPException(status_code=422, detail="El historial contiene horas duplicadas.")
    esperadas = pd.date_range(indice.min(), indice.max(), freq="h")
    if not indice.equals(esperadas):
        raise HTTPException(status_code=422, detail="Faltan horas en el calendario; conserva las ausencias como cero.")
    siguiente_dia = indice.max().normalize() + pd.Timedelta(days=1)
    fecha = pd.Timestamp(solicitud.fecha) if solicitud.fecha is not None else siguiente_dia
    if fecha != siguiente_dia:
        raise HTTPException(status_code=422, detail="La fecha debe ser el día siguiente al último día del historial.")
    ayer = pd.date_range(fecha - pd.Timedelta(days=1), periods=24, freq="h")
    if not ayer.isin(indice).all():
        raise HTTPException(status_code=422, detail="Se requieren las 24 horas de la víspera antes de pronosticar.")
    # Los ceros son ausencias; el pipeline imputará los rezagos faltantes.
    historia = datos.alquileres.mask(datos.alquileres.eq(0))
    return fecha, historia


@app.post("/predict", response_model=RespuestaPrediccion)
def predict(solicitud: SolicitudPrediccion, request: Request):
    champion = request.app.state.champion
    fecha, historia = preparar_historial(solicitud, request.app.state.config)
    if fecha <= champion["origen"]:
        raise HTTPException(status_code=422, detail="La fecha debe ser posterior al origen del modelo.")
    timestamps = pd.date_range(fecha, periods=24, freq="h")
    try:
        entradas = crear_features(timestamps, historia, champion["origen"])
        predicciones = np.asarray(champion["modelo"].predict(entradas), dtype=float)
        if predicciones.shape != (24,) or not np.isfinite(predicciones).all():
            raise ValueError("El modelo no devolvió 24 predicciones finitas.")
        predicciones = np.maximum(predicciones, 0)
    except Exception as exc:
        LOGGER.exception("Falló la predicción del champion.")
        raise HTTPException(status_code=500, detail="No fue posible generar las 24 predicciones.") from exc
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    return {"fecha": fecha.date(), "dia_semana": dias[fecha.dayofweek],
            "modelo": champion["nombre"], "alias": champion["alias"],
            "version_modelo": champion["version"], "run_id_entrenamiento": champion["run_id"],
            "historial_hasta": historia.index.max().to_pydatetime(),
            "total_rentas_estimadas": float(predicciones.sum()),
            "predicciones": [{"timestamp": t.to_pydatetime(), "rentas_estimadas": float(p)}
                             for t, p in zip(timestamps, predicciones)]}
