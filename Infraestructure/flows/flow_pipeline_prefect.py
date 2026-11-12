"""Entrena y evalúa pronósticos de las 24 horas del día siguiente.

Ejecución desde la raíz del proyecto:
    poetry run python Infraestructure/flows/flow_pipeline_prefect.py
    poetry run python Infraestructure/flows/flow_pipeline_prefect.py --mlflow

PREFECT_API_URL permite conectar el flow al servidor de Prefect.
MLFLOW_TRACKING_URI se utiliza cuando se habilita el registro con --mlflow.
"""

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import yaml
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
CONFIG = RAIZ / "config/config.yaml"
ARTEFACTOS = RAIZ / "notebook/artifacts/02_Artifacts_prediction_by_next_day"
CATEGORICAS = ["dia_semana", "hora"]
NUMERICAS = [
    "dias_desde_inicio", "fecha_sin", "fecha_cos",
    "fin_semana_hora_sin", "fin_semana_hora_cos",
    "lag_24h", "lag_48h", "lag_168h",
    "media_horaria_dia_anterior", "media_horaria_7dias_previos",
]
FEATURES = CATEGORICAS + NUMERICAS


def resolver_ruta(ruta):
    ruta = Path(ruta)
    return ruta.resolve() if ruta.is_absolute() else (RAIZ / ruta).resolve()


def guardar_json(ruta, contenido):
    with Path(ruta).open("w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, ensure_ascii=False, indent=2, allow_nan=False)


def crear_features(timestamps, historial, origen_fecha):
    """Construye entradas usando solo registros anteriores al día objetivo."""
    t = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    historia = historial.sort_index()
    if historia.empty or not historia.index.is_unique:
        raise ValueError("El historial debe contener timestamps únicos.")
    x = pd.DataFrame(index=t.index)
    x["dia_semana"] = t.dt.dayofweek.astype(int)
    x["hora"] = t.dt.hour.astype(int)
    x["dias_desde_inicio"] = (t.dt.normalize() - origen_fecha).dt.days
    angulo = 2 * np.pi * (t.dt.dayofyear - 1) / 365.25
    x["fecha_sin"], x["fecha_cos"] = np.sin(angulo), np.cos(angulo)
    fin_semana = t.dt.dayofweek.ge(5).astype(int)
    x["fin_semana_hora_sin"] = fin_semana * np.sin(2 * np.pi * t.dt.hour / 24)
    x["fin_semana_hora_cos"] = fin_semana * np.cos(2 * np.pi * t.dt.hour / 24)
    for horas in [24, 48, 168]:
        fuentes = pd.DatetimeIndex(t - pd.Timedelta(hours=horas))
        if not (fuentes < pd.DatetimeIndex(t.dt.normalize())).all():
            raise ValueError("Un rezago utiliza información del día objetivo.")
        x[f"lag_{horas}h"] = historia.reindex(fuentes).to_numpy()
    total_diario = historia.resample("D").sum(min_count=24)
    dias = pd.DatetimeIndex(t.dt.normalize())
    x["media_horaria_dia_anterior"] = (
        total_diario.reindex(dias - pd.Timedelta(days=1)).to_numpy() / 24
    )
    calendario = pd.date_range(
        historia.index.min().normalize(),
        max(historia.index.max().normalize(), dias.max()), freq="D",
    )
    promedio = total_diario.reindex(calendario).shift(1).rolling(7, min_periods=3).mean() / 24
    x["media_horaria_7dias_previos"] = promedio.reindex(dias).to_numpy()
    return x[FEATURES]


def calcular_metricas(real, pred):
    return {
        "mae": float(mean_absolute_error(real, pred)),
        "rmse": float(np.sqrt(mean_squared_error(real, pred))),
        "r2": float(r2_score(real, pred)),
    }


@task
def cargar_configuracion(ruta):
    with Path(ruta).open(encoding="utf-8") as archivo:
        # safe_load también permite reutilizar las configuraciones JSON anteriores.
        cfg = yaml.safe_load(archivo)
    if not isinstance(cfg, dict):
        raise ValueError("La configuración debe contener un mapa de parámetros.")
    if not 0 < cfg["train_size"] < 1 or not np.isclose(cfg["train_size"] + cfg["test_size"], 1):
        raise ValueError("train_size y test_size deben sumar 1 y estar entre 0 y 1.")
    if cfg["horizonte_horas"] != 24 or cfg["objetivo"] != "alquileres":
        raise ValueError("Se requiere horizonte de 24 horas y objetivo alquileres.")
    if set(cfg["candidatos"]) != {"regresion_lineal", "random_forest", "gradient_boosting"}:
        raise ValueError("Se requieren los tres candidatos de regresión.")
    for nombre, parametros in cfg["candidatos"].items():
        if not isinstance(parametros, dict):
            raise ValueError(f"Los parámetros de {nombre} deben ser un mapa.")
        if nombre != "regresion_lineal":
            parametros.setdefault("random_state", cfg["semilla"])
    return cfg


@task
def cargar_datos(ruta):
    ruta = Path(ruta)
    huella = hashlib.sha256(ruta.read_bytes()).hexdigest()
    original = pd.read_csv(ruta, encoding="utf-8")
    df = original[["Date", "Hour", "Rented Bike Count"]].rename(columns={
        "Date": "fecha", "Hour": "hora", "Rented Bike Count": "alquileres",
    }).copy()
    df["fecha"] = pd.to_datetime(df.fecha, format="%d/%m/%Y", errors="raise")
    if df.isna().any().any():
        raise ValueError("Hay valores faltantes en las columnas requeridas.")
    if not (df.hora.between(0, 23) & df.hora.mod(1).eq(0)).all():
        raise ValueError("Las horas deben ser enteros entre 0 y 23.")
    if not (np.isfinite(df.alquileres) & df.alquileres.ge(0) & df.alquileres.mod(1).eq(0)).all():
        raise ValueError("Los alquileres deben ser conteos enteros no negativos.")
    df["timestamp"] = df.fecha + pd.to_timedelta(df.hora, unit="h")
    df = df.sort_values("timestamp").reset_index(drop=True)
    esperadas = pd.date_range(df.timestamp.min(), df.timestamp.max(), freq="h")
    if not df.timestamp.is_unique or not pd.DatetimeIndex(df.timestamp).equals(esperadas):
        raise ValueError("El calendario contiene duplicados o huecos horarios.")
    if not df.groupby("fecha").size().eq(24).all():
        raise ValueError("El calendario debe incluir días de 24 horas.")
    # Los ceros se conservan en el archivo, pero representan registros ausentes.
    df["registro_ausente"] = df.alquileres.eq(0)
    print(f"Datos: {len(df)} horas; ausencias: {df.registro_ausente.sum()}; SHA256: {huella}")
    return df, huella


@task
def preparar_datos(df, cfg):
    dias = pd.Index(df.fecha.drop_duplicates())
    posicion = int(len(dias) * cfg["train_size"])
    if not 0 < posicion < len(dias):
        raise ValueError("No hay días suficientes para dividir el calendario.")
    corte = dias[posicion]
    train_calendario = df.loc[df.fecha < corte].copy()
    test = df.loc[df.fecha >= corte].copy().reset_index(drop=True)
    train = train_calendario.loc[~train_calendario.registro_ausente]
    historial = df.set_index("timestamp").alquileres.mask(df.set_index("timestamp").registro_ausente)
    origen = df.fecha.min()
    x_train = crear_features(train.timestamp, historial, origen)
    y_train = train.alquileres.reset_index(drop=True)
    x_test = crear_features(test.timestamp, historial, origen)
    if len(y_train) < 2 or (~test.registro_ausente).sum() < 2:
        raise ValueError("No hay etiquetas observadas suficientes para entrenar y evaluar.")
    # Verifica que las entradas no cambian al retirar el día objetivo y el futuro.
    for dia in [corte, df.fecha.max()]:
        horas = pd.date_range(dia, periods=24, freq="h")
        pd.testing.assert_frame_equal(
            crear_features(horas, historial, origen),
            crear_features(horas, historial.loc[historial.index < dia], origen),
        )
    particion = pd.DataFrame([
        {"conjunto": nombre, "desde": datos.timestamp.min(), "hasta": datos.timestamp.max(),
         "dias": datos.fecha.nunique(), "horas_calendario": len(datos),
         "porcentaje_calendario": 100 * len(datos) / len(df),
         "etiquetas_observadas": int((~datos.registro_ausente).sum())}
        for nombre, datos in [("train", train_calendario), ("test", test)]
    ])
    print(particion.to_string(index=False))
    return {"train": train, "test": test, "x_train": x_train, "y_train": y_train,
            "x_test": x_test, "historial": historial, "origen": origen,
            "corte": corte, "particion": particion}


@task
def entrenar_candidato(nombre, parametros, datos):
    estimadores = {"regresion_lineal": LinearRegression,
                   "random_forest": RandomForestRegressor,
                   "gradient_boosting": GradientBoostingRegressor}
    preprocesado = ColumnTransformer([
        ("num", Pipeline([
            ("imputacion", SimpleImputer(strategy="median", add_indicator=True)),
            ("escalado", StandardScaler()),
        ]), NUMERICAS),
        ("cat", OneHotEncoder(categories=[list(range(7)), list(range(24))],
                              drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ])
    modelo = Pipeline([("preprocesado", preprocesado), ("modelo", estimadores[nombre](**parametros))])
    inicio = time.perf_counter()
    modelo.fit(datos["x_train"], datos["y_train"])
    bruto = modelo.predict(datos["x_test"])
    if not np.isfinite(bruto).all():
        raise ValueError(f"Predicciones no finitas: {nombre}")
    print(f"Entrenado: {nombre}")
    return {"nombre": nombre, "modelo": modelo, "pred": np.maximum(bruto, 0),
            "bruto": bruto, "segundos_fit_y_predict": time.perf_counter() - inicio}


@task
def evaluar_candidatos(datos, resultados):
    evaluacion = datos["test"].copy()
    evaluacion["dia_semana"] = evaluacion.fecha.dt.dayofweek
    media = float(datos["y_train"].mean())
    media_grupo = datos["train"].assign(dia_semana=datos["train"].fecha.dt.dayofweek).groupby(
        ["dia_semana", "hora"]
    ).alquileres.mean()
    indice = pd.MultiIndex.from_frame(evaluacion[["dia_semana", "hora"]])
    referencias = {
        "referencia_global": np.full(len(evaluacion), media),
        "referencia_dia_hora": media_grupo.reindex(indice).fillna(media).to_numpy(),
    }
    for nombre, columna in [("referencia_ayer", "lag_24h"), ("referencia_semana_anterior", "lag_168h")]:
        referencias[nombre] = datos["x_test"][columna].fillna(
            pd.Series(referencias["referencia_dia_hora"])
        ).to_numpy()
    predicciones = {**referencias, **{r["nombre"]: r["pred"] for r in resultados}}
    observado = ~evaluacion.registro_ausente
    filas = []
    diario = evaluacion.groupby("fecha").agg(
        total_archivo=("alquileres", "sum"),
        horas_observadas=("registro_ausente", lambda s: (~s).sum()),
    )
    diario["dia_completo_observado"] = diario.horas_observadas.eq(24)
    if diario.dia_completo_observado.sum() < 2:
        raise ValueError("Se necesitan al menos dos días completos para medir R² diario.")
    for nombre, pred in predicciones.items():
        evaluacion[nombre] = pred
        diario[nombre] = evaluacion.groupby("fecha")[nombre].sum()
        filas.append({"modelo": nombre, **calcular_metricas(evaluacion.loc[observado, "alquileres"], pred[observado])})
    for resultado in resultados:
        evaluacion[f"{resultado['nombre']}_sin_recorte"] = resultado["bruto"]
    completos = diario.loc[diario.dia_completo_observado]
    metricas_diarias = pd.DataFrame({
        nombre: calcular_metricas(completos.total_archivo, completos[nombre])
        for nombre in predicciones
    }).T.sort_values("mae")
    metricas_horarias = pd.DataFrame(filas).set_index("modelo").sort_values("mae")
    print("Métricas diarias:\n", metricas_diarias.to_string())
    return {"metricas_horarias": metricas_horarias, "metricas_diarias": metricas_diarias,
            "predicciones_horarias": evaluacion, "predicciones_diarias": diario}


@task
def guardar_artefactos(salidas, cfg, datos, resultados, evaluacion, huella):
    salidas = Path(salidas)
    salidas.mkdir(parents=True, exist_ok=True)
    (salidas / "flow_pipeline_prefect.py").write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    guardar_json(salidas / "config.json", cfg)
    guardar_json(salidas / "metadatos.json", {
        "dataset_sha256": huella, "origen_fecha": datos["origen"].isoformat(),
        "fecha_corte": datos["corte"].isoformat(), "features": FEATURES,
        "numericas": NUMERICAS, "categoricas": CATEGORICAS, "horizonte_horas": 24,
        "origen_pronostico": "00:00; historial disponible hasta las 23:00 de la víspera",
        "evaluacion": "Modelos fijos; historial observado actualizado al avanzar cada día",
        "postprocesamiento": "Recortar predicciones negativas a cero",
        "python": platform.python_version(),
        "versiones": {p: version(p) for p in ["pandas", "numpy", "scikit-learn", "prefect", "joblib"]},
    })
    for nombre, tabla in {**evaluacion, "particion": datos["particion"],
                          "features_train": datos["x_train"], "features_test": datos["x_test"]}.items():
        tabla.to_csv(salidas / f"{nombre}.csv", index=True)
    datos["historial"].rename("alquileres_observados").to_csv(
        salidas / "historial_observado.csv", index_label="timestamp")
    futuro = pd.date_range(datos["test"].fecha.max() + pd.Timedelta(days=1), periods=24, freq="h")
    x_futuro = crear_features(futuro, datos["historial"], datos["origen"])
    pronostico = pd.DataFrame({"timestamp": futuro})
    for resultado in resultados:
        nombre, modelo = resultado["nombre"], resultado["modelo"]
        ruta = salidas / f"{nombre}.joblib"
        joblib.dump(modelo, ruta)
        np.testing.assert_allclose(joblib.load(ruta).predict(datos["x_test"].head(24)), resultado["bruto"][:24])
        pronostico[nombre] = np.maximum(modelo.predict(x_futuro), 0)
    pronostico.to_csv(salidas / "pronostico_dia_siguiente.csv", index=False)
    pronostico.drop(columns="timestamp").sum().rename("rentas_estimadas").to_csv(
        salidas / "totales_dia_siguiente.csv")
    return str(salidas)


@task
def registrar_mlflow(resultados, datos, evaluacion, cfg, salidas, huella, tracking_uri):
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["experimento_mlflow"])
    runs = {}
    for resultado in resultados:
        nombre, modelo = resultado["nombre"], resultado["modelo"]
        with mlflow.start_run(run_name=nombre) as run:
            mlflow.log_params({"modelo": nombre, "train_size": cfg["train_size"],
                               "horizonte_horas": 24, "fecha_corte": str(datos["corte"]),
                               **cfg["candidatos"][nombre]})
            mlflow.set_tags({"dataset_sha256": huella, "features": ",".join(FEATURES),
                             "postprocesamiento": "maximum(pred, 0)", "fase": "entrenamiento"})
            for escala in ["horarias", "diarias"]:
                mlflow.log_metrics({f"{escala}_{k}": float(v) for k, v in
                                    evaluacion[f"metricas_{escala}"].loc[nombre].items()})
            mlflow.log_artifacts(salidas, artifact_path="resultados")
            ejemplo = datos["x_train"].dropna().head(3)
            info = mlflow.sklearn.log_model(
                modelo, name="model", input_example=ejemplo,
                signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
            runs[nombre] = {"run_id": run.info.run_id, "model_uri": info.model_uri}
    guardar_json(Path(salidas) / "runs_mlflow.json", runs)
    return runs


@flow(name="entrenamiento-seoul-bike-dia-siguiente", log_prints=True)
def flow_pipeline_prefect(
    config=str(CONFIG), datos=None,
    salidas=None, mlflow_habilitado=None, tracking_uri=None,
):
    cfg = cargar_configuracion(resolver_ruta(config))
    if mlflow_habilitado is not None:
        cfg["registrar_mlflow"] = mlflow_habilitado
    ruta_datos = resolver_ruta(datos if datos is not None else cfg.get("datos", {}).get("ruta", "data/SeoulBikeData.csv"))
    if salidas is None:
        identificador = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        base_salidas = resolver_ruta(cfg.get("artefactos", {}).get("carpeta", ARTEFACTOS / "prefect_runs"))
        salidas = base_salidas / identificador
    else:
        salidas = resolver_ruta(salidas)
    if Path(salidas).exists() and any(Path(salidas).iterdir()):
        raise ValueError("La carpeta de salida debe estar vacía para conservar ejecuciones previas.")
    df, huella = cargar_datos(ruta_datos)
    preparados = preparar_datos(df, cfg)
    resultados = [entrenar_candidato(nombre, parametros, preparados)
                  for nombre, parametros in cfg["candidatos"].items()]
    evaluacion = evaluar_candidatos(preparados, resultados)
    carpeta = guardar_artefactos(salidas, cfg, preparados, resultados, evaluacion, huella)
    runs = {}
    if cfg.get("registrar_mlflow", False):
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or cfg.get("mlflow_tracking_uri", "http://localhost:5000")
        runs = registrar_mlflow(resultados, preparados, evaluacion, cfg, carpeta, huella, uri)
    candidatos = evaluacion["metricas_diarias"].loc[list(cfg["candidatos"])].sort_values("mae")
    mejor = candidatos.index[0]
    referencia_mae = float(evaluacion["metricas_diarias"].loc["referencia_ayer", "mae"])
    resumen = {
        "mejor_modelo": mejor, "mae_diario": float(candidatos.loc[mejor, "mae"]),
        "r2_diario": float(candidatos.loc[mejor, "r2"]),
        "mae_referencia_ayer": referencia_mae,
        "supera_referencia_ayer_mae": bool(candidatos.loc[mejor, "mae"] < referencia_mae),
        "artefactos": carpeta, "runs_mlflow": runs,
    }
    guardar_json(Path(carpeta) / "resumen.json", resumen)
    create_markdown_artifact(
        key="comparacion-seoul-bike-dia-siguiente",
        markdown=(f"# Pronóstico del día siguiente\n\nMejor candidato: **{mejor}**\n\n"
                  f"MAE diario: {resumen['mae_diario']:.2f}; R²: {resumen['r2_diario']:.3f}\n\n"
                  f"MAE de repetir ayer: {referencia_mae:.2f}\n\nArtefactos: `{carpeta}`"),
        description="Comparación temporal de tres regresores y referencias históricas",
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    return resumen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--datos", default=None, help="Sobrescribe la ruta del dataset del YAML")
    parser.add_argument("--salidas", default=None)
    parser.add_argument("--mlflow", action="store_true", default=None, help="Registrar runs y modelos en MLflow")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    flow_pipeline_prefect(config=args.config, datos=args.datos, salidas=args.salidas,
                          mlflow_habilitado=args.mlflow, tracking_uri=args.tracking_uri)
