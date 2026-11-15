# API de rentas de bicicletas

La API carga al iniciar el modelo de MLflow indicado en `registro.nombre_modelo`,
resuelve su alias `champion` y conserva esa versión hasta reiniciar el proceso.
El run del modelo debe contener `resultados/metadatos.json`, generado por el
pipeline de entrenamiento: la API usa su origen de fecha para reconstruir las
mismas variables. No carga un archivo joblib local como sustituto del champion.

## Preparación

1. Levantar MLflow desde `Services/docker-compose.yml`.
2. Registrar el mejor regresor de los artefactos guardados, sin reentrenar:
   `poetry run python -m Infraestructure.flows.registrar_champion`.
   Se crea `seoul-bike-next-day` y se asigna el alias `champion` después de
   comprobar que el modelo registrado conserva sus predicciones. Si ya existe
   el alias, se conserva; `--reemplazar` permite actualizarlo expresamente.

El registro usa por defecto los artefactos de `prefect_runs/validacion_pipeline`,
generados con el entorno de Poetry. `--artefactos` permite indicar otra ejecución
que incluya sus metadatos, métricas, features y modelos. Para entrenar de nuevo
puede ejecutarse `poetry run python -m Infraestructure.flows.entrenamiento_flow`.
Con la configuración actual registra los modelos y promueve automáticamente el
mejor candidato si mejora al champion sobre la misma evaluación. También puede
gestionar las versiones y aliases desde la interfaz de MLflow.

## Ejecutar desde la raíz del proyecto

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
poetry run uvicorn Infraestructure.api.main:app --host 127.0.0.1 --port 8000
```

- `GET /health`: estado, versión y run del modelo cargado.
- `POST /predict`: 24 predicciones horarias y total del día siguiente.
- `/docs`: documentación interactiva y esquema de las peticiones.

```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post `
  -ContentType "application/json" -Body '{"fecha":"2018-12-01"}'
```

También puede enviarse `{}`: se pronostica el día siguiente al último día del CSV
(01/12/2018 con el dataset actual, no el día siguiente a la fecha del servidor).
La fecha solicitada debe coincidir con ese día y la víspera debe estar completa.

Opcionalmente, `historial` admite una lista de registros con `timestamp` y
`alquileres`, que reemplaza al CSV solo para esa petición. Los timestamps deben
ser horas exactas sin zona horaria, sin duplicados ni huecos de calendario.
Se requiere como mínimo la víspera completa; hasta siete días previos aportan
los rezagos semanales. Los alquileres cero representan registros ausentes.
El historial enviado no se escribe ni se usa para reentrenar el modelo.

Las variables se construyen solo con días anteriores al objetivo; no se requiere
clima futuro. Se aplica la imputación aprendida por el pipeline y se recortan las
predicciones negativas a cero. Las rentas estimadas son valores esperados, no
conteos enteros garantizados ni intervalos de confianza.

`SEOUL_BIKE_CONFIG` permite indicar otro YAML. Si cambia `champion`, hay que
reiniciar la API para cargar la nueva versión. Si no existe el alias, MLflow no
está disponible o faltan sus metadatos, el inicio falla con el error correspondiente.

Referencias: [inicio y cierre de FastAPI](https://fastapi.tiangolo.com/advanced/events/)
y [versiones y aliases de MLflow](https://www.mlflow.org/docs/latest/ml/model-registry/workflow).
