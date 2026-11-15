# Predicción de rentas de bicicletas en Seúl

El proyecto pronostica las 24 horas del día siguiente con regresión lineal,
Random Forest y Gradient Boosting. Usa calendario y demanda histórica; el CSV
original se conserva y sus conteos cero se consideran registros ausentes.

## Servicios y configuración

Desde la raíz del repositorio:

```powershell
poetry install
docker compose -f Services/docker-compose.yml up -d
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:PYTHONUTF8 = "1"
```

`config/config.yaml` define las rutas, los modelos, la división temporal 65/35,
el registro en MLflow y el horario de inferencia. Las rutas se resuelven desde
la raíz del repositorio. Las variables de entorno de conexión tienen prioridad.

## Entrenamiento y champion

```powershell
poetry run python -m Infraestructure.flows.entrenamiento_flow
```

El flow realiza ingesta, preparación, entrenamiento, evaluación y registro de
los tres modelos. El candidato con menor MAE diario puede recibir el alias
`champion` de `seoul-bike-next-day`. Si ya existe un champion comparable, solo
lo reemplaza cuando mejora el MAE; en un empate conserva la versión actual.
Si cambian el dataset o el corte de evaluación, deja pendiente la promoción
para revisión manual, evitando comparar métricas incompatibles.

La promoción está controlada por `registro.promover_automaticamente`. Si se
desactiva `registrar_mlflow`, solo se generan artefactos locales y no se promueve
un modelo. El script `registrar_champion` permite registrar artefactos propios
de una ejecución previa de forma manual.

## Inferencia diaria en Prefect

Ejecución manual de comprobación:

```powershell
poetry run python -m Infraestructure.flows.inferencia_flow
```

El flow obtiene el alias champion en cada ejecución, carga su versión exacta
y sus metadatos, ingesta el historial, predice las 24 horas y guarda CSV/JSON.
En MLflow registra el run de inferencia y su vínculo con la versión y el run de
entrenamiento. Las salidas quedan en `notebook/artifacts/inferencia/`.

Para registrar ambos despliegues y atender los trabajos:

```powershell
poetry run python -m Infraestructure.flows.desplegar
```

- Entrenamiento: despliegue manual, sin horario automático.
- Inferencia: una ejecución diaria a las **06:00 America/Bogota**, configurable
  mediante `despliegue.cron` y `despliegue.timezone`.

**El proceso `desplegar` y los servicios Docker deben permanecer activos.**
Cerrar el proceso pausa el horario; volver a ejecutarlo registra los mismos
despliegues y vuelve a atenderlos. No está instalado como servicio de Windows
ni arranca automáticamente después de reiniciar el equipo.

Los despliegues y ejecuciones se consultan en `http://localhost:4200`.
El servidor de Prefect por sí solo no ejecuta el código: necesita este proceso
servidor de flows. Véase [serve de Prefect](https://reference.prefect.io/prefect/flows/).

**Dataset histórico:** la fecha pronosticada es el día siguiente al último día
completo del CSV, no la fecha del reloj del equipo. Con el archivo actual siempre
será el 01/12/2018. Para pronosticar nuevos días deben incorporarse previamente
sus observaciones históricas al origen de datos. El horario diario no inventa
datos nuevos ni avanza recursivamente usando predicciones como si fueran registros.

## API con FastAPI

```powershell
poetry run uvicorn Infraestructure.api.main:app --host 127.0.0.1 --port 8000
```

- `GET /health`: versión cargada y estado.
- `POST /predict` con `{}`: 24 predicciones y total diario.
- `http://localhost:8000/docs`: documentación interactiva.

La API y la inferencia comparten el cargador de champion y la preparación de
variables. La API carga su versión al iniciar: después de una promoción debe
reiniciarse. La inferencia de Prefect consulta el alias en cada ejecución.
Más ejemplos en [Infraestructure/api/README.md](Infraestructure/api/README.md).

La carpeta `_actividad-2/` contiene únicamente material de referencia.
