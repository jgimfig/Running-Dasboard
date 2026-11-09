# Running Dashboard

Dashboard interactivo para analizar entrenamientos de running a partir de ficheros `.fit` (Garmin / WorkOutDoors).

## Estructura del proyecto

```
Running/
├── fit_raw/          ← Ficheros .fit de Garmin (42 actividades)
├── assets/
│   └── custom.css   ← Estilos personalizados del dashboard
├── app.py            ← Aplicación Dash (servidor web)
├── data_loader.py    ← Parser de ficheros .fit → DataFrames
├── requirements.txt  ← Dependencias Python
├── start.sh          ← Script de arranque (crea venv + lanza)
└── .venv/            ← Entorno virtual (se crea automáticamente)
```

## Requisitos

- **Python 3.9+** (recomendado 3.11+)
- **macOS / Linux / Windows** con terminal

## Instalación y ejecución

### Opción A — Script automático (recomendado)

```bash
cd /ruta/al/directorio/Running
chmod +x start.sh
./start.sh
```

El script:
1. Crea el entorno virtual `.venv` si no existe
2. Instala / actualiza todas las dependencias
3. Lanza el servidor en `http://localhost:8050`

### Opción B — Manual

```bash
cd /ruta/al/directorio/Running

# Crear entorno virtual
python3 -m venv .venv

# Activar entorno
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# Instalar dependencias
pip install -r requirements.txt

# Lanzar la aplicación
python app.py
```

Abre el navegador en **http://localhost:8050**

### Opción C — Docker

```bash
cd /ruta/al/directorio/Running
docker compose up -d --build
```

Abre **http://localhost:8050**. Comandos útiles:

```bash
docker compose logs -f        # ver logs
docker compose restart        # recargar datos tras añadir actividades
docker compose down           # parar
```

- La imagen **solo contiene el código**. La carpeta `fit_raw/` (FIT + `apple_health_export/`) se monta en
  `/data` en modo solo lectura, así que los datos personales nunca quedan dentro de la imagen.
- El primer arranque tarda ~35 s porque parsea el XML de Apple Health; el resultado se guarda en el volumen
  `health-cache` y los siguientes arranques tardan ~10 s.
- Se sirve con `gunicorn` (1 worker, 4 threads). Variables de entorno: `DATA_DIR` (por defecto `/data`)
  y `CACHE_DIR` (por defecto `/cache`).
- Para llevarlo a otra máquina: copia `app.py`, `data_loader.py`, `assets/`, `requirements.txt`,
  `Dockerfile`, `docker-compose.yml`, `.dockerignore` y tu carpeta `fit_raw/`, y ejecuta el mismo comando.

## Dependencias

Versiones fijadas en `requirements.txt` (probadas con Python 3.13):

| Paquete | Versión | Para qué se usa |
|---------|---------|-----------------|
| `fitparse` | 1.2.0 | Leer ficheros `.fit` de Garmin |
| `pandas` | 3.0.6 | Manipulación de datos |
| `numpy` | 2.5.4 | Cálculos numéricos |
| `dash` | 4.4.1 | Servidor web del dashboard |
| `plotly` | 7.1.0 | Gráficas interactivas |
| `dash-bootstrap-components` | 2.0.4 | Tema oscuro y componentes UI |
| `gunicorn` | 23.0.0 | Servidor WSGI en Docker |

## Funcionalidades del dashboard

### Filtros (barra superior)
- **Fecha**: por defecto selecciona el **mes anterior y el actual**. Arrastra los manejadores para cambiar el
  período; todos los tabs se actualizan al soltar.
- **Distancia**: rango en km.
- **Etiquetas**: selección múltiple (incluye «Sin etiqueta»); muestra las sesiones con alguna de ellas. Vacío = todas.

### Tabs

| Tab | Contenido |
|-----|-----------|
| **Resumen** | KPIs totales, volumen semanal, km acumulados, ritmo por actividad |
| **Progresión** | Tendencias de ritmo, FC, cadencia y eficiencia aeróbica a lo largo del tiempo |
| **Frecuencia Cardíaca** | Distribución por zonas Z1-Z5, FC por actividad, deriva de FC entre laps |
| **Dinámica de Carrera** | Cadencia, oscilación vertical, tiempo de apoyo, potencia |
| **Detalle Actividad** | Selecciona una sesión: tabla de laps, FC/ritmo por distancia, altimetría, potencia |
| **Etiquetas** | Asigna una o varias etiquetas a cada sesión (Entreno, Series, Carrera, Trail, Tirada larga, Recuperación) y crea etiquetas propias |

### Etiquetas
Cada sesión puede tener **varias** etiquetas (p. ej. «Trail» + «Tirada larga»). El filtro muestra las sesiones que tengan
**cualquiera** de las etiquetas elegidas. Se guardan en `activity_tags.json` (en Docker, en el volumen `/cache`,
variable `TAGS_FILE` para cambiar la ruta), así que sobreviven a reinicios y reconstrucciones de la imagen.
La tabla de edición ignora el filtro de etiquetas para que las filas no desaparezcan mientras editas.

## Añadir nuevas actividades

1. Exporta el fichero `.fit` desde Garmin Connect o WorkOutDoors
2. Cópialo en la carpeta `fit_raw/`
3. Reinicia el servidor (`./start.sh`)

## Notas técnicas

- Las zonas de FC se calculan automáticamente sobre la **FC máxima real** registrada en los datos
- **Limpieza de outliers** al cargar: la FC fuera de 30–230 bpm (p. ej. 0 = sensor sin señal) se descarta y los
  ritmos más lentos de 12 min/km (paradas/caminata) se ignoran, de modo que no afectan a medias, gráficas ni zonas
  (`PACE_MAX_MIN_KM` y `HR_VALID_RANGE` en `data_loader.py`)
- Todas las gráficas muestran la fecha de la sesión al pasar el ratón
- La cadencia se multiplica por 2 (Garmin registra pasos por pierna, no totales)
- El slider de fecha solo dispara el recálculo al **soltar** el manejador (`updatemode="mouseup"`)
- Los datos se cargan en memoria al arrancar (~120K registros GPS para 42 actividades)
