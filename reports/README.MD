# Sistema de Recomendacion Academica UTP

Sistema inteligente de gestion y analisis de datos estudiantiles para la Universidad Tecnologica de Panama.
Integra un pipeline de datos en Python con clustering KMeans, un motor de recomendacion hibrido,
generacion de informes con IA via Groq y un dashboard interactivo construido en Power BI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KMeans-orange)
![LLM](https://img.shields.io/badge/LLM-Groq%20openai%2Fgpt--oss--20b-green)
![License](https://img.shields.io/badge/Licencia-MIT-lightgrey)


## Tabla de Contenidos

1. [Descripcion del Proyecto](#descripcion-del-proyecto)
2. [Problema que Resuelve](#problema-que-resuelve)
3. [Arquitectura del Sistema](#arquitectura-del-sistema)
4. [Funcionalidades del Dashboard](#funcionalidades-del-dashboard)
5. [Tecnologias Utilizadas](#tecnologias-utilizadas)
6. [Estructura del Repositorio](#estructura-del-repositorio)
7. [Instalacion y Configuracion](#instalacion-y-configuracion)
8. [Uso Paso a Paso](#uso-paso-a-paso)
9. [Datasets Simulados](#datasets-simulados)
10. [Modelos de Machine Learning](#modelos-de-machine-learning)
11. [API de LLM Gratuita](#api-de-llm-gratuita)
12. [Autores](#autores)


## Descripcion del Proyecto

Este proyecto implementa un sistema de recomendacion academica para docentes y asesores de la UTP.
El backend en Python se encarga de toda la logica de datos: ingesta de dos fuentes, limpieza,
feature engineering, clustering de perfiles estudiantiles, motor de recomendacion de materias
y generacion de informes personalizados con inteligencia artificial.

Power BI actua como capa de visualizacion, consumiendo los archivos CSV exportados por el pipeline
para mostrar un dashboard ejecutivo de cuatro paginas completamente interactivo.


## Problema que Resuelve

Los docentes de la UTP atienden a cientos de estudiantes por semestre sin contar con herramientas
que les permitan identificar rapidamente quienes estan en riesgo de desercion, que materias son
mas convenientes para cada perfil academico, ni obtener un informe detallado por estudiante sin
invertir horas de analisis manual.

Este sistema centraliza toda esa informacion y la convierte en decisiones accionables con un clic.


## Arquitectura del Sistema

```
Fuente 1: students_performance.csv         Fuente 2: courses_catalog.csv
          (rendimiento estudiantil)                   (catalogo UTP)
                   |                                        |
                   +------------- pipeline/ ----------------+
                                       |
                            ingestion.py          carga y validacion
                            preprocessing.py      limpieza y merge
                            feature_engineering   features para ML
                                       |
                                   models/
                            clustering.py         KMeans k=4
                            recommender.py        scoring hibrido
                                       |
                                   reports/
                            llm_reports.py        Groq API (gpt-oss-20b)
                                       |
                            export_for_powerbi.py     exporta 6 CSVs
                            generate_ai_reports.py    genera informes IA
                                       |
                               powerbi_output/
                            pbi_estudiantes.csv
                            pbi_recomendaciones.csv
                            pbi_informes_ia.csv
                            pbi_materias.csv
                            pbi_estadisticas_cluster.csv
                            pbi_estadisticas_facultad.csv
                                       |
                            Dashboard Power BI (.pbix)
```


## Funcionalidades del Dashboard

El dashboard de Power BI tiene cuatro paginas interactivas:

**Pagina 1: Panel General**
KPIs principales (total estudiantes, estudiantes en riesgo, GPA promedio, score de riesgo promedio),
grafico de dona con distribucion por perfil de cluster y grafico de barras por facultad.

**Pagina 2: Estudiantes en Riesgo**
Tabla filtrable con todos los estudiantes en perfiles En Riesgo y Riesgo Critico. Incluye
segmentaciones por cluster y facultad, y un grafico de barras ordenado por score de riesgo descendente.

**Pagina 3: Recomendaciones**
Segmentacion por nombre de estudiante que filtra dinamicamente una tabla con las 5 materias
recomendadas, incluyendo area, creditos, nivel de dificultad, puntaje de compatibilidad y razon.

**Pagina 4: Informes con IA**
Seleccion de estudiante que muestra el informe academico completo generado por Groq, con resumen
del perfil, evaluacion del riesgo, fortalezas detectadas, areas de mejora y plan de accion.


## Tecnologias Utilizadas

| Categoria          | Tecnologia                             |
|--------------------|----------------------------------------|
| Lenguaje           | Python 3.10+                           |
| Dashboard          | Power BI Desktop                       |
| Machine Learning   | scikit-learn (KMeans, Silhouette)      |
| Procesamiento      | Pandas, NumPy                          |
| LLM gratuito       | Groq API con openai/gpt-oss-20b        |
| Variables entorno  | python-dotenv                          |
| Visualizacion      | Power BI con temas de colores UTP      |


## Estructura del Repositorio

```
utp-academic-system/
|
+-- export_for_powerbi.py           Pipeline completo + exportacion de CSVs
+-- generate_ai_reports.py          Generacion batch de informes con Groq
+-- requirements.txt                Dependencias del proyecto
+-- .env.example                    Plantilla de variables de entorno
+-- .gitignore
+-- README.md
|
+-- data/
|   +-- generate_data.py            Generador de datasets simulados UTP
|   +-- students_performance.csv    Fuente 1: rendimiento estudiantil
|   +-- courses_catalog.csv         Fuente 2: catalogo de 61 materias
|   +-- student_grades.csv          Fuente 3: historial de notas (5246 registros)
|
+-- pipeline/
|   +-- ingestion.py                Carga y validacion de las dos fuentes
|   +-- preprocessing.py           Limpieza, normalizacion y merge
|   +-- feature_engineering.py     Creacion de 5 features para ML
|
+-- models/
|   +-- clustering.py              KMeans k=4 con etiquetas semanticas automaticas
|   +-- recommender.py             Motor de recomendacion hibrido (4 componentes)
|
+-- reports/
|   +-- llm_reports.py             Generacion de informes con Groq API
|
+-- powerbi_output/                 CSVs listos para conectar en Power BI
    +-- pbi_estudiantes.csv
    +-- pbi_recomendaciones.csv
    +-- pbi_informes_ia.csv
    +-- pbi_materias.csv
    +-- pbi_estadisticas_cluster.csv
    +-- pbi_estadisticas_facultad.csv
```


## Instalacion y Configuracion

### Requisitos previos

- Python 3.10 o superior
- Power BI Desktop (descarga gratuita en microsoft.com/power-bi)
- Cuenta gratuita en console.groq.com para los informes con IA

### Pasos de instalacion

1. Clona el repositorio

```bash
git clone https://github.com/tu_usuario/utp-academic-system.git
cd utp-academic-system
```

2. Crea y activa un entorno virtual

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

3. Instala las dependencias

```bash
pip install -r requirements.txt
```

4. Configura la API key gratuita de Groq

```bash
cp .env.example .env
# Edita el archivo .env con tu API key de https://console.groq.com
# GROQ_API_KEY=tu_clave_aqui
```


## Uso Paso a Paso

**Paso 1: Genera los datasets simulados**

```bash
python data/generate_data.py
```

Crea tres CSVs con 300 estudiantes, 61 materias y 5246 registros de notas.

**Paso 2: Exporta los datos para Power BI**

```bash
python export_for_powerbi.py
```

Ejecuta el pipeline completo (ingesta, limpieza, clustering, recomendaciones) y exporta
seis archivos CSV enriquecidos en la carpeta powerbi_output/.

**Paso 3: Genera los informes de IA**

```bash
python generate_ai_reports.py
```

Llama a Groq por cada estudiante en riesgo y genera un informe academico personalizado.
Exporta pbi_informes_ia.csv en powerbi_output/.

Para probar con pocos estudiantes primero:

```bash
python generate_ai_reports.py --limite 10
```

**Paso 4: Conecta los CSVs en Power BI Desktop**

Abre Power BI Desktop y para cada archivo en powerbi_output/ usa:
Obtener datos > Texto/CSV

**Paso 5: Crea las relaciones en la vista Modelo**

```
pbi_estudiantes.student_id  →  pbi_recomendaciones.student_id  (1 a varios)
pbi_estudiantes.student_id  →  pbi_informes_ia.student_id      (1 a 1)
pbi_materias.course_id      →  pbi_notas_por_materia.course_id (1 a varios)
```

**Paso 6: Construye el dashboard**

Crea cuatro paginas en Power BI usando los CSVs como fuentes de cada visual.
Guarda el archivo como utp_academic_advisor.pbix dentro de la carpeta del proyecto.

**Para actualizar los datos:**

```bash
python export_for_powerbi.py
python generate_ai_reports.py
```

Luego en Power BI haz clic en Actualizar para reflejar los cambios.


## Datasets Simulados

**students_performance.csv (Fuente 1)**
300 estudiantes distribuidos en 6 facultades: FISC, FIC, FIE, FIEM, FIIA y FACIN.
Columnas principales: student_id, nombre_completo, facultad_codigo, carrera, semestre_actual,
promedio_general, asistencia_promedio, creditos_aprobados, tasa_reprobacion, estado_academico.

**courses_catalog.csv (Fuente 2)**
61 materias entre comunes (primer y segundo ano) y especificas por facultad.
Columnas principales: course_id, codigo, nombre, facultad, semestre_recomendado, creditos,
area_conocimiento, nivel_dificultad, prerequisitos_ids.

**student_grades.csv (Fuente 3)**
5246 registros de historial de notas por estudiante y materia.
Columnas: student_id, course_id, nota, aprobado, num_intentos.


## Modelos de Machine Learning

### Clustering con KMeans (k=4)

Features utilizadas y normalizadas con StandardScaler:

| Feature              | Descripcion                                      |
|----------------------|--------------------------------------------------|
| promedio_general     | GPA del estudiante (0 a 100)                     |
| asistencia_promedio  | Porcentaje de asistencia (0 a 100)               |
| tasa_reprobacion     | Proporcion de materias reprobadas (0 a 1)        |
| avance_carrera       | Creditos aprobados sobre creditos esperados      |
| intensidad_academica | Materias cursadas por semestre en promedio       |

Los clusters se etiquetan automaticamente ordenando centroides por promedio_general:

| Perfil              | GPA Tipico | Asistencia | Descripcion                           |
|---------------------|------------|------------|---------------------------------------|
| Alto Rendimiento    | 85 a 100   | 90 a 100%  | Perfil excelente, candidatos avance   |
| Rendimiento Regular | 70 a 84    | 75 a 90%   | Perfil solido, requiere mantenimiento |
| En Riesgo           | 55 a 69    | 60 a 75%   | Alertas activas, atencion recomendada |
| Riesgo Critico      | 0 a 54     | 0 a 60%    | Intervencion inmediata requerida      |

Silhouette score obtenido: 0.3388 (por encima del umbral de 0.3 indica clusters bien definidos).

### Motor de Recomendacion Hibrido

Cada materia candidata recibe un puntaje calculado con cuatro componentes:

```
puntaje = 0.30 x afinidad_por_area
        + 0.25 x compatibilidad_dificultad
        + 0.25 x tasa_exito_en_cluster
        + 0.20 x compatibilidad_de_semestre
```

Filtros previos al scoring: la materia debe pertenecer a la facultad del estudiante o ser
comun, y todos sus prerequisitos deben estar aprobados.


## API de LLM Gratuita

Este proyecto usa la API de Groq con el modelo openai/gpt-oss-20b (reemplazo oficial
de llama-3.1-8b-instant segun deprecacion anunciada el 17 de junio de 2026).

Para obtener tu API key gratuita:

1. Ve a https://console.groq.com y crea una cuenta.
2. En el menu lateral haz clic en API Keys.
3. Crea una nueva clave y copiala.
4. Agrega al archivo .env: GROQ_API_KEY=tu_clave_aqui

Los limites del plan gratuito son suficientes para generar informes de todos los
estudiantes en riesgo de un semestre academico completo.


## Autores

Desarrollado como proyecto integrador del curso de Ciencia de Datos.
Universidad Tecnologica de Panama, Facultad de Ingenieria en Sistemas Computacionales.


## Licencia

MIT License. Libre para uso academico y educativo.