"""
Modulo de generacion de informes academicos con LLM.
Utiliza la API gratuita de Groq con el modelo llama-3.1-8b-instant
para producir reportes personalizados de cada estudiante.

Para activar este modulo, el usuario debe:
    1. Crear una cuenta gratuita en https://console.groq.com
    2. Generar una API key.
    3. Agregarla al archivo .env como GROQ_API_KEY=tu_clave_aqui.
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "openai/gpt-oss-20b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """
Eres un asesor academico experto de la Universidad Tecnologica de Panama (UTP).
Tu funcion es generar informes academicos claros, profesionales y empáticos
para ayudar a docentes a entender el estado de sus estudiantes y tomar decisiones.

Reglas para generar el informe:
- Usa un tono profesional pero accesible.
- Organiza el informe en secciones con titulos en mayuscula.
- Sé especifico con los datos provistos, no inventes informacion.
- Para estudiantes en riesgo, propone acciones concretas y alcanzables.
- El informe debe tener entre 300 y 450 palabras.
- No uses markdown con asteriscos, solo titulos en mayuscula y parrafos.
- Escribe en español de Panama.
""".strip()


def is_api_configured() -> bool:
    """
    Verifica si la API key de Groq esta configurada en el entorno.

    Returns:
        True si la clave existe y no esta vacia.
    """
    return bool(GROQ_API_KEY and GROQ_API_KEY.strip())


def build_student_context(
    student_row: pd.Series,
    recommendations: pd.DataFrame
) -> str:
    """
    Construye el contexto estructurado del estudiante para enviarlo al LLM.

    Args:
        student_row:     Fila del DataFrame de estudiantes con features y cluster.
        recommendations: DataFrame con las materias recomendadas.

    Returns:
        Cadena de texto con toda la informacion del estudiante formateada.
    """
    nombre = student_row.get("nombre_completo", "Estudiante")
    facultad = student_row.get("facultad_nombre", "N/A")
    carrera = student_row.get("carrera", "N/A")
    semestre = student_row.get("semestre_actual", "N/A")
    gpa = student_row.get("promedio_general", 0)
    asistencia = student_row.get("asistencia_promedio", 0)
    tasa_rep = student_row.get("tasa_reprobacion", 0)
    avance = student_row.get("avance_carrera", 0)
    creditos_aprobados = student_row.get("creditos_aprobados", 0)
    creditos_reprobados = student_row.get("creditos_reprobados", 0)
    estado = student_row.get("estado_academico", "N/A")
    cluster = student_row.get("cluster_label", "N/A")
    riesgo_score = student_row.get("riesgo_score", 0)

    lineas_recomendaciones = []
    if recommendations is not None and not recommendations.empty:
        for _, rec in recommendations.iterrows():
            lineas_recomendaciones.append(
                f"  - {rec['nombre']} ({rec['codigo']}) | "
                f"Creditos: {rec['creditos']} | "
                f"Dificultad: {rec['nivel_dificultad']}/5 | "
                f"Razon: {rec['razon']}"
            )
    recomendaciones_texto = "\n".join(lineas_recomendaciones) if lineas_recomendaciones else "No disponibles."

    contexto = f"""
DATOS DEL ESTUDIANTE
Nombre            : {nombre}
Facultad          : {facultad}
Carrera           : {carrera}
Semestre actual   : {semestre}
Estado academico  : {estado}
Perfil de cluster : {cluster}
Score de riesgo   : {riesgo_score}/100

INDICADORES ACADEMICOS
Promedio general     : {gpa:.1f} / 100
Asistencia promedio  : {asistencia:.1f}%
Tasa de reprobacion  : {tasa_rep*100:.1f}%
Avance en carrera    : {avance:.1f}%
Creditos aprobados   : {creditos_aprobados}
Creditos reprobados  : {creditos_reprobados}

MATERIAS RECOMENDADAS PARA EL PROXIMO SEMESTRE
{recomendaciones_texto}

INSTRUCCION
Genera un informe academico completo para este estudiante dirigido al docente asesor.
Incluye: resumen del perfil, evaluacion del riesgo academico, fortalezas detectadas,
areas de mejora, y un plan de accion concreto con las materias recomendadas.
""".strip()

    return contexto


def generate_student_report(
    student_row: pd.Series,
    recommendations: pd.DataFrame
) -> str:
    """
    Genera un informe academico individual usando el LLM de Groq.

    Args:
        student_row:     Fila del DataFrame de estudiantes.
        recommendations: DataFrame con las materias recomendadas para el estudiante.

    Returns:
        Texto del informe generado. Si la API no esta configurada, retorna
        un mensaje de instruccion para el usuario.
    """
    if not is_api_configured():
        return (
            "INFORME NO DISPONIBLE\n\n"
            "Para activar la generacion de informes con IA, configura tu API key gratuita de Groq:\n\n"
            "1. Crea una cuenta gratuita en https://console.groq.com\n"
            "2. Ve a 'API Keys' y genera una nueva clave.\n"
            "3. Copia la clave en el archivo .env del proyecto:\n"
            "   GROQ_API_KEY=tu_clave_aqui\n"
            "4. Reinicia la aplicacion Streamlit.\n\n"
            "La API de Groq es completamente gratuita con limites generosos."
        )

    try:
        import requests

        contexto = build_student_context(student_row, recommendations)

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": contexto}
            ],
            "temperature": 0.6,
            "max_tokens": 700
        }

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json"
        }

        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        informe = data["choices"][0]["message"]["content"].strip()
        return informe

    except Exception as e:
        return (
            f"ERROR AL GENERAR EL INFORME\n\n"
            f"Se produjo un error al contactar la API de Groq: {str(e)}\n\n"
            "Verifica que tu GROQ_API_KEY sea valida y que tengas conexion a internet."
        )


def build_cohort_context(stats_df: pd.DataFrame, n_students: int, n_en_riesgo: int) -> str:
    """
    Construye el contexto de grupo para el informe de cohorte.

    Args:
        stats_df:     DataFrame de estadisticas por cluster.
        n_students:   Total de estudiantes en el grupo.
        n_en_riesgo:  Estudiantes con estado En Riesgo o Riesgo Critico.

    Returns:
        Cadena de texto con el resumen del grupo.
    """
    lineas = []
    for _, row in stats_df.iterrows():
        lineas.append(
            f"  Perfil '{row['cluster_label']}': {row['n_estudiantes']} estudiantes | "
            f"GPA promedio {row['promedio_general']:.1f} | "
            f"Asistencia {row['asistencia_promedio']:.1f}% | "
            f"Reprobacion {row['tasa_reprobacion']*100:.1f}%"
        )

    contexto = f"""
RESUMEN DE COHORTE
Total de estudiantes  : {n_students}
Estudiantes en riesgo : {n_en_riesgo} ({n_en_riesgo/max(n_students,1)*100:.1f}%)

DISTRIBUCION POR PERFILES
{chr(10).join(lineas)}

INSTRUCCION
Genera un informe ejecutivo del estado academico general de esta cohorte.
Incluye: evaluacion global, grupos que requieren intervencion prioritaria,
tendencias preocupantes y recomendaciones estrategicas para el cuerpo docente.
""".strip()

    return contexto


def generate_cohort_report(
    stats_df: pd.DataFrame,
    n_students: int,
    n_en_riesgo: int
) -> str:
    """
    Genera un informe ejecutivo de cohorte usando el LLM de Groq.

    Args:
        stats_df:    DataFrame con estadisticas por cluster.
        n_students:  Total de estudiantes.
        n_en_riesgo: Cantidad en situacion de riesgo.

    Returns:
        Texto del informe de cohorte.
    """
    if not is_api_configured():
        return (
            "INFORME DE COHORTE NO DISPONIBLE\n\n"
            "Configura tu GROQ_API_KEY en el archivo .env para activar esta funcionalidad."
        )

    try:
        import requests

        contexto = build_cohort_context(stats_df, n_students, n_en_riesgo)

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": contexto}
            ],
            "temperature": 0.5,
            "max_tokens": 600
        }

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json"
        }

        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"ERROR AL GENERAR EL INFORME DE COHORTE: {str(e)}"
