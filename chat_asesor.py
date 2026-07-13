"""
Chat interactivo del Asesor Academico UTP.
Permite hacer preguntas en lenguaje natural sobre sus estudiantes,
materias y estadisticas academicas. El LLM recibe el contexto completo del
pipeline para responder con datos reales.

Uso:
    python chat_asesor.py
"""

import sys
import os
import json
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.ingestion import load_all_data
from pipeline.preprocessing import run_preprocessing
from pipeline.feature_engineering import create_features
from models.clustering import train_clustering, assign_cluster_labels

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "openai/gpt-oss-20b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT_TEMPLATE = """
Eres el Asesor Academico Inteligente de la Universidad Tecnologica de Panama (UTP).
Tienes acceso al contexto completo de datos academicos del semestre actual y debes
responder preguntas del usuario de forma clara, precisa y profesional.

CONTEXTO DE DATOS ACTUALES:
{contexto}

REGLAS:
- Responde siempre en español de Panama.
- Usa los datos provistos, no inventes informacion.
- Se conciso pero completo, maximo 200 palabras por respuesta.
- Si el usuario pregunta por un estudiante especifico busca su nombre en los datos.
- Si no encuentras la informacion solicitada dilo claramente.
- Puedes hacer calculos simples sobre los datos provistos.
- Mantén un tono profesional y empatico.
""".strip()


def build_context(df: pd.DataFrame, courses: pd.DataFrame, grades: pd.DataFrame) -> str:
    """
    Construye el contexto de datos que el LLM usara para responder preguntas.
    Resume la informacion clave del pipeline para no exceder el limite de tokens.

    Args:
        df:      DataFrame de estudiantes con clusters asignados.
        courses: DataFrame del catalogo de materias.
        grades:  DataFrame de historial de notas.

    Returns:
        Cadena de texto con el resumen del contexto academico.
    """
    total = len(df)
    en_riesgo = len(df[df["cluster_label"].isin(["En Riesgo", "Riesgo Critico"])])
    criticos  = len(df[df["cluster_label"] == "Riesgo Critico"])
    gpa_prom  = df["promedio_general"].mean()
    asist_prom = df["asistencia_promedio"].mean()

    dist_cluster = df["cluster_label"].value_counts().to_dict()
    dist_facultad = df["facultad_codigo"].value_counts().to_dict()

    top_riesgo = df.nlargest(10, "riesgo_score")[
        ["nombre_completo", "facultad_codigo", "carrera",
         "semestre_actual", "promedio_general", "asistencia_promedio",
         "riesgo_score", "cluster_label", "estado_academico"]
    ].to_dict(orient="records")

    notas_materia = grades.groupby("course_id").agg(
        tasa_aprobacion=("aprobado", "mean"),
        nota_promedio=("nota", "mean"),
        total=("student_id", "count")
    ).reset_index()
    notas_materia = notas_materia.merge(
        courses[["course_id", "nombre", "facultad", "area_conocimiento"]],
        on="course_id", how="left"
    )
    materias_dificiles = notas_materia.nsmallest(5, "tasa_aprobacion")[
        ["nombre", "facultad", "tasa_aprobacion", "nota_promedio"]
    ].to_dict(orient="records")

    lista_estudiantes = df[
        ["nombre_completo", "facultad_codigo", "carrera", "semestre_actual",
         "promedio_general", "asistencia_promedio", "tasa_reprobacion",
         "riesgo_score", "cluster_label", "estado_academico"]
    ].to_dict(orient="records")

    contexto = f"""
RESUMEN GENERAL
Total de estudiantes     : {total}
Estudiantes en riesgo    : {en_riesgo} ({en_riesgo/total*100:.1f}%)
Riesgo critico           : {criticos}
GPA promedio general     : {gpa_prom:.1f}
Asistencia promedio      : {asist_prom:.1f}%

DISTRIBUCION POR PERFIL
{json.dumps(dist_cluster, ensure_ascii=False, indent=2)}

DISTRIBUCION POR FACULTAD
{json.dumps(dist_facultad, ensure_ascii=False, indent=2)}

TOP 10 ESTUDIANTES CON MAYOR RIESGO
{json.dumps(top_riesgo, ensure_ascii=False, indent=2)}

MATERIAS CON MAYOR TASA DE REPROBACION
{json.dumps(materias_dificiles, ensure_ascii=False, indent=2)}

LISTA COMPLETA DE ESTUDIANTES
{json.dumps(lista_estudiantes, ensure_ascii=False, indent=2)}
""".strip()

    return contexto


def ask_llm(pregunta: str, historial: list, system_prompt: str) -> str:
    """
    Envia la pregunta del usuario al LLM de Groq con el historial de conversacion.

    Args:
        pregunta:      Pregunta actual del usuario.
        historial:     Lista de mensajes anteriores de la conversacion.
        system_prompt: Prompt de sistema con el contexto de datos.

    Returns:
        Respuesta del LLM como cadena de texto.
    """
    mensajes = [{"role": "system", "content": system_prompt}]
    mensajes += historial
    mensajes.append({"role": "user", "content": pregunta})

    payload = {
        "model": GROQ_MODEL,
        "messages": mensajes,
        "temperature": 0.5,
        "max_tokens": 500
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json"
    }

    response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def print_bienvenida():
    """Imprime el mensaje de bienvenida del chat."""
    print()
    print("=" * 60)
    print("  ASESOR ACADEMICO UTP  ")
    print("  Sistema de Consulta Inteligente para Docentes")
    print("=" * 60)
    print()
    print("Puedes preguntarme sobre tus estudiantes, por ejemplo:")
    print("  - Cuales son los estudiantes en mayor riesgo de FISC?")
    print("  - Dame un resumen de Carlos Pimentel")
    print("  - Que materia tiene mayor tasa de reprobacion?")
    print("  - Cuantos estudiantes de FIEM estan en riesgo critico?")
    print("  - Que perfil predomina en el semestre 3?")
    print()
    print("Escribe 'salir' para terminar la sesion.")
    print("-" * 60)
    print()


def main():
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY no configurada en el archivo .env")
        print("Obtén tu clave gratuita en https://console.groq.com")
        sys.exit(1)

    print("Cargando datos del pipeline...")

    raw       = load_all_data()
    processed = run_preprocessing(raw)
    df        = create_features(processed["merged"])
    courses   = processed["courses"]
    grades    = processed["grades"]
    result    = train_clustering(df)
    df        = assign_cluster_labels(df, result)

    contexto      = build_context(df, courses, grades)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(contexto=contexto)

    print("Datos cargados correctamente.")
    print_bienvenida()

    historial = []

    while True:
        try:
            pregunta = input("Usuario: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSesion finalizada.")
            break

        if not pregunta:
            continue

        if pregunta.lower() in ["salir", "exit", "quit", "bye"]:
            print("\nSesion finalizada. Hasta luego.")
            break

        print()
        print("Asesor: ", end="", flush=True)

        try:
            respuesta = ask_llm(pregunta, historial, system_prompt)
            print(respuesta)

            historial.append({"role": "user",      "content": pregunta})
            historial.append({"role": "assistant",  "content": respuesta})

            # Mantener solo los ultimos 10 turnos para no exceder el limite de tokens
            if len(historial) > 20:
                historial = historial[-20:]

        except Exception as e:
            print(f"Error al contactar la API: {str(e)}")

        print()
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()