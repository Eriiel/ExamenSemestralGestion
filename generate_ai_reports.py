"""
Generador batch de informes academicos con IA para Power BI.

Llama a la API de Groq para generar un informe personalizado por cada
estudiante en situacion de riesgo y exporta los resultados en un CSV
que Power BI puede mostrar en tarjetas o tablas de texto.

El archivo generado se llama pbi_informes_ia.csv y contiene:
    student_id, nombre, cluster_label, informe_ia, fecha_generacion

Uso:
    python generate_ai_reports.py                  (solo estudiantes en riesgo)
    python generate_ai_reports.py --todos          (todos los estudiantes)
    python generate_ai_reports.py --limite 20      (primeros N estudiantes en riesgo)

Requisito:
    GROQ_API_KEY configurada en el archivo .env del proyecto.
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from pipeline.ingestion import load_all_data
from pipeline.preprocessing import run_preprocessing
from pipeline.feature_engineering import create_features
from models.clustering import train_clustering, assign_cluster_labels
from models.recommender import generate_recommendations
from reports.llm_reports import (
    generate_student_report,
    is_api_configured,
    GROQ_API_KEY
)

OUTPUT_DIR = Path(__file__).parent / "powerbi_output"
OUTPUT_DIR.mkdir(exist_ok=True)

PERFILES_EN_RIESGO = {"En Riesgo", "Riesgo Critico"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera informes de IA para estudiantes y los exporta para Power BI."
    )
    parser.add_argument(
        "--todos",
        action="store_true",
        help="Genera informes para todos los estudiantes, no solo los de riesgo."
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Limita el numero de informes a generar (util para pruebas)."
    )
    return parser.parse_args()


def generar_informe_con_reintento(student_row, recommendations, max_intentos=3):
    """
    Intenta generar el informe con hasta max_intentos reintentos si falla la API.

    Args:
        student_row:     Fila del estudiante.
        recommendations: Recomendaciones generadas para ese estudiante.
        max_intentos:    Numero maximo de reintentos ante error.

    Returns:
        Texto del informe o mensaje de error.
    """
    for intento in range(1, max_intentos + 1):
        informe = generate_student_report(student_row, recommendations)
        if not informe.startswith("ERROR"):
            return informe
        if intento < max_intentos:
            espera = intento * 3
            print(f"    Reintento {intento}/{max_intentos} en {espera}s...")
            time.sleep(espera)
    return informe


def main():
    args = parse_args()

    print("Generador de Informes Academicos con IA para Power BI")
    print()

    if not is_api_configured():
        print("ERROR: GROQ_API_KEY no configurada.")
        print()
        print("Pasos para obtener tu API key gratuita:")
        print("  1. Ve a https://console.groq.com y crea una cuenta.")
        print("  2. En el menu lateral haz clic en API Keys.")
        print("  3. Crea una nueva clave y copiala.")
        print("  4. Agrega al archivo .env:  GROQ_API_KEY=tu_clave_aqui")
        print("  5. Vuelve a ejecutar este script.")
        sys.exit(1)

    print("API de Groq configurada correctamente.")
    print()

    print("Cargando y procesando datos del pipeline...")
    raw       = load_all_data()
    processed = run_preprocessing(raw)
    df        = create_features(processed["merged"])
    courses   = processed["courses"]
    grades    = processed["grades"]

    result = train_clustering(df)
    df     = assign_cluster_labels(df, result)
    print(f"  {len(df)} estudiantes cargados con clusters asignados.")

    if args.todos:
        df_objetivo = df.copy()
        print(f"  Modo: todos los estudiantes ({len(df_objetivo)} en total).")
    else:
        df_objetivo = df[df["cluster_label"].isin(PERFILES_EN_RIESGO)].copy()
        df_objetivo = df_objetivo.sort_values("riesgo_score", ascending=False)
        print(f"  Modo: solo estudiantes en riesgo ({len(df_objetivo)} encontrados).")

    if args.limite:
        df_objetivo = df_objetivo.head(args.limite)
        print(f"  Limite aplicado: {args.limite} estudiantes.")

    print()
    total = len(df_objetivo)
    print(f"Generando {total} informes con Groq (llama-3.1-8b-instant)...")
    print("Esto puede tomar varios minutos segun el limite de la API gratuita.")
    print()

    registros = []
    errores   = 0
    fecha_gen = datetime.now().strftime("%Y-%m-%d %H:%M")

    for i, (_, row) in enumerate(df_objetivo.iterrows(), 1):
        nombre = row["nombre_completo"]
        cluster = row["cluster_label"]
        print(f"  [{i:3d}/{total}] {nombre} ({cluster})...", end=" ", flush=True)

        recs = generate_recommendations(
            row["student_id"], df, courses, grades, top_n=5
        )

        informe = generar_informe_con_reintento(row, recs)

        if informe.startswith("ERROR"):
            errores += 1
            estado = "ERROR"
        else:
            estado = "OK"

        print(estado)

        registros.append({
            "student_id":       row["student_id"],
            "nombre_completo":  nombre,
            "facultad_codigo":  row["facultad_codigo"],
            "carrera":          row["carrera"],
            "semestre_actual":  row["semestre_actual"],
            "promedio_general": row["promedio_general"],
            "cluster_label":    cluster,
            "riesgo_score":     row["riesgo_score"],
            "estado_academico": row["estado_academico"],
            "informe_ia":       informe,
            "fecha_generacion": fecha_gen
        })

        # Pausa entre llamadas para respetar rate limits del plan gratuito
        if i < total:
            time.sleep(1.2)

    df_informes = pd.DataFrame(registros)
    path = OUTPUT_DIR / "pbi_informes_ia.csv"
    df_informes.to_csv(path, index=False, encoding="utf-8-sig")

    print()
    print("Resumen de generacion:")
    print(f"  Total informes generados : {len(registros)}")
    print(f"  Exitosos                 : {len(registros) - errores}")
    print(f"  Con error                : {errores}")
    print(f"  Archivo exportado        : {path.resolve()}")
    print()
    print("Conecta pbi_informes_ia.csv en Power BI como fuente de datos.")
    print("Usa un visual de Tabla o Tarjeta con el campo 'informe_ia' para mostrar el texto.")


if __name__ == "__main__":
    main()
