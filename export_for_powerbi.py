"""
Script de exportacion de datos enriquecidos para Power BI.

Ejecuta el pipeline completo de datos, aplica clustering y genera
recomendaciones para cada estudiante, luego exporta cuatro archivos CSV
listos para conectar como fuente de datos en Power BI Desktop.

Archivos generados en la carpeta powerbi_output/:
    1. pbi_estudiantes.csv       : estudiantes con cluster, riesgo y metricas
    2. pbi_recomendaciones.csv   : top 5 recomendaciones por estudiante
    3. pbi_estadisticas.csv      : resumen estadistico por cluster y facultad
    4. pbi_materias.csv          : catalogo de materias con metadatos

Uso:
    python export_for_powerbi.py
"""

import sys
import os
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.ingestion import load_all_data
from pipeline.preprocessing import run_preprocessing
from pipeline.feature_engineering import create_features
from models.clustering import (
    train_clustering,
    assign_cluster_labels,
    get_cluster_statistics,
    CLUSTER_CONFIG
)
from models.recommender import generate_recommendations

OUTPUT_DIR = Path(__file__).parent / "powerbi_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def export_estudiantes(df: pd.DataFrame) -> None:
    """
    Exporta el dataset de estudiantes enriquecido con cluster, riesgo y avance.
    Este es el archivo principal de Power BI con una fila por estudiante.
    """
    columnas = [
        "student_id", "nombre_completo", "cedula", "correo",
        "facultad_codigo", "facultad_nombre", "carrera",
        "anio_ingreso", "semestre_actual",
        "promedio_general", "asistencia_promedio",
        "creditos_aprobados", "creditos_reprobados",
        "tasa_reprobacion", "avance_carrera", "intensidad_academica",
        "estado_academico", "riesgo_score",
        "cluster_label", "cluster_color",
        "n_materias_cursadas", "n_aprobadas", "n_reprobadas"
    ]
    columnas_existentes = [c for c in columnas if c in df.columns]
    df_export = df[columnas_existentes].copy()

    df_export["tasa_reprobacion_pct"] = (df_export["tasa_reprobacion"] * 100).round(1)
    df_export["avance_carrera_pct"]   = df_export["avance_carrera"].round(1)
    df_export["promedio_general"]     = df_export["promedio_general"].round(1)
    df_export["asistencia_promedio"]  = df_export["asistencia_promedio"].round(1)
    df_export["riesgo_score"]         = df_export["riesgo_score"].round(1)

    # Agregar columna de semaforo de riesgo para filtros rapidos en Power BI
    def semaforo(row):
        if row["cluster_label"] == "Riesgo Critico":
            return "Critico"
        elif row["cluster_label"] == "En Riesgo":
            return "Alto"
        elif row["cluster_label"] == "Rendimiento Regular":
            return "Medio"
        else:
            return "Bajo"

    df_export["nivel_riesgo"] = df_export.apply(semaforo, axis=1)

    path = OUTPUT_DIR / "pbi_estudiantes.csv"
    df_export.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  pbi_estudiantes.csv exportado: {len(df_export)} filas -> {path}")


def export_recomendaciones(
    df: pd.DataFrame,
    courses: pd.DataFrame,
    grades: pd.DataFrame
) -> None:
    """
    Exporta las top 5 recomendaciones de materias para cada estudiante.
    Genera una fila por recomendacion (relacion N:5 con pbi_estudiantes.csv).
    """
    registros = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 50 == 0:
            print(f"    Generando recomendaciones: {i}/{total} estudiantes...")

        recs = generate_recommendations(
            row["student_id"], df, courses, grades, top_n=5
        )
        if recs.empty:
            continue

        for pos, (_, rec) in enumerate(recs.iterrows(), 1):
            registros.append({
                "student_id":             row["student_id"],
                "nombre_estudiante":      row["nombre_completo"],
                "facultad_codigo":        row["facultad_codigo"],
                "cluster_label":          row["cluster_label"],
                "posicion":               pos,
                "course_id":              rec["course_id"],
                "codigo_materia":         rec["codigo"],
                "nombre_materia":         rec["nombre"],
                "area_conocimiento":      rec["area_conocimiento"],
                "semestre_recomendado":   rec["semestre_recomendado"],
                "creditos":               rec["creditos"],
                "nivel_dificultad":       rec["nivel_dificultad"],
                "puntaje_compatibilidad": round(rec["puntaje_recomendacion"] * 100, 1),
                "razon":                  rec["razon"],
                "descripcion":            rec["descripcion"]
            })

    df_recs = pd.DataFrame(registros)
    path = OUTPUT_DIR / "pbi_recomendaciones.csv"
    df_recs.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  pbi_recomendaciones.csv exportado: {len(df_recs)} filas -> {path}")


def export_estadisticas(df: pd.DataFrame) -> None:
    """
    Exporta dos tablas de resumen para las visualizaciones de Power BI:
    estadisticas por cluster y estadisticas por facultad.
    """
    stats_cluster = get_cluster_statistics(df)
    stats_cluster["tasa_reprobacion_pct"] = (stats_cluster["tasa_reprobacion"] * 100).round(1)
    stats_cluster["descripcion"] = stats_cluster["cluster_label"].map(
        {k: v["descripcion"] for k, v in CLUSTER_CONFIG.items()}
    )
    stats_cluster["color"] = stats_cluster["cluster_label"].map(
        {k: v["color"] for k, v in CLUSTER_CONFIG.items()}
    )

    path_cluster = OUTPUT_DIR / "pbi_estadisticas_cluster.csv"
    stats_cluster.to_csv(path_cluster, index=False, encoding="utf-8-sig")
    print(f"  pbi_estadisticas_cluster.csv exportado: {len(stats_cluster)} filas")

    stats_facultad = df.groupby(["facultad_codigo", "facultad_nombre", "cluster_label"]).agg(
        n_estudiantes=("student_id", "count"),
        promedio_general=("promedio_general", "mean"),
        asistencia_promedio=("asistencia_promedio", "mean"),
        riesgo_score=("riesgo_score", "mean"),
        tasa_reprobacion=("tasa_reprobacion", "mean")
    ).round(2).reset_index()

    path_fac = OUTPUT_DIR / "pbi_estadisticas_facultad.csv"
    stats_facultad.to_csv(path_fac, index=False, encoding="utf-8-sig")
    print(f"  pbi_estadisticas_facultad.csv exportado: {len(stats_facultad)} filas")


def export_materias(courses: pd.DataFrame) -> None:
    """
    Exporta el catalogo de materias como tabla de dimension para Power BI.
    """
    path = OUTPUT_DIR / "pbi_materias.csv"
    courses.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  pbi_materias.csv exportado: {len(courses)} filas -> {path}")


def export_notas_resumen(grades: pd.DataFrame, courses: pd.DataFrame) -> None:
    """
    Exporta un resumen de notas por materia con tasa de aprobacion y promedio.
    Util para visualizar cuales materias tienen mayor indice de reprobacion.
    """
    resumen = grades.groupby("course_id").agg(
        total_intentos=("student_id", "count"),
        aprobados=("aprobado", "sum"),
        nota_promedio=("nota", "mean"),
        nota_min=("nota", "min"),
        nota_max=("nota", "max")
    ).reset_index()

    resumen["tasa_aprobacion_pct"] = (resumen["aprobados"] / resumen["total_intentos"] * 100).round(1)
    resumen["nota_promedio"] = resumen["nota_promedio"].round(1)

    resumen = resumen.merge(
        courses[["course_id", "codigo", "nombre", "facultad",
                 "area_conocimiento", "nivel_dificultad", "semestre_recomendado"]],
        on="course_id",
        how="left"
    )

    path = OUTPUT_DIR / "pbi_notas_por_materia.csv"
    resumen.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  pbi_notas_por_materia.csv exportado: {len(resumen)} filas -> {path}")


def main():
    print("Iniciando exportacion de datos para Power BI...")
    print()

    print("Paso 1: Cargando y procesando datos del pipeline...")
    raw       = load_all_data()
    processed = run_preprocessing(raw)
    df        = create_features(processed["merged"])
    courses   = processed["courses"]
    grades    = processed["grades"]
    print(f"  {len(df)} estudiantes cargados y preprocesados.")

    print()
    print("Paso 2: Ejecutando modelo de clustering (KMeans k=4)...")
    result = train_clustering(df)
    df     = assign_cluster_labels(df, result)
    print(f"  Clustering completado. Silhouette score: {result['silhouette']}")
    for label, count in df["cluster_label"].value_counts().items():
        print(f"    {label}: {count} estudiantes")

    print()
    print("Paso 3: Exportando archivos CSV para Power BI...")
    export_estudiantes(df)
    export_materias(courses)
    export_notas_resumen(grades, courses)
    export_estadisticas(df)

    print()
    print("Paso 4: Generando tabla de recomendaciones (puede tardar 1-2 minutos)...")
    export_recomendaciones(df, courses, grades)

    print()
    print("Exportacion completada.")
    print(f"Archivos listos en: {OUTPUT_DIR.resolve()}")
    print()
    print("Archivos generados para Power BI:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s} {size_kb:7.1f} KB")
    print()
    print("Siguiente paso: Abre Power BI Desktop y conecta cada CSV")
    print("  como fuente de datos (Obtener datos > Texto/CSV).")


if __name__ == "__main__":
    main()
