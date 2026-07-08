"""
Modulo de preprocesamiento de datos.
Limpia, normaliza y combina los datasets cargados por el modulo de ingesta.
"""

import pandas as pd
import numpy as np


def clean_students(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y normaliza el dataset de estudiantes.

    Operaciones aplicadas:
        - Elimina filas con valores nulos en columnas criticas.
        - Ajusta rangos fuera de limites (notas 0-100, asistencia 0-100).
        - Estandariza los nombres de estado academico.

    Args:
        df: DataFrame crudo de students_performance.csv.

    Returns:
        DataFrame limpio y normalizado.
    """
    df = df.copy()

    columnas_criticas = ["student_id", "facultad_codigo", "promedio_general", "asistencia_promedio"]
    df.dropna(subset=columnas_criticas, inplace=True)

    df["promedio_general"] = df["promedio_general"].clip(0, 100)
    df["asistencia_promedio"] = df["asistencia_promedio"].clip(0, 100)
    df["tasa_reprobacion"] = df["tasa_reprobacion"].clip(0, 1)
    df["riesgo_score"] = df["riesgo_score"].clip(0, 100)
    df["creditos_aprobados"] = df["creditos_aprobados"].clip(lower=0)
    df["creditos_reprobados"] = df["creditos_reprobados"].clip(lower=0)

    df["semestre_actual"] = df["semestre_actual"].clip(1, 10)

    df["student_id"] = df["student_id"].astype(str).str.strip()
    df["nombre_completo"] = df["nombre_completo"].astype(str).str.strip().str.title()
    df["estado_academico"] = df["estado_academico"].astype(str).str.strip()

    df.reset_index(drop=True, inplace=True)
    return df


def clean_courses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y normaliza el catalogo de materias.

    Operaciones aplicadas:
        - Elimina filas con codigo o nombre nulo.
        - Rellena valores nulos en prerequisitos con cadena vacia.
        - Valida rango de dificultad (1-5) y semestre (1-10).

    Args:
        df: DataFrame crudo de courses_catalog.csv.

    Returns:
        DataFrame del catalogo limpio y normalizado.
    """
    df = df.copy()

    df.dropna(subset=["course_id", "codigo", "nombre"], inplace=True)

    df["prerequisitos_ids"] = df["prerequisitos_ids"].fillna("").astype(str)
    df["prerequisitos_codigos"] = df["prerequisitos_codigos"].fillna("").astype(str)
    df["descripcion"] = df["descripcion"].fillna("Sin descripcion disponible.")

    df["nivel_dificultad"] = df["nivel_dificultad"].clip(1, 5)
    df["semestre_recomendado"] = df["semestre_recomendado"].clip(1, 10)
    df["creditos"] = df["creditos"].clip(1, 8)

    df["course_id"] = df["course_id"].astype(int)
    df["codigo"] = df["codigo"].astype(str).str.strip().str.upper()
    df["nombre"] = df["nombre"].astype(str).str.strip()
    df["facultad"] = df["facultad"].astype(str).str.strip().str.upper()
    df["area_conocimiento"] = df["area_conocimiento"].astype(str).str.strip()

    df.reset_index(drop=True, inplace=True)
    return df


def clean_grades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el historial de notas.

    Operaciones aplicadas:
        - Elimina registros sin student_id o course_id.
        - Corrige notas fuera del rango 0-100.
        - Recalcula el campo 'aprobado' para consistencia (nota >= 70).
        - Ajusta numero de intentos al rango 1-3.

    Args:
        df: DataFrame crudo de student_grades.csv.

    Returns:
        DataFrame de notas limpio.
    """
    df = df.copy()

    df.dropna(subset=["student_id", "course_id", "nota"], inplace=True)

    df["nota"] = df["nota"].clip(0, 100)

    df["aprobado"] = df["nota"] >= 70

    df["num_intentos"] = df["num_intentos"].clip(1, 3).astype(int)
    df["course_id"] = df["course_id"].astype(int)
    df["student_id"] = df["student_id"].astype(str).str.strip()

    df.reset_index(drop=True, inplace=True)
    return df


def build_student_course_summary(grades: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un resumen por estudiante con conteos y promedios de sus notas.

    Args:
        grades: DataFrame limpio de notas.

    Returns:
        DataFrame con metricas derivadas por student_id:
            - n_materias_cursadas
            - n_aprobadas
            - n_reprobadas
            - nota_promedio_historica
            - nota_max
            - nota_min
    """
    resumen = grades.groupby("student_id").agg(
        n_materias_cursadas=("course_id", "count"),
        n_aprobadas=("aprobado", "sum"),
        nota_promedio_historica=("nota", "mean"),
        nota_max=("nota", "max"),
        nota_min=("nota", "min"),
    ).reset_index()

    resumen["n_reprobadas"] = resumen["n_materias_cursadas"] - resumen["n_aprobadas"]
    resumen["nota_promedio_historica"] = resumen["nota_promedio_historica"].round(2)
    return resumen


def merge_all(
    students: pd.DataFrame,
    courses: pd.DataFrame,
    grades: pd.DataFrame
) -> pd.DataFrame:
    """
    Combina los tres datasets en un DataFrame enriquecido por estudiante.

    Args:
        students: DataFrame limpio de estudiantes.
        courses:  DataFrame limpio de materias.
        grades:   DataFrame limpio de notas.

    Returns:
        DataFrame enriquecido con metricas historicas por estudiante.
    """
    resumen = build_student_course_summary(grades)

    df_merged = students.merge(resumen, on="student_id", how="left")

    df_merged["n_materias_cursadas"] = df_merged["n_materias_cursadas"].fillna(0).astype(int)
    df_merged["n_aprobadas"] = df_merged["n_aprobadas"].fillna(0).astype(int)
    df_merged["n_reprobadas"] = df_merged["n_reprobadas"].fillna(0).astype(int)
    df_merged["nota_promedio_historica"] = df_merged["nota_promedio_historica"].fillna(
        df_merged["promedio_general"]
    )

    return df_merged


def run_preprocessing(raw_data: dict) -> dict:
    """
    Ejecuta el pipeline completo de preprocesamiento.

    Args:
        raw_data: Diccionario con las tres fuentes crudas (students, courses, grades).

    Returns:
        Diccionario con los datos limpios y el DataFrame enriquecido:
            'students': DataFrame limpio de estudiantes.
            'courses':  DataFrame limpio de materias.
            'grades':   DataFrame limpio de notas.
            'merged':   DataFrame combinado enriquecido.
    """
    students_clean = clean_students(raw_data["students"])
    courses_clean = clean_courses(raw_data["courses"])
    grades_clean = clean_grades(raw_data["grades"])

    merged = merge_all(students_clean, courses_clean, grades_clean)

    return {
        "students": students_clean,
        "courses": courses_clean,
        "grades": grades_clean,
        "merged": merged
    }
