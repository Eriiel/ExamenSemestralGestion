"""
Modulo de ingesta de datos para el Sistema de Recomendacion Academica UTP.
Responsable de cargar y validar los tres datasets desde sus archivos CSV.

Fuentes de datos:
    Fuente 1: students_performance.csv  (rendimiento academico por estudiante)
    Fuente 2: courses_catalog.csv       (catalogo de materias UTP)
    Fuente 3: student_grades.csv        (historial de notas por materia)
"""

import pandas as pd
import os
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def load_student_performance() -> pd.DataFrame:
    """
    Carga el dataset de rendimiento estudiantil desde CSV.
    Fuente 1: students_performance.csv

    Returns:
        DataFrame con columnas de rendimiento academico por estudiante.

    Raises:
        FileNotFoundError: Si el archivo no existe. Ejecutar generate_data.py primero.
    """
    filepath = DATA_DIR / "students_performance.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {filepath}\n"
            "Ejecuta primero: python data/generate_data.py"
        )
    df = pd.read_csv(filepath, encoding="utf-8")
    return df


def load_courses_catalog() -> pd.DataFrame:
    """
    Carga el catalogo de materias desde CSV.
    Fuente 2: courses_catalog.csv

    Returns:
        DataFrame con informacion de todas las materias disponibles en la UTP.

    Raises:
        FileNotFoundError: Si el archivo no existe. Ejecutar generate_data.py primero.
    """
    filepath = DATA_DIR / "courses_catalog.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {filepath}\n"
            "Ejecuta primero: python data/generate_data.py"
        )
    df = pd.read_csv(filepath, encoding="utf-8")
    return df


def load_student_grades() -> pd.DataFrame:
    """
    Carga el historial de notas de los estudiantes desde CSV.
    Fuente 3: student_grades.csv

    Returns:
        DataFrame con registros de nota, aprobacion y numero de intentos por curso.

    Raises:
        FileNotFoundError: Si el archivo no existe. Ejecutar generate_data.py primero.
    """
    filepath = DATA_DIR / "student_grades.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {filepath}\n"
            "Ejecuta primero: python data/generate_data.py"
        )
    df = pd.read_csv(filepath, encoding="utf-8")
    return df


def validate_dataframes(students: pd.DataFrame, courses: pd.DataFrame, grades: pd.DataFrame) -> bool:
    """
    Valida la integridad basica de los tres datasets cargados.

    Args:
        students: DataFrame de estudiantes.
        courses:  DataFrame de materias.
        grades:   DataFrame de notas.

    Returns:
        True si todos los datasets pasan las validaciones basicas.

    Raises:
        ValueError: Si algun dataset esta vacio o le faltan columnas criticas.
    """

    columnas_estudiantes = [
        "student_id", "nombre_completo", "facultad_codigo", "semestre_actual",
        "promedio_general", "asistencia_promedio", "creditos_aprobados",
        "tasa_reprobacion", "estado_academico", "riesgo_score"
    ]
    columnas_materias = [
        "course_id", "codigo", "nombre", "facultad", "semestre_recomendado",
        "creditos", "area_conocimiento", "nivel_dificultad"
    ]
    columnas_notas = [
        "student_id", "course_id", "nota", "aprobado", "num_intentos"
    ]

    if students.empty:
        raise ValueError("El dataset de estudiantes esta vacio.")
    if courses.empty:
        raise ValueError("El catalogo de materias esta vacio.")
    if grades.empty:
        raise ValueError("El historial de notas esta vacio.")

    for col in columnas_estudiantes:
        if col not in students.columns:
            raise ValueError(f"Columna faltante en students: '{col}'")

    for col in columnas_materias:
        if col not in courses.columns:
            raise ValueError(f"Columna faltante en courses: '{col}'")

    for col in columnas_notas:
        if col not in grades.columns:
            raise ValueError(f"Columna faltante en grades: '{col}'")

    return True


def load_all_data() -> dict:
    """
    Carga y valida todas las fuentes de datos del pipeline.

    Returns:
        Diccionario con las tres fuentes de datos:
            'students': DataFrame de rendimiento estudiantil.
            'courses':  DataFrame del catalogo de materias.
            'grades':   DataFrame del historial de notas.
    """
    students = load_student_performance()
    courses = load_courses_catalog()
    grades = load_student_grades()

    validate_dataframes(students, courses, grades)

    return {
        "students": students,
        "courses": courses,
        "grades": grades
    }
