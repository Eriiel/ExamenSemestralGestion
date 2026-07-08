"""
Modulo de ingenieria de caracteristicas.
Transforma los datos preprocesados en vectores de features para los modelos de ML.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "promedio_general",
    "asistencia_promedio",
    "tasa_reprobacion",
    "avance_carrera",
    "intensidad_academica"
]


def compute_avance_carrera(df: pd.DataFrame) -> pd.Series:
    """
    Calcula el porcentaje de avance en la carrera por estudiante.
    Se calcula como creditos aprobados sobre el total de creditos esperados
    segun el semestre actual.

    Args:
        df: DataFrame enriquecido con creditos_aprobados y semestre_actual.

    Returns:
        Serie con el porcentaje de avance (0 a 100).
    """
    creditos_esperados = df["semestre_actual"] * 16
    avance = (df["creditos_aprobados"] / creditos_esperados.replace(0, 1)) * 100
    return avance.clip(0, 100).round(2)


def compute_intensidad_academica(df: pd.DataFrame) -> pd.Series:
    """
    Calcula la intensidad academica como promedio de materias cursadas por semestre.
    Refleja si el estudiante toma una carga normal, reducida o sobrecargada.

    Args:
        df: DataFrame con n_materias_cursadas y semestre_actual.

    Returns:
        Serie con la intensidad (materias por semestre promedio).
    """
    intensidad = df["n_materias_cursadas"] / df["semestre_actual"].replace(0, 1)
    return intensidad.clip(0, 10).round(2)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea todas las features necesarias para el modelo de clustering.

    Features generadas:
        promedio_general     : GPA del estudiante (0-100).
        asistencia_promedio  : Porcentaje de asistencia (0-100).
        tasa_reprobacion     : Proporcion de materias reprobadas (0-1).
        avance_carrera       : Porcentaje de creditos completados vs esperados.
        intensidad_academica : Materias promedio cursadas por semestre.

    Args:
        df: DataFrame enriquecido proveniente del preprocesamiento.

    Returns:
        DataFrame original con columnas de features adicionales.
    """
    df = df.copy()

    df["avance_carrera"] = compute_avance_carrera(df)
    df["intensidad_academica"] = compute_intensidad_academica(df)

    # Rellenar posibles nulos en features con la mediana del grupo
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)

    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Extrae y escala la matriz de features para el modelo de ML.

    Args:
        df: DataFrame con columnas de features ya calculadas.

    Returns:
        Tupla (X_scaled, scaler):
            X_scaled : numpy array normalizado con StandardScaler.
            scaler   : objeto StandardScaler ajustado para reutilizar.
    """
    X = df[FEATURE_COLUMNS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def get_feature_names() -> list:
    """
    Devuelve los nombres de las features utilizadas en el modelo.

    Returns:
        Lista de nombres de columnas de features.
    """
    return FEATURE_COLUMNS.copy()
