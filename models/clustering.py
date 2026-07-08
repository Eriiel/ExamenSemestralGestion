"""
Modulo de clustering para identificacion de perfiles academicos estudiantiles.
Utiliza KMeans con k=4 para segmentar estudiantes en cuatro perfiles:
    Alto Rendimiento, Rendimiento Regular, En Riesgo, Riesgo Critico.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pipeline.feature_engineering import get_feature_matrix, FEATURE_COLUMNS


N_CLUSTERS = 4
RANDOM_STATE = 42

# Colores y configuracion visual por perfil para el dashboard
CLUSTER_CONFIG = {
    "Alto Rendimiento": {
        "color": "#2ecc71",
        "descripcion": "Estudiante con excelente desempeno academico, alta asistencia y minima tasa de reprobacion.",
        "recomendacion_general": "Considerar materias avanzadas o electivas de especializacion.",
        "icono": "estrella"
    },
    "Rendimiento Regular": {
        "color": "#3498db",
        "descripcion": "Estudiante con desempeno academico adecuado, cumple con los estandares minimos requeridos.",
        "recomendacion_general": "Reforzar materias de mayor dificultad y mantener habitos de estudio.",
        "icono": "libro"
    },
    "En Riesgo": {
        "color": "#f39c12",
        "descripcion": "Estudiante con indicadores de alerta: GPA bajo, asistencia irregular o alto nivel de reprobaciones.",
        "recomendacion_general": "Se recomienda asesoria academica y reduccion de carga de materias.",
        "icono": "alerta"
    },
    "Riesgo Critico": {
        "color": "#e74c3c",
        "descripcion": "Estudiante en situacion critica con multiples indicadores de desercion o suspension academica.",
        "recomendacion_general": "Requiere intervencion inmediata del departamento academico y apoyo psicopedagogico.",
        "icono": "critico"
    }
}


def train_clustering(df: pd.DataFrame) -> dict:
    """
    Entrena el modelo KMeans sobre el DataFrame de estudiantes con features calculadas.

    El orden de labels se determina automaticamente segun el centroide
    de promedio_general (de mayor a menor) para asignar etiquetas semanticas.

    Args:
        df: DataFrame con features de clustering calculadas por feature_engineering.

    Returns:
        Diccionario con:
            'model'          : objeto KMeans entrenado.
            'labels'         : array de etiquetas de cluster (0-3).
            'label_names'    : array de nombres semanticos por estudiante.
            'silhouette'     : puntuacion de silueta del clustering.
            'centroids_df'   : DataFrame con centroides por cluster etiquetado.
            'cluster_map'    : dict de numero de cluster a nombre semantico.
    """
    X_scaled, scaler = get_feature_matrix(df)

    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20, max_iter=500)
    labels_raw = model.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, labels_raw)

    centroids_scaled = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids_original, columns=FEATURE_COLUMNS)
    centroids_df["cluster_num"] = range(N_CLUSTERS)

    # Ordenar clusters por promedio_general descendente para asignar etiquetas
    ranking = centroids_df.sort_values("promedio_general", ascending=False)["cluster_num"].tolist()
    etiquetas_semanticas = ["Alto Rendimiento", "Rendimiento Regular", "En Riesgo", "Riesgo Critico"]
    cluster_map = {num_cluster: etiqueta for num_cluster, etiqueta in zip(ranking, etiquetas_semanticas)}

    label_names = np.array([cluster_map[label] for label in labels_raw])
    centroids_df["perfil"] = centroids_df["cluster_num"].map(cluster_map)

    return {
        "model": model,
        "scaler": scaler,
        "labels": labels_raw,
        "label_names": label_names,
        "silhouette": round(silhouette, 4),
        "centroids_df": centroids_df,
        "cluster_map": cluster_map
    }


def assign_cluster_labels(df: pd.DataFrame, clustering_result: dict) -> pd.DataFrame:
    """
    Asigna los resultados del clustering al DataFrame de estudiantes.

    Args:
        df:                 DataFrame de estudiantes con features calculadas.
        clustering_result:  Diccionario devuelto por train_clustering().

    Returns:
        DataFrame con columnas adicionales:
            'cluster_num'   : numero de cluster asignado (0-3).
            'cluster_label' : nombre semantico del perfil.
            'cluster_color' : color hex para visualizacion.
    """
    df = df.copy()
    df["cluster_num"] = clustering_result["labels"]
    df["cluster_label"] = clustering_result["label_names"]
    df["cluster_color"] = df["cluster_label"].map(
        {k: v["color"] for k, v in CLUSTER_CONFIG.items()}
    )
    return df


def get_cluster_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadisticas descriptivas por cluster para el dashboard.

    Args:
        df: DataFrame con columna 'cluster_label' asignada.

    Returns:
        DataFrame con metricas promedio por cluster:
            promedio_general, asistencia_promedio, tasa_reprobacion,
            avance_carrera, n_estudiantes.
    """
    stats = df.groupby("cluster_label").agg(
        n_estudiantes=("student_id", "count"),
        promedio_general=("promedio_general", "mean"),
        asistencia_promedio=("asistencia_promedio", "mean"),
        tasa_reprobacion=("tasa_reprobacion", "mean"),
        avance_carrera=("avance_carrera", "mean"),
        riesgo_score=("riesgo_score", "mean")
    ).round(2).reset_index()

    # Ordenar segun jerarquia de perfiles
    orden = ["Alto Rendimiento", "Rendimiento Regular", "En Riesgo", "Riesgo Critico"]
    stats["cluster_label"] = pd.Categorical(stats["cluster_label"], categories=orden, ordered=True)
    stats = stats.sort_values("cluster_label").reset_index(drop=True)
    return stats


def predict_new_student(features_dict: dict, clustering_result: dict) -> str:
    """
    Predice el perfil de un nuevo estudiante dado un diccionario de features.

    Args:
        features_dict:      Dict con valores para cada feature de FEATURE_COLUMNS.
        clustering_result:  Diccionario devuelto por train_clustering().

    Returns:
        Nombre semantico del cluster predicho.
    """
    model = clustering_result["model"]
    scaler = clustering_result["scaler"]
    cluster_map = clustering_result["cluster_map"]

    X = np.array([[features_dict[f] for f in FEATURE_COLUMNS]])
    X_scaled = scaler.transform(X)
    cluster_num = model.predict(X_scaled)[0]
    return cluster_map[cluster_num]
