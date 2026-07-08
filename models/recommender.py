"""
Motor de recomendacion de materias para el Sistema de Recomendacion Academica UTP.
Implementa un enfoque hibrido que combina:
    1. Filtrado basado en contenido (afinidad por area, nivel vs dificultad).
    2. Filtrado basado en cluster (cursos con exito en estudiantes del mismo perfil).
    3. Filtrado por prerequisitos completados.
"""

import pandas as pd
import numpy as np


# Pesos del sistema de scoring hibrido (suman 1.0)
PESO_AREA_AFINIDAD     = 0.30
PESO_NIVEL_DIFICULTAD  = 0.25
PESO_CLUSTER_EXITO     = 0.25
PESO_SEMESTRE_MATCH    = 0.20


def get_completed_courses(student_id: str, grades: pd.DataFrame) -> set:
    """
    Retorna el conjunto de course_id aprobados por un estudiante.

    Args:
        student_id: Identificador del estudiante.
        grades:     DataFrame de historial de notas.

    Returns:
        Set de course_id aprobados por el estudiante.
    """
    aprobadas = grades[
        (grades["student_id"] == student_id) & (grades["aprobado"] == True)
    ]
    return set(aprobadas["course_id"].tolist())


def check_prerequisites_met(course_row: pd.Series, completed_ids: set) -> bool:
    """
    Verifica si el estudiante ha completado todos los prerequisitos de una materia.

    Args:
        course_row:     Fila del catalogo de materias.
        completed_ids:  Set de course_id ya aprobados por el estudiante.

    Returns:
        True si todos los prerequisitos estan satisfechos (o no tiene prerequisitos).
    """
    prereq_str = str(course_row.get("prerequisitos_ids", "")).strip()
    if not prereq_str or prereq_str == "nan":
        return True

    prereq_ids = [int(p) for p in prereq_str.split(",") if p.strip().isdigit()]
    return all(pid in completed_ids for pid in prereq_ids)


def compute_area_affinity(student_id: str, course_area: str, grades: pd.DataFrame, courses: pd.DataFrame) -> float:
    """
    Calcula la afinidad del estudiante con un area de conocimiento
    basandose en su nota promedio en materias de esa misma area.

    Args:
        student_id:  Identificador del estudiante.
        course_area: Area de conocimiento de la materia candidata.
        grades:      DataFrame de notas.
        courses:     DataFrame del catalogo.

    Returns:
        Puntaje de afinidad normalizado entre 0.0 y 1.0.
    """
    cursos_area = courses[courses["area_conocimiento"] == course_area]["course_id"].tolist()

    notas_area = grades[
        (grades["student_id"] == student_id) &
        (grades["course_id"].isin(cursos_area))
    ]["nota"]

    if notas_area.empty:
        return 0.5

    return float(notas_area.mean()) / 100.0


def compute_difficulty_match(student_gpa: float, course_difficulty: int) -> float:
    """
    Calcula que tan bien se adapta la dificultad del curso al nivel del estudiante.
    Un curso muy dificil para un estudiante de bajo GPA da un puntaje bajo.
    Un estudiante de alto rendimiento se beneficia de cursos desafiantes.

    Args:
        student_gpa:       Promedio general del estudiante (0-100).
        course_difficulty: Nivel de dificultad de la materia (1-5).

    Returns:
        Puntaje de compatibilidad entre 0.0 y 1.0.
    """
    nivel_estudiante = student_gpa / 20.0

    diferencia = abs(nivel_estudiante - course_difficulty)
    score = max(0.0, 1.0 - (diferencia / 4.0))
    return score


def compute_cluster_success_rate(
    cluster_label: str,
    course_id: int,
    students: pd.DataFrame,
    grades: pd.DataFrame
) -> float:
    """
    Calcula la tasa de aprobacion de una materia entre estudiantes del mismo cluster.
    Refleja el exito historico del perfil de este estudiante en ese curso.

    Args:
        cluster_label: Nombre del cluster del estudiante.
        course_id:     ID de la materia candidata.
        students:      DataFrame de estudiantes con cluster_label asignado.
        grades:        DataFrame de notas.

    Returns:
        Tasa de exito (0.0 a 1.0). Retorna 0.5 si no hay datos historicos.
    """
    if "cluster_label" not in students.columns:
        return 0.5

    estudiantes_mismo_cluster = students[students["cluster_label"] == cluster_label]["student_id"].tolist()

    notas_cluster = grades[
        (grades["student_id"].isin(estudiantes_mismo_cluster)) &
        (grades["course_id"] == course_id)
    ]

    if notas_cluster.empty:
        return 0.5

    return float(notas_cluster["aprobado"].mean())


def compute_semester_match(student_semester: int, course_recommended_semester: int) -> float:
    """
    Calcula la compatibilidad de semestre entre el estudiante y el curso.
    Favorece cursos del semestre siguiente al actual o del mismo nivel.

    Args:
        student_semester:          Semestre actual del estudiante.
        course_recommended_semester: Semestre recomendado del curso.

    Returns:
        Puntaje de compatibilidad de semestre entre 0.0 y 1.0.
    """
    diferencia = course_recommended_semester - student_semester

    if diferencia == 1:
        return 1.0
    elif diferencia == 0:
        return 0.85
    elif diferencia == 2:
        return 0.65
    elif diferencia < 0:
        return 0.2
    else:
        return max(0.0, 0.5 - (diferencia - 2) * 0.15)


def score_course(
    student_row: pd.Series,
    course_row: pd.Series,
    grades: pd.DataFrame,
    students: pd.DataFrame,
    courses: pd.DataFrame
) -> float:
    """
    Calcula el puntaje de recomendacion para una materia candidata.
    Combina los cuatro componentes del sistema hibrido.

    Args:
        student_row: Fila del DataFrame de estudiantes.
        course_row:  Fila del catalogo de materias.
        grades:      DataFrame de notas.
        students:    DataFrame completo de estudiantes (con cluster_label).
        courses:     DataFrame del catalogo.

    Returns:
        Puntaje total de recomendacion entre 0.0 y 1.0.
    """
    area_score = compute_area_affinity(
        student_row["student_id"],
        course_row["area_conocimiento"],
        grades,
        courses
    )

    difficulty_score = compute_difficulty_match(
        student_row["promedio_general"],
        course_row["nivel_dificultad"]
    )

    cluster_label = student_row.get("cluster_label", "Rendimiento Regular")
    cluster_score = compute_cluster_success_rate(
        cluster_label,
        course_row["course_id"],
        students,
        grades
    )

    semester_score = compute_semester_match(
        student_row["semestre_actual"],
        course_row["semestre_recomendado"]
    )

    total_score = (
        PESO_AREA_AFINIDAD    * area_score +
        PESO_NIVEL_DIFICULTAD * difficulty_score +
        PESO_CLUSTER_EXITO    * cluster_score +
        PESO_SEMESTRE_MATCH   * semester_score
    )

    return round(total_score, 4)


def generate_recommendations(
    student_id: str,
    students: pd.DataFrame,
    courses: pd.DataFrame,
    grades: pd.DataFrame,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Genera las top N recomendaciones de materias para un estudiante especifico.

    El proceso es:
        1. Obtener materias ya aprobadas por el estudiante.
        2. Filtrar materias elegibles (facultad o comunes, prerequisitos cumplidos).
        3. Puntuar cada materia candidata con el sistema hibrido.
        4. Retornar las top N ordenadas por puntaje.

    Args:
        student_id: Identificador del estudiante.
        students:   DataFrame de estudiantes con cluster_label.
        courses:    DataFrame del catalogo de materias.
        grades:     DataFrame de notas.
        top_n:      Numero de recomendaciones a devolver.

    Returns:
        DataFrame con las top N materias recomendadas y sus puntajes.
    """
    student_row = students[students["student_id"] == student_id]
    if student_row.empty:
        return pd.DataFrame()

    student_row = student_row.iloc[0]
    fac_codigo = student_row["facultad_codigo"]
    completed_ids = get_completed_courses(student_id, grades)

    # Filtrar materias elegibles: comunes + de su facultad, sin prerequisitos pendientes
    cursos_elegibles = courses[
        (courses["facultad"].isin(["COMUN", fac_codigo]))
    ].copy()

    # Excluir materias ya aprobadas
    cursos_elegibles = cursos_elegibles[
        ~cursos_elegibles["course_id"].isin(completed_ids)
    ]

    # Excluir materias sin prerequisitos cumplidos
    cursos_elegibles = cursos_elegibles[
        cursos_elegibles.apply(
            lambda row: check_prerequisites_met(row, completed_ids), axis=1
        )
    ]

    if cursos_elegibles.empty:
        return pd.DataFrame()

    # Calcular puntaje por materia candidata
    scores = []
    for _, curso in cursos_elegibles.iterrows():
        puntaje = score_course(student_row, curso, grades, students, courses)
        razones = build_recommendation_reason(student_row, curso, grades, courses)
        scores.append({
            "course_id": curso["course_id"],
            "codigo": curso["codigo"],
            "nombre": curso["nombre"],
            "area_conocimiento": curso["area_conocimiento"],
            "semestre_recomendado": curso["semestre_recomendado"],
            "creditos": curso["creditos"],
            "nivel_dificultad": curso["nivel_dificultad"],
            "descripcion": curso["descripcion"],
            "puntaje_recomendacion": puntaje,
            "razon": razones
        })

    resultado = pd.DataFrame(scores)
    resultado = resultado.sort_values("puntaje_recomendacion", ascending=False).head(top_n)
    resultado = resultado.reset_index(drop=True)
    resultado.index += 1

    return resultado


def build_recommendation_reason(
    student_row: pd.Series,
    course_row: pd.Series,
    grades: pd.DataFrame,
    courses: pd.DataFrame
) -> str:
    """
    Construye una explicacion legible del porque se recomienda una materia.

    Args:
        student_row: Fila del estudiante.
        course_row:  Fila de la materia recomendada.
        grades:      DataFrame de notas.
        courses:     DataFrame del catalogo.

    Returns:
        Cadena con la razon principal de la recomendacion.
    """
    area = course_row["area_conocimiento"]
    dificultad = course_row["nivel_dificultad"]
    gpa = student_row["promedio_general"]
    semestre_curso = course_row["semestre_recomendado"]
    semestre_estudiante = student_row["semestre_actual"]

    razones = []

    affinity = compute_area_affinity(
        student_row["student_id"], area, grades, courses
    )
    if affinity >= 0.78:
        razones.append(f"alta afinidad con el area de {area} ({affinity*100:.0f}% promedio)")

    if dificultad <= 2 and gpa < 70:
        razones.append("nivel de dificultad adecuado para reforzar bases")
    elif dificultad >= 4 and gpa >= 85:
        razones.append("nivel de dificultad apropiado para su alto rendimiento")
    elif abs(dificultad - (gpa / 20)) <= 1:
        razones.append("dificultad alineada con su nivel academico actual")

    if semestre_curso == semestre_estudiante + 1:
        razones.append("materia indicada para el proximo semestre")
    elif semestre_curso == semestre_estudiante:
        razones.append("materia de su semestre actual pendiente de cursar")

    if not razones:
        razones.append("prerequisitos cumplidos y elegible segun su avance")

    return "; ".join(razones[:2]).capitalize() + "."
