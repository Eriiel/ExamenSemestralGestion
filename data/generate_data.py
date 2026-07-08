"""
Modulo de generacion de datos simulados para el Sistema de Recomendacion Academica UTP.
Genera tres fuentes de datos que alimentan el pipeline:
  1. students_performance.csv  (rendimiento academico por estudiante)
  2. courses_catalog.csv       (catalogo de materias de la UTP)
  3. student_grades.csv        (historial de notas por estudiante y materia)
"""

import pandas as pd
import numpy as np
import random
import os
from pathlib import Path

np.random.seed(42)
random.seed(42)

OUTPUT_DIR = Path(__file__).parent


# Definicion de facultades, carreras y sus codigos oficiales UTP

FACULTADES = {
    "FISC": {
        "nombre": "Facultad de Ingenieria de Sistemas Computacionales",
        "carreras": ["Ingenieria en Sistemas y Computacion", "Licenciatura en Desarrollo de Software"],
        "total_creditos": 180
    },
    "FIC": {
        "nombre": "Facultad de Ingenieria Civil",
        "carreras": ["Ingenieria Civil", "Licenciatura en Topografia"],
        "total_creditos": 185
    },
    "FIE": {
        "nombre": "Facultad de Ingenieria Electricista",
        "carreras": ["Ingenieria Electricista", "Ingenieria en Telecomunicaciones"],
        "total_creditos": 183
    },
    "FIEM": {
        "nombre": "Facultad de Ingenieria Mecanica",
        "carreras": ["Ingenieria Mecanica", "Ingenieria Mecatronica"],
        "total_creditos": 182
    },
    "FIIA": {
        "nombre": "Facultad de Ingenieria Industrial y Administrativa",
        "carreras": ["Ingenieria Industrial", "Administracion de Empresas Tecnologicas"],
        "total_creditos": 178
    },
    "FACIN": {
        "nombre": "Facultad de Ciencias Naturales y Exactas",
        "carreras": ["Licenciatura en Matematica", "Licenciatura en Fisica Aplicada"],
        "total_creditos": 170
    }
}

NOMBRES = [
    "Carlos", "Maria", "Luis", "Ana", "Jose", "Laura", "Juan", "Sofia",
    "Pedro", "Gabriela", "Miguel", "Valeria", "Diego", "Camila", "Andres",
    "Daniela", "Ricardo", "Fernanda", "Alejandro", "Paola", "Eduardo",
    "Isabella", "Roberto", "Natalia", "Francisco", "Andrea", "Jorge",
    "Monica", "Hector", "Patricia", "Manuel", "Lucia", "Rafael", "Diana",
    "Sergio", "Carolina", "Fernando", "Adriana", "Pablo", "Melissa"
]

APELLIDOS = [
    "Garcia", "Rodriguez", "Martinez", "Lopez", "Gonzalez", "Perez",
    "Sanchez", "Ramirez", "Torres", "Flores", "Rivera", "Gomez",
    "Diaz", "Reyes", "Cruz", "Morales", "Ortiz", "Gutierrez",
    "Chavez", "Vargas", "Castillo", "Romero", "Herrera", "Medina",
    "Aguilar", "Jimenez", "Moreno", "Munoz", "Alvarado", "Ruiz",
    "Espino", "Aparicio", "Caballero", "Barrios", "Pimentel",
    "Ceballos", "Montenegro", "Villalba", "Cardenas", "Solis"
]


def generate_courses_catalog() -> pd.DataFrame:
    """
    Genera el catalogo de materias de la UTP con 80 cursos distribuidos
    entre materias comunes y especificas por facultad.
    """

    cursos = []

    # Materias comunes a todas las facultades (primer y segundo ano)
    comunes = [
        {"codigo": "MTM-101", "nombre": "Matematica Basica", "semestre_rec": 1,
         "creditos": 3, "area": "Matematicas", "dificultad": 3,
         "prereq_codigos": [], "facultad": "COMUN",
         "descripcion": "Fundamentos de algebra, trigonometria y funciones matematicas."},
        {"codigo": "MTM-201", "nombre": "Calculo I", "semestre_rec": 2,
         "creditos": 4, "area": "Matematicas", "dificultad": 4,
         "prereq_codigos": ["MTM-101"], "facultad": "COMUN",
         "descripcion": "Limites, derivadas e integrales de funciones de una variable."},
        {"codigo": "MTM-301", "nombre": "Calculo II", "semestre_rec": 3,
         "creditos": 4, "area": "Matematicas", "dificultad": 4,
         "prereq_codigos": ["MTM-201"], "facultad": "COMUN",
         "descripcion": "Integrales multiples, series y ecuaciones diferenciales ordinarias."},
        {"codigo": "MTM-401", "nombre": "Calculo III", "semestre_rec": 4,
         "creditos": 4, "area": "Matematicas", "dificultad": 5,
         "prereq_codigos": ["MTM-301"], "facultad": "COMUN",
         "descripcion": "Calculo vectorial, funciones de varias variables y teoremas de Stokes y Gauss."},
        {"codigo": "MTM-501", "nombre": "Algebra Lineal", "semestre_rec": 3,
         "creditos": 3, "area": "Matematicas", "dificultad": 3,
         "prereq_codigos": ["MTM-201"], "facultad": "COMUN",
         "descripcion": "Matrices, vectores, transformaciones lineales y valores propios."},
        {"codigo": "MTM-601", "nombre": "Estadistica y Probabilidad", "semestre_rec": 4,
         "creditos": 3, "area": "Matematicas", "dificultad": 3,
         "prereq_codigos": ["MTM-201"], "facultad": "COMUN",
         "descripcion": "Distribuciones de probabilidad, inferencia estadistica y analisis de datos."},
        {"codigo": "FIS-101", "nombre": "Fisica I", "semestre_rec": 2,
         "creditos": 4, "area": "Fisica", "dificultad": 4,
         "prereq_codigos": ["MTM-101"], "facultad": "COMUN",
         "descripcion": "Mecanica clasica: cinematica, dinamica, trabajo y energia."},
        {"codigo": "FIS-201", "nombre": "Fisica II", "semestre_rec": 3,
         "creditos": 4, "area": "Fisica", "dificultad": 4,
         "prereq_codigos": ["FIS-101", "MTM-201"], "facultad": "COMUN",
         "descripcion": "Electromagnetismo, ondas y termodinamica basica."},
        {"codigo": "QMC-101", "nombre": "Quimica General", "semestre_rec": 1,
         "creditos": 3, "area": "Ciencias Basicas", "dificultad": 3,
         "prereq_codigos": [], "facultad": "COMUN",
         "descripcion": "Estructura atomica, enlaces quimicos y reacciones quimicas fundamentales."},
        {"codigo": "HUM-101", "nombre": "Espanol Tecnico", "semestre_rec": 1,
         "creditos": 2, "area": "Humanidades", "dificultad": 2,
         "prereq_codigos": [], "facultad": "COMUN",
         "descripcion": "Redaccion tecnica, ortografia y comunicacion escrita profesional."},
        {"codigo": "HUM-201", "nombre": "Ingles Tecnico I", "semestre_rec": 2,
         "creditos": 2, "area": "Humanidades", "dificultad": 2,
         "prereq_codigos": [], "facultad": "COMUN",
         "descripcion": "Vocabulario tecnico en ingles y lectura de textos especializados."},
        {"codigo": "HUM-301", "nombre": "Ingles Tecnico II", "semestre_rec": 3,
         "creditos": 2, "area": "Humanidades", "dificultad": 2,
         "prereq_codigos": ["HUM-201"], "facultad": "COMUN",
         "descripcion": "Comunicacion oral y escrita avanzada en ingles para ingenieria."},
        {"codigo": "ADM-101", "nombre": "Administracion General", "semestre_rec": 2,
         "creditos": 2, "area": "Administracion", "dificultad": 2,
         "prereq_codigos": [], "facultad": "COMUN",
         "descripcion": "Principios de administracion, planeacion estrategica y liderazgo."},
        {"codigo": "ARS-101", "nombre": "Dibujo Tecnico", "semestre_rec": 1,
         "creditos": 3, "area": "Diseno", "dificultad": 2,
         "prereq_codigos": [], "facultad": "COMUN",
         "descripcion": "Normas de dibujo tecnico, vistas ortogonales y AutoCAD basico."},
        {"codigo": "ETI-101", "nombre": "Etica Profesional", "semestre_rec": 8,
         "creditos": 2, "area": "Humanidades", "dificultad": 1,
         "prereq_codigos": [], "facultad": "COMUN",
         "descripcion": "Valores eticos, responsabilidad social y deontologia en ingenieria."},
    ]

    # Materias especificas por facultad
    especificas = {
        "FISC": [
            {"codigo": "ISC-101", "nombre": "Introduccion a la Programacion", "semestre_rec": 1,
             "creditos": 3, "area": "Programacion", "dificultad": 2,
             "prereq_codigos": [], "descripcion": "Logica de programacion, algoritmos y pseudocodigo con Python."},
            {"codigo": "ISC-201", "nombre": "Programacion Orientada a Objetos", "semestre_rec": 2,
             "creditos": 3, "area": "Programacion", "dificultad": 3,
             "prereq_codigos": ["ISC-101"], "descripcion": "Clases, herencia, polimorfismo y patrones de diseno basicos en Java."},
            {"codigo": "ISC-301", "nombre": "Estructuras de Datos", "semestre_rec": 3,
             "creditos": 4, "area": "Programacion", "dificultad": 4,
             "prereq_codigos": ["ISC-201", "MTM-101"], "descripcion": "Listas, arboles, grafos y algoritmos de ordenamiento y busqueda."},
            {"codigo": "ISC-401", "nombre": "Base de Datos I", "semestre_rec": 4,
             "creditos": 3, "area": "Bases de Datos", "dificultad": 3,
             "prereq_codigos": ["ISC-201"], "descripcion": "Modelo relacional, SQL, normalizacion y diseno de esquemas."},
            {"codigo": "ISC-402", "nombre": "Base de Datos II", "semestre_rec": 5,
             "creditos": 3, "area": "Bases de Datos", "dificultad": 4,
             "prereq_codigos": ["ISC-401"], "descripcion": "Procedimientos almacenados, triggers, transacciones y optimizacion."},
            {"codigo": "ISC-501", "nombre": "Redes de Computadoras", "semestre_rec": 5,
             "creditos": 3, "area": "Redes", "dificultad": 3,
             "prereq_codigos": ["ISC-201"], "descripcion": "Modelo OSI, protocolos TCP/IP, segmentacion y enrutamiento."},
            {"codigo": "ISC-502", "nombre": "Seguridad Informatica", "semestre_rec": 6,
             "creditos": 3, "area": "Redes", "dificultad": 4,
             "prereq_codigos": ["ISC-501"], "descripcion": "Criptografia, vulnerabilidades, firewalls y auditorias de seguridad."},
            {"codigo": "ISC-601", "nombre": "Ingenieria de Software I", "semestre_rec": 5,
             "creditos": 3, "area": "Ingenieria de Software", "dificultad": 3,
             "prereq_codigos": ["ISC-301"], "descripcion": "Ciclo de vida del software, metodologias agiles y modelado UML."},
            {"codigo": "ISC-602", "nombre": "Ingenieria de Software II", "semestre_rec": 6,
             "creditos": 3, "area": "Ingenieria de Software", "dificultad": 4,
             "prereq_codigos": ["ISC-601"], "descripcion": "Pruebas de software, arquitectura de sistemas y gestion de proyectos."},
            {"codigo": "ISC-701", "nombre": "Sistemas Operativos", "semestre_rec": 5,
             "creditos": 3, "area": "Sistemas", "dificultad": 4,
             "prereq_codigos": ["ISC-301"], "descripcion": "Procesos, hilos, memoria virtual, sistemas de archivos y Linux avanzado."},
            {"codigo": "ISC-801", "nombre": "Inteligencia Artificial", "semestre_rec": 7,
             "creditos": 3, "area": "IA y Datos", "dificultad": 4,
             "prereq_codigos": ["MTM-601", "ISC-301"], "descripcion": "Busqueda heuristica, logica difusa, redes neuronales y aprendizaje automatico."},
            {"codigo": "ISC-802", "nombre": "Ciencia de Datos", "semestre_rec": 8,
             "creditos": 3, "area": "IA y Datos", "dificultad": 4,
             "prereq_codigos": ["ISC-801", "MTM-601"], "descripcion": "Pipeline de datos, visualizacion, ML aplicado y despliegue de modelos."},
            {"codigo": "ISC-901", "nombre": "Desarrollo Web Full Stack", "semestre_rec": 6,
             "creditos": 3, "area": "Desarrollo Web", "dificultad": 3,
             "prereq_codigos": ["ISC-401"], "descripcion": "HTML, CSS, JavaScript, React, APIs REST y despliegue en la nube."},
            {"codigo": "ISC-902", "nombre": "Computacion en la Nube", "semestre_rec": 8,
             "creditos": 3, "area": "Sistemas", "dificultad": 3,
             "prereq_codigos": ["ISC-501", "ISC-701"], "descripcion": "AWS, Azure, contenedores Docker, Kubernetes y arquitecturas serverless."},
            {"codigo": "ISC-1001", "nombre": "Proyecto de Graduacion FISC", "semestre_rec": 10,
             "creditos": 6, "area": "Investigacion", "dificultad": 5,
             "prereq_codigos": ["ISC-602", "ISC-801"], "descripcion": "Desarrollo e implementacion de un proyecto de software de grado."},
        ],
        "FIC": [
            {"codigo": "CIV-101", "nombre": "Topografia I", "semestre_rec": 2,
             "creditos": 3, "area": "Topografia", "dificultad": 3,
             "prereq_codigos": ["ARS-101"], "descripcion": "Levantamientos topograficos, altimetria y uso de equipo de precision."},
            {"codigo": "CIV-201", "nombre": "Materiales de Construccion", "semestre_rec": 3,
             "creditos": 3, "area": "Construccion", "dificultad": 3,
             "prereq_codigos": ["QMC-101"], "descripcion": "Propiedades mecanicas del concreto, acero, madera y materiales compuestos."},
            {"codigo": "CIV-301", "nombre": "Mecanica de Fluidos", "semestre_rec": 4,
             "creditos": 4, "area": "Hidraulica", "dificultad": 4,
             "prereq_codigos": ["FIS-201", "MTM-301"], "descripcion": "Estatica y dinamica de fluidos, ecuacion de Bernoulli y flujo en tuberias."},
            {"codigo": "CIV-401", "nombre": "Estructuras I", "semestre_rec": 5,
             "creditos": 4, "area": "Estructuras", "dificultad": 4,
             "prereq_codigos": ["CIV-201", "MTM-401"], "descripcion": "Analisis de cargas, vigas, columnas y estructuras isostativas."},
            {"codigo": "CIV-501", "nombre": "Hidraulica", "semestre_rec": 5,
             "creditos": 4, "area": "Hidraulica", "dificultad": 4,
             "prereq_codigos": ["CIV-301"], "descripcion": "Flujo en canales abiertos, bombas hidraulicas y redes de distribucion."},
            {"codigo": "CIV-601", "nombre": "Diseno de Pavimentos", "semestre_rec": 7,
             "creditos": 3, "area": "Vias", "dificultad": 4,
             "prereq_codigos": ["CIV-401"], "descripcion": "Metodos de diseno AASHTO, capas de asfalto y hormigon hidraulico."},
            {"codigo": "CIV-701", "nombre": "Proyecto de Graduacion FIC", "semestre_rec": 10,
             "creditos": 6, "area": "Investigacion", "dificultad": 5,
             "prereq_codigos": ["CIV-601", "CIV-501"], "descripcion": "Proyecto integral de ingenieria civil con diseno, calculo y presupuesto."},
        ],
        "FIE": [
            {"codigo": "ELE-101", "nombre": "Circuitos Electricos I", "semestre_rec": 2,
             "creditos": 4, "area": "Circuitos", "dificultad": 4,
             "prereq_codigos": ["FIS-201", "MTM-201"], "descripcion": "Leyes de Kirchhoff, analisis nodal, thevenin y Norton."},
            {"codigo": "ELE-201", "nombre": "Circuitos Electricos II", "semestre_rec": 3,
             "creditos": 4, "area": "Circuitos", "dificultad": 4,
             "prereq_codigos": ["ELE-101", "MTM-301"], "descripcion": "Regimen sinusoidal, potencia activa y reactiva, filtros y resonancia."},
            {"codigo": "ELE-301", "nombre": "Electronica I", "semestre_rec": 4,
             "creditos": 3, "area": "Electronica", "dificultad": 4,
             "prereq_codigos": ["ELE-201"], "descripcion": "Semiconductores, diodos, transistores BJT y amplificadores basicos."},
            {"codigo": "ELE-401", "nombre": "Maquinas Electricas", "semestre_rec": 5,
             "creditos": 4, "area": "Potencia", "dificultad": 4,
             "prereq_codigos": ["ELE-201"], "descripcion": "Transformadores, motores y generadores de corriente alterna y continua."},
            {"codigo": "ELE-501", "nombre": "Sistemas de Control", "semestre_rec": 6,
             "creditos": 3, "area": "Control", "dificultad": 5,
             "prereq_codigos": ["ELE-401", "MTM-601"], "descripcion": "Funcion de transferencia, diagramas de Bode y controladores PID."},
            {"codigo": "ELE-601", "nombre": "Sistemas de Potencia", "semestre_rec": 7,
             "creditos": 4, "area": "Potencia", "dificultad": 5,
             "prereq_codigos": ["ELE-401"], "descripcion": "Transmision y distribucion de energia electrica, flujos de carga y cortocircuitos."},
            {"codigo": "ELE-701", "nombre": "Proyecto de Graduacion FIE", "semestre_rec": 10,
             "creditos": 6, "area": "Investigacion", "dificultad": 5,
             "prereq_codigos": ["ELE-601", "ELE-501"], "descripcion": "Proyecto de diseno e implementacion en sistemas electricos o electronicos."},
        ],
        "FIEM": [
            {"codigo": "MEC-101", "nombre": "Mecanica de Solidos I", "semestre_rec": 2,
             "creditos": 4, "area": "Mecanica", "dificultad": 4,
             "prereq_codigos": ["FIS-101", "MTM-201"], "descripcion": "Estatica, equilibrio de cuerpos rigidos, celosias y momentos de inercia."},
            {"codigo": "MEC-201", "nombre": "Termodinamica I", "semestre_rec": 3,
             "creditos": 4, "area": "Termica", "dificultad": 4,
             "prereq_codigos": ["FIS-201", "MTM-201"], "descripcion": "Primera y segunda ley de la termodinamica, ciclos termodinamicos."},
            {"codigo": "MEC-301", "nombre": "Diseno Mecanico", "semestre_rec": 5,
             "creditos": 4, "area": "Diseno", "dificultad": 4,
             "prereq_codigos": ["MEC-101", "ARS-101"], "descripcion": "Diseno de elementos de maquinas, engranes, ejes y rodamientos."},
            {"codigo": "MEC-401", "nombre": "Mecanismos y Maquinas", "semestre_rec": 5,
             "creditos": 3, "area": "Mecanica", "dificultad": 4,
             "prereq_codigos": ["MEC-101"], "descripcion": "Cinematica de mecanismos, analisis de movimiento y sintesis de mecanismos."},
            {"codigo": "MEC-501", "nombre": "Control de Procesos Industriales", "semestre_rec": 7,
             "creditos": 3, "area": "Control", "dificultad": 4,
             "prereq_codigos": ["MEC-401", "MTM-601"], "descripcion": "Sensores, actuadores, PLCs y automatizacion de procesos."},
            {"codigo": "MEC-601", "nombre": "Proyecto de Graduacion FIEM", "semestre_rec": 10,
             "creditos": 6, "area": "Investigacion", "dificultad": 5,
             "prereq_codigos": ["MEC-501", "MEC-301"], "descripcion": "Proyecto de diseno e implementacion de sistemas mecanicos o mecatronicos."},
        ],
        "FIIA": [
            {"codigo": "IND-101", "nombre": "Procesos de Manufactura", "semestre_rec": 2,
             "creditos": 3, "area": "Produccion", "dificultad": 3,
             "prereq_codigos": ["ARS-101"], "descripcion": "Fundiciones, maquinado, soldadura y procesos de conformado."},
            {"codigo": "IND-201", "nombre": "Investigacion de Operaciones I", "semestre_rec": 4,
             "creditos": 3, "area": "Optimizacion", "dificultad": 4,
             "prereq_codigos": ["MTM-601"], "descripcion": "Programacion lineal, metodo simplex y teoria de grafos aplicada."},
            {"codigo": "IND-301", "nombre": "Control de Calidad", "semestre_rec": 5,
             "creditos": 3, "area": "Calidad", "dificultad": 3,
             "prereq_codigos": ["MTM-601"], "descripcion": "Cartas de control, muestreo de aceptacion e ISO 9001."},
            {"codigo": "IND-401", "nombre": "Logistica y Cadena de Suministro", "semestre_rec": 6,
             "creditos": 3, "area": "Logistica", "dificultad": 3,
             "prereq_codigos": ["IND-201"], "descripcion": "Gestion de inventarios, distribucion de planta y modelos de transporte."},
            {"codigo": "IND-501", "nombre": "Seguridad e Higiene Industrial", "semestre_rec": 7,
             "creditos": 3, "area": "Seguridad", "dificultad": 2,
             "prereq_codigos": [], "descripcion": "Normativas OSHA, evaluacion de riesgos ocupacionales y planes de emergencia."},
            {"codigo": "IND-601", "nombre": "Proyecto de Graduacion FIIA", "semestre_rec": 10,
             "creditos": 6, "area": "Investigacion", "dificultad": 5,
             "prereq_codigos": ["IND-401", "IND-301"], "descripcion": "Proyecto de mejora de procesos con impacto operativo y economico medible."},
        ],
        "FACIN": [
            {"codigo": "MAT-501", "nombre": "Ecuaciones Diferenciales", "semestre_rec": 4,
             "creditos": 4, "area": "Matematicas", "dificultad": 5,
             "prereq_codigos": ["MTM-301"], "descripcion": "EDOs, sistemas de ecuaciones, series de Fourier y ecuaciones en derivadas parciales."},
            {"codigo": "MAT-601", "nombre": "Analisis Numerico", "semestre_rec": 5,
             "creditos": 3, "area": "Matematicas", "dificultad": 4,
             "prereq_codigos": ["MTM-401", "ISC-101"], "descripcion": "Metodos numericos para ecuaciones, integracion y EDOs con Python."},
            {"codigo": "MAT-701", "nombre": "Estadistica Inferencial", "semestre_rec": 5,
             "creditos": 3, "area": "Estadistica", "dificultad": 4,
             "prereq_codigos": ["MTM-601"], "descripcion": "Pruebas de hipotesis, regresion multiple y analisis de varianza."},
            {"codigo": "MAT-801", "nombre": "Metodos Matematicos", "semestre_rec": 6,
             "creditos": 4, "area": "Matematicas", "dificultad": 5,
             "prereq_codigos": ["MAT-501", "MTM-401"], "descripcion": "Transformadas de Laplace y Fourier, funciones complejas y analisis tensorial."},
            {"codigo": "MAT-901", "nombre": "Proyecto de Graduacion FACIN", "semestre_rec": 10,
             "creditos": 6, "area": "Investigacion", "dificultad": 5,
             "prereq_codigos": ["MAT-801", "MAT-701"], "descripcion": "Investigacion matematica aplicada a problemas de ingenieria o computacion."},
        ]
    }

    course_id = 1
    for curso in comunes:
        cursos.append({
            "course_id": course_id,
            "codigo": curso["codigo"],
            "nombre": curso["nombre"],
            "facultad": curso["facultad"],
            "semestre_recomendado": curso["semestre_rec"],
            "creditos": curso["creditos"],
            "area_conocimiento": curso["area"],
            "nivel_dificultad": curso["dificultad"],
            "prerequisitos_codigos": ",".join(curso["prereq_codigos"]),
            "descripcion": curso["descripcion"]
        })
        course_id += 1

    for fac_codigo, lista_cursos in especificas.items():
        for curso in lista_cursos:
            cursos.append({
                "course_id": course_id,
                "codigo": curso["codigo"],
                "nombre": curso["nombre"],
                "facultad": fac_codigo,
                "semestre_recomendado": curso["semestre_rec"],
                "creditos": curso["creditos"],
                "area_conocimiento": curso["area"],
                "nivel_dificultad": curso["dificultad"],
                "prerequisitos_codigos": ",".join(curso["prereq_codigos"]),
                "descripcion": curso["descripcion"]
            })
            course_id += 1

    df = pd.DataFrame(cursos)

    # Convertir codigos de prerequisitos a IDs numericos para facilitar el procesamiento
    codigo_to_id = dict(zip(df["codigo"], df["course_id"]))
    df["prerequisitos_ids"] = df["prerequisitos_codigos"].apply(
        lambda x: ",".join(str(codigo_to_id[c]) for c in x.split(",") if c in codigo_to_id)
        if x else ""
    )

    return df


def generate_students(n_students: int = 300) -> pd.DataFrame:
    """
    Genera datos simulados de rendimiento academico para n estudiantes.
    Asigna perfiles de rendimiento que determinan sus metricas academicas.
    """

    perfiles = {
        "Alto Rendimiento": {
            "proporcion": 0.25,
            "gpa_media": 90, "gpa_std": 5,
            "asistencia_media": 94, "asistencia_std": 4,
            "tasa_reprobacion_media": 0.04, "tasa_reprobacion_std": 0.03
        },
        "Rendimiento Regular": {
            "proporcion": 0.40,
            "gpa_media": 77, "gpa_std": 5,
            "asistencia_media": 83, "asistencia_std": 6,
            "tasa_reprobacion_media": 0.12, "tasa_reprobacion_std": 0.05
        },
        "En Riesgo": {
            "proporcion": 0.25,
            "gpa_media": 63, "gpa_std": 5,
            "asistencia_media": 68, "asistencia_std": 8,
            "tasa_reprobacion_media": 0.28, "tasa_reprobacion_std": 0.08
        },
        "Riesgo Critico": {
            "proporcion": 0.10,
            "gpa_media": 48, "gpa_std": 7,
            "asistencia_media": 52, "asistencia_std": 10,
            "tasa_reprobacion_media": 0.50, "tasa_reprobacion_std": 0.12
        }
    }

    estudiantes = []
    student_id = 1
    fac_codigos = list(FACULTADES.keys())
    anios_ingreso = [2020, 2021, 2022, 2023, 2024]

    nombres_usados = set()

    for perfil_nombre, cfg in perfiles.items():
        n_perfil = int(n_students * cfg["proporcion"])

        for _ in range(n_perfil):
            fac_codigo = random.choice(fac_codigos)
            fac_info = FACULTADES[fac_codigo]
            carrera = random.choice(fac_info["carreras"])
            anio_ingreso = random.choice(anios_ingreso)

            # Calcular semestre actual basado en anio de ingreso (maximo 10)
            semestres_transcurridos = (2025 - anio_ingreso) * 2
            semestre_actual = min(max(1, semestres_transcurridos + random.randint(-1, 1)), 10)

            # Generar nombre unico
            intentos = 0
            while True:
                nombre = random.choice(NOMBRES)
                apellido1 = random.choice(APELLIDOS)
                apellido2 = random.choice(APELLIDOS)
                nombre_completo = f"{nombre} {apellido1} {apellido2}"
                if nombre_completo not in nombres_usados or intentos > 20:
                    nombres_usados.add(nombre_completo)
                    break
                intentos += 1

            # Generar cedula en formato panameno
            cedula = f"{random.randint(1, 12)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"

            # Generar metricas academicas con base en el perfil
            gpa = np.clip(np.random.normal(cfg["gpa_media"], cfg["gpa_std"]), 0, 100)
            asistencia = np.clip(np.random.normal(cfg["asistencia_media"], cfg["asistencia_std"]), 0, 100)
            tasa_reprobacion = np.clip(
                np.random.normal(cfg["tasa_reprobacion_media"], cfg["tasa_reprobacion_std"]), 0, 0.95
            )

            # Calcular creditos basados en semestre y desempeno
            creditos_por_semestre = 16
            creditos_esperados = semestre_actual * creditos_por_semestre
            creditos_aprobados = int(creditos_esperados * (1 - tasa_reprobacion) * random.uniform(0.85, 1.0))
            creditos_reprobados = int(creditos_esperados * tasa_reprobacion * random.uniform(0.8, 1.2))
            creditos_aprobados = max(0, min(creditos_aprobados, fac_info["total_creditos"]))
            creditos_reprobados = max(0, creditos_reprobados)

            # Determinar estado academico
            if gpa < 50 or tasa_reprobacion > 0.45:
                estado = "Suspension Academica"
            elif gpa < 65 or tasa_reprobacion > 0.25:
                estado = "En Riesgo"
            elif gpa < 75:
                estado = "Regular"
            else:
                estado = "Activo"

            # Score de riesgo (0 a 100, mayor = mas riesgo)
            riesgo_score = round(
                (100 - gpa) * 0.4 +
                (100 - asistencia) * 0.35 +
                tasa_reprobacion * 100 * 0.25, 2
            )

            estudiantes.append({
                "student_id": f"UTP-{str(student_id).zfill(4)}",
                "nombre_completo": nombre_completo,
                "cedula": cedula,
                "correo": f"{nombre.lower()}.{apellido1.lower()}{str(student_id).zfill(3)}@utp.ac.pa",
                "facultad_codigo": fac_codigo,
                "facultad_nombre": fac_info["nombre"],
                "carrera": carrera,
                "anio_ingreso": anio_ingreso,
                "semestre_actual": semestre_actual,
                "promedio_general": round(gpa, 2),
                "asistencia_promedio": round(asistencia, 2),
                "creditos_aprobados": creditos_aprobados,
                "creditos_reprobados": creditos_reprobados,
                "tasa_reprobacion": round(tasa_reprobacion, 4),
                "estado_academico": estado,
                "riesgo_score": riesgo_score,
                "perfil_real": perfil_nombre
            })

            student_id += 1

    df = pd.DataFrame(estudiantes)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def generate_grades(students_df: pd.DataFrame, courses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera el historial de notas por estudiante y materia.
    Simula calificaciones realistas basadas en el perfil academico del estudiante.
    """

    registros = []

    # Mapa de codigo de materia a course_id
    codigo_to_id = dict(zip(courses_df["codigo"], courses_df["course_id"]))

    # Mapa de course_id a semestre recomendado
    curso_semestre = dict(zip(courses_df["course_id"], courses_df["semestre_recomendado"]))

    # Mapa de course_id a facultad
    curso_facultad = dict(zip(courses_df["course_id"], courses_df["facultad"]))

    # Mapa de course_id a dificultad
    curso_dificultad = dict(zip(courses_df["course_id"], courses_df["nivel_dificultad"]))

    for _, estudiante in students_df.iterrows():
        fac = estudiante["facultad_codigo"]
        semestre_actual = estudiante["semestre_actual"]
        gpa = estudiante["promedio_general"]
        tasa_reprobacion = estudiante["tasa_reprobacion"]

        # Seleccionar materias elegibles para este estudiante
        cursos_comunes = courses_df[courses_df["facultad"] == "COMUN"]["course_id"].tolist()
        cursos_facultad = courses_df[courses_df["facultad"] == fac]["course_id"].tolist()
        cursos_elegibles = cursos_comunes + cursos_facultad

        # Tomar solo materias hasta el semestre actual del estudiante
        cursos_cursados_ids = [
            c for c in cursos_elegibles
            if curso_semestre.get(c, 99) <= semestre_actual
        ]

        # Ajustar cantidad de materias segun semestre actual
        n_materias = min(len(cursos_cursados_ids), semestre_actual * 4)
        cursos_cursados_ids = cursos_cursados_ids[:n_materias]

        for course_id in cursos_cursados_ids:
            dificultad = curso_dificultad.get(course_id, 3)
            sem_recomendado = curso_semestre.get(course_id, 5)

            # Nota base influenciada por GPA del estudiante y dificultad del curso
            nota_media = gpa - (dificultad - 3) * 5
            nota_media = np.clip(nota_media, 20, 100)
            nota = np.clip(np.random.normal(nota_media, 8), 0, 100)

            aprobado = nota >= 70
            num_intentos = 1

            # Si reprobo con alta probabilidad segun perfil, agregar segundo intento
            if not aprobado and random.random() < (1 - tasa_reprobacion * 0.5):
                nota_segunda = np.clip(nota + random.uniform(5, 20), 0, 100)
                aprobado_segunda = nota_segunda >= 70
                if aprobado_segunda:
                    nota = nota_segunda
                    aprobado = True
                    num_intentos = 2

            anio_cursado = estudiante["anio_ingreso"] + (sem_recomendado // 2)
            anio_cursado = min(anio_cursado, 2025)

            registros.append({
                "student_id": estudiante["student_id"],
                "course_id": course_id,
                "semestre_cursado": sem_recomendado,
                "anio_cursado": anio_cursado,
                "nota": round(nota, 1),
                "aprobado": aprobado,
                "num_intentos": num_intentos
            })

    return pd.DataFrame(registros)


def main():
    """Punto de entrada principal para generar los tres datasets."""

    print("Generando catalogo de materias UTP...")
    courses_df = generate_courses_catalog()
    courses_path = OUTPUT_DIR / "courses_catalog.csv"
    courses_df.to_csv(courses_path, index=False, encoding="utf-8")
    print(f"  {len(courses_df)} materias guardadas en {courses_path}")

    print("Generando dataset de estudiantes...")
    students_df = generate_students(n_students=300)
    students_path = OUTPUT_DIR / "students_performance.csv"
    students_df.to_csv(students_path, index=False, encoding="utf-8")
    print(f"  {len(students_df)} estudiantes guardados en {students_path}")

    print("Generando historial de notas...")
    grades_df = generate_grades(students_df, courses_df)
    grades_path = OUTPUT_DIR / "student_grades.csv"
    grades_df.to_csv(grades_path, index=False, encoding="utf-8")
    print(f"  {len(grades_df)} registros de notas guardados en {grades_path}")

    print("\nResumen del dataset generado:")
    print(f"  Total estudiantes : {len(students_df)}")
    print(f"  Total materias    : {len(courses_df)}")
    print(f"  Total registros   : {len(grades_df)}")
    print(f"  Facultades        : {students_df['facultad_codigo'].nunique()}")
    print(f"  Distribucion por perfil:")
    for perfil, count in students_df["perfil_real"].value_counts().items():
        print(f"    {perfil}: {count} estudiantes")

    print("\nDatasets generados exitosamente.")


if __name__ == "__main__":
    main()
