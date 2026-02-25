# taller4-GPI

# Taller 4 - Gestión de Proyectos e Investigación (GPI)

## Descripción

Este repositorio contiene la implementación del Taller 4 del curso GPI.  
El objetivo es estructurar un proyecto reproducible sincronizado con un repositorio remoto en GitHub, siguiendo buenas prácticas de organización, versionamiento y automatización.

El proyecto implementa un pipeline automatizado para:

1. Generación de datos sintéticos
2. Procesamiento de datos
3. Estadísticas descriptivas
4. Pronósticos
5. Generación de resultados (figuras y tablas)

---

## Estructura del Proyecto
└── proyecto/
├── data/
│ ├── raw/ # Datos simulados sin procesar
│ └── processed/ # Datos procesados
├── src/ # Funciones reutilizables (módulos)
├── scripts/ # Scripts ejecutables del pipeline
├── results/
│ ├── figures/ # Gráficos generados
│ └── tables/ # Tablas generadas
├── environment.yml # Dependencias del entorno
├── runall.ps1 # Script maestro de ejecución
└── README.md


---

## Requisitos

- Python 3.9+
- Librerías especificadas en `environment.yml`

Opcionalmente se puede crear un entorno con:

```bash
conda env create -f environment.yml
conda activate nombre_entorno
---
## Ejecuttar el proyecto
.\runall.ps1