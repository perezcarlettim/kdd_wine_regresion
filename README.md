# Proceso KDD: Regresión Lineal vs. Árbol de Decisión 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kddwineregresion.streamlit.app/)



**Minería de Datos II · ITSE · Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial**

Estimación del puntaje de calidad (`quality`) de vinos tintos *Vinho Verde* a partir de 11 variables fisicoquímicas, comparando una **Regresión Lineal Múltiple** y un **Árbol de Decisión de Regresión** siguiendo de forma explícita el proceso **KDD** (Knowledge Discovery in Databases).

## Dataset

- **Wine Quality (Vinho Verde tinto)** — UCI Machine Learning Repository (Cortez et al., 2009).
- 1.599 muestras, 11 predictores numéricos continuos y el target `quality` (observado entre 3 y 8).
- El notebook lo descarga automáticamente desde: https://archive.ics.uci.edu/dataset/186/wine+quality

## Estructura del repositorio

```
kdd_wine_regresion/

├── docs/
│   └── TP1_Evaluativo_2026.pdf        # Consigna del trabajo práctico
├── notebooks/
│   └── TP1_KDD_Vino_Regresion.ipynb   # Código completo y comentado de las 6 etapas KDD
└── README.md
├── app.py                             # App web de Streamlit (una pestaña por etapa KDD)
├── data/                              # (opcional) winequality-red.csv para correr sin internet
├── informe/                           # Informe (.docx) generado por el notebook (máx. 4 páginas)
├── requirements.txt                   # Dependencias de Python
├── .gitignore

```
## Etapas KDD cubiertas

1. **Selección** — carga del dataset y EDA descriptivo.
2. **Preprocesamiento / Limpieza** — nulos, duplicados, outliers (IQR), unidades y formatos.
3. **Transformación** — separación X/y, partición 80/20 estratificada, escalado.
4. **Minería de datos** — Regresión Lineal y árboles con distintas profundidades (selección por validación cruzada).
5. **Interpretación / Evaluación** — MAE, RMSE, R², análisis de sobreajuste, coeficientes e importancias.
6. **Conocimiento / Selección del modelo** — tabla comparativa, modelo final y conclusiones.
7. **Preguntas de análisis (20–24)** — con evidencia empírica en el código.


## Tecnologías

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · python-docx

## Autor/a

Perez Carletti, Martin

