# TP Evaluativo Nº 1 — Proceso KDD: Regresión Lineal vs. Árbol de Decisión

**Minería de Datos II · ITSE · Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial**

Estimación del puntaje de calidad (`quality`) de vinos tintos *Vinho Verde* a partir de 11 variables fisicoquímicas, comparando una **Regresión Lineal Múltiple** y un **Árbol de Decisión de Regresión** siguiendo de forma explícita el proceso **KDD** (Knowledge Discovery in Databases).

## Dataset

- **Wine Quality (Vinho Verde tinto)** — UCI Machine Learning Repository (Cortez et al., 2009).
- 1.599 muestras, 11 predictores numéricos continuos y el target `quality` (observado entre 3 y 8).
- El notebook lo descarga automáticamente desde: https://archive.ics.uci.edu/dataset/186/wine+quality

## Estructura del repositorio

```
tp1-kdd-vino-regresion/
├── notebooks/
│   └── TP1_KDD_Vino_Regresion.ipynb   # Código completo y comentado de las 6 etapas KDD
├── docs/
│   └── TP1_Evaluativo_2026.pdf        # Consigna del trabajo práctico
├── app.py                             # App web de Streamlit (una pestaña por etapa KDD)
├── data/                              # (opcional) winequality-red.csv para correr sin internet
├── informe/                           # Informe (.docx) generado por el notebook (máx. 4 páginas)
├── requirements.txt                   # Dependencias de Python
├── .gitignore
└── README.md
```

## Etapas KDD cubiertas

1. **Selección** — carga del dataset y EDA descriptivo.
2. **Preprocesamiento / Limpieza** — nulos, duplicados, outliers (IQR), unidades y formatos.
3. **Transformación** — separación X/y, partición 80/20 estratificada, escalado.
4. **Minería de datos** — Regresión Lineal y árboles con distintas profundidades (selección por validación cruzada).
5. **Interpretación / Evaluación** — MAE, RMSE, R², análisis de sobreajuste, coeficientes e importancias.
6. **Conocimiento / Selección del modelo** — tabla comparativa, modelo final y conclusiones.
7. **Preguntas de análisis (20–24)** — con evidencia empírica en el código.

## App web (Streamlit)

La app `app.py` muestra el proceso completo en pestañas: **Selección · Limpieza · Transformación · Minería de datos · Evaluación · Modelo final y conclusiones · Preguntas 20–24**. Todo se calcula en vivo con los datos reales y la semilla se puede cambiar desde la barra lateral.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Los datos se leen de `data/winequality-red.csv` si existe; si no, se descargan de UCI; y si tampoco hay internet, la app pide subir el CSV.

### Publicarla gratis en Streamlit Community Cloud
1. Subí el repositorio a GitHub (debe ser público, o conectá tu cuenta).
2. Entrá a [share.streamlit.io](https://share.streamlit.io) e iniciá sesión con GitHub.
3. **Create app** → elegí el repositorio, rama `main` y archivo principal `app.py`.
4. **Deploy**. En un par de minutos tenés una URL pública para compartir.

## Cómo ejecutarlo

### Opción A: Google Colab
1. Abrir el notebook en Colab (`Archivo > Abrir cuaderno > GitHub` y pegar la URL del repositorio).
2. `Entorno de ejecución > Ejecutar todo`.

### Opción B: Local
```bash
git clone https://github.com/<tu-usuario>/tp1-kdd-vino-regresion.git
cd tp1-kdd-vino-regresion

python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook notebooks/TP1_KDD_Vino_Regresion.ipynb
```

> La última celda del notebook genera `Informe_TP1_KDD_Vino.docx` con los resultados reales. Moverlo a la carpeta `informe/` antes de hacer el commit.

## Tecnologías

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · python-docx

## Autor/a

- Nombre: _completar_
- Comisión: _completar_
