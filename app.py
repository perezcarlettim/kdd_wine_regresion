"""
TP Evaluativo Nº 1 — Proceso KDD completo: Regresión Lineal vs. Árbol de Decisión
Minería de Datos II · ITSE · Tecnicatura Superior en Ciencia de Datos e IA

App de Streamlit: una pestaña por etapa KDD, con los resultados calculados en vivo.
Ejecutar con:  streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
LOCAL_CSV = Path(__file__).parent / "data" / "winequality-red.csv"
PURPLE = "#5b2c6f"

st.set_page_config(page_title="TP1 KDD · Wine Quality", page_icon="🍷", layout="wide")
sns.set_theme(style="whitegrid")


# ======================================================================
# Utilidades
# ======================================================================
def metricas(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def tabla_iqr(d, target="quality"):
    filas = []
    for c in d.columns.drop(target):
        q1, q3 = d[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((d[c] < lo) | (d[c] > hi)).sum())
        filas.append({"variable": c, "Q1": q1, "Q3": q3, "lím. inf": lo, "lím. sup": hi,
                      "n outliers": n, "% outliers": 100 * n / len(d)})
    return pd.DataFrame(filas).set_index("variable").sort_values("% outliers", ascending=False)


def mostrar(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


@st.cache_data(show_spinner="Cargando dataset...")
def cargar_local_o_url():
    """1) CSV local en data/  2) URL oficial de UCI  3) None (se pide subir el CSV)."""
    if LOCAL_CSV.exists():
        return pd.read_csv(LOCAL_CSV, sep=";"), "archivo local data/winequality-red.csv"
    try:
        return pd.read_csv(URL, sep=";"), "enlace oficial de UCI"
    except Exception:
        return None, None


# ======================================================================
# Pipeline KDD completo (cacheado)
# ======================================================================
def correr_experimento(d, etiqueta, seed):
    Xe, ye = d.drop(columns="quality"), d["quality"]
    Xtr, Xte, ytr, yte = train_test_split(Xe, ye, test_size=0.2, random_state=seed, stratify=ye)
    sc = StandardScaler().fit(Xtr)
    lr = LinearRegression().fit(sc.transform(Xtr), ytr)
    kf_ = KFold(5, shuffle=True, random_state=seed)
    cv_ = {k: cross_val_score(DecisionTreeRegressor(max_depth=k, random_state=seed),
                              Xtr, ytr, cv=kf_, scoring="r2").mean() for k in range(1, 21)}
    kbest = max(cv_, key=cv_.get)
    ar = DecisionTreeRegressor(max_depth=kbest, random_state=seed).fit(Xtr, ytr)
    ar_full = DecisionTreeRegressor(random_state=seed).fit(Xtr, ytr)
    out = pd.DataFrame({
        "Regresión Lineal": metricas(yte, lr.predict(sc.transform(Xte))),
        f"Árbol (depth={kbest} por CV)": metricas(yte, ar.predict(Xte)),
        "Árbol sin límite": metricas(yte, ar_full.predict(Xte)),
    }).T
    out.insert(0, "dataset", etiqueta)
    out.insert(1, "n_filas", len(d))
    return out


@st.cache_data(show_spinner="Ejecutando el pipeline KDD...")
def correr_pipeline(df_raw, seed):
    R = {"seed": seed}

    # ---------------- Etapa 2: limpieza ----------------
    df = df_raw.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    R["cols_texto"] = list(df.select_dtypes(exclude="number").columns)
    R["nulos"] = df.isnull().sum()
    R["ceros"] = (df == 0).sum()[lambda s: s > 0]
    R["n_dup"] = int(df.duplicated().sum())
    df_con_dup = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    R["df"], R["df_con_dup"] = df, df_con_dup

    R["tab_out"] = tabla_iqr(df)
    mask_any = pd.Series(False, index=df.index)
    for c in df.columns.drop("quality"):
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask_any |= (df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)
    R["pct_any"], R["n_any"] = float(mask_any.mean()), int(mask_any.sum())

    R["checks"] = {
        "pH entre 0 y 14": bool(df["ph"].between(0, 14).all()),
        "Densidad entre 0.98 y 1.01 g/cm³": bool(df["density"].between(0.98, 1.01).all()),
        "Alcohol entre 5 y 20 %": bool(df["alcohol"].between(5, 20).all()),
        "SO₂ libre ≤ SO₂ total": bool((df["free_sulfur_dioxide"] <= df["total_sulfur_dioxide"]).all()),
        "Sin valores negativos": bool((df >= 0).all().all()),
        "quality entero en [3, 8]": bool(df["quality"].between(3, 8).all() and (df["quality"] % 1 == 0).all()),
    }

    # ---------------- Etapa 3: transformación ----------------
    X, y = df.drop(columns="quality"), df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_sc = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    R.update(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
             X_train_sc=X_train_sc, X_test_sc=X_test_sc)
    R["dist_split"] = pd.DataFrame({"train": y_train.value_counts(normalize=True).sort_index(),
                                    "test": y_test.value_counts(normalize=True).sort_index()})

    # ---------------- Etapa 4: minería ----------------
    lin_reg = LinearRegression().fit(X_train_sc, y_train)
    arboles = {d: DecisionTreeRegressor(max_depth=d, random_state=seed).fit(X_train, y_train)
               for d in [3, 5, None]}
    R["lin_reg"], R["arboles"] = lin_reg, arboles
    R["info_arboles"] = pd.DataFrame(
        [{"max_depth": "sin límite" if d is None else d, "profundidad real": m.get_depth(),
          "hojas": m.get_n_leaves()} for d, m in arboles.items()]).set_index("max_depth")

    rango_prof = list(range(1, 21))
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_r2 = {d: cross_val_score(DecisionTreeRegressor(max_depth=d, random_state=seed),
                                X_train, y_train, cv=kf, scoring="r2").mean() for d in rango_prof}
    mejor_prof = max(cv_r2, key=cv_r2.get)
    arbol_mejor = DecisionTreeRegressor(max_depth=mejor_prof, random_state=seed).fit(X_train, y_train)
    R.update(rango_prof=rango_prof, cv_r2=cv_r2, mejor_prof=mejor_prof, arbol_mejor=arbol_mejor)

    # ---------------- Etapa 5: evaluación ----------------
    R["res_lr"] = pd.DataFrame({"train": metricas(y_train, lin_reg.predict(X_train_sc)),
                                "test": metricas(y_test, lin_reg.predict(X_test_sc))}).T
    filas = []
    for d, m in arboles.items():
        mtr, mte = metricas(y_train, m.predict(X_train)), metricas(y_test, m.predict(X_test))
        filas.append({"max_depth": "sin límite" if d is None else d,
                      "MAE_train": mtr["MAE"], "MAE_test": mte["MAE"],
                      "RMSE_train": mtr["RMSE"], "RMSE_test": mte["RMSE"],
                      "R2_train": mtr["R2"], "R2_test": mte["R2"],
                      "Brecha R2": mtr["R2"] - mte["R2"]})
    R["res_arbol"] = pd.DataFrame(filas).set_index("max_depth")

    r2_tr, r2_te = [], []
    for d in rango_prof:
        m = DecisionTreeRegressor(max_depth=d, random_state=seed).fit(X_train, y_train)
        r2_tr.append(r2_score(y_train, m.predict(X_train)))
        r2_te.append(r2_score(y_test, m.predict(X_test)))
    curva = pd.DataFrame({"R2_train": r2_tr, "R2_test": r2_te}, index=rango_prof)
    curva["brecha"] = curva["R2_train"] - curva["R2_test"]
    R["curva"], R["prof_max_test"] = curva, int(curva["R2_test"].idxmax())

    R["coef"] = pd.Series(lin_reg.coef_, index=X.columns).sort_values()
    R["imp"] = pd.Series(arbol_mejor.feature_importances_, index=X.columns).sort_values()

    # ---------------- Etapa 6: selección ----------------
    fila_lr = metricas(y_test, lin_reg.predict(X_test_sc))
    fila_ar = metricas(y_test, arbol_mejor.predict(X_test))
    fila_full = metricas(y_test, arboles[None].predict(X_test))
    R["tabla_final"] = pd.DataFrame({
        "Regresión Lineal": fila_lr,
        f"Árbol (max_depth={mejor_prof}, elegido por CV)": fila_ar,
        "Árbol sin límite (referencia)": fila_full}).T
    cv_lr = cross_val_score(make_pipeline(StandardScaler(), LinearRegression()),
                            X_train, y_train, cv=kf, scoring="r2")
    cv_ar = cross_val_score(DecisionTreeRegressor(max_depth=mejor_prof, random_state=seed),
                            X_train, y_train, cv=kf, scoring="r2")
    ar_tr = metricas(y_train, arbol_mejor.predict(X_train))
    R["robustez"] = pd.DataFrame({
        "R2 CV media": [cv_lr.mean(), cv_ar.mean()],
        "R2 CV desvío": [cv_lr.std(), cv_ar.std()],
        "Brecha R2 train-test": [R["res_lr"].loc["train", "R2"] - fila_lr["R2"], ar_tr["R2"] - fila_ar["R2"]],
    }, index=["Regresión Lineal", f"Árbol (depth={mejor_prof})"])
    R["fila_lr"], R["fila_ar"] = fila_lr, fila_ar
    R["elige_arbol"] = bool((fila_ar["R2"] - fila_lr["R2"] > 0.02) and (cv_ar.mean() > cv_lr.mean()))
    R["y_pred_lr"] = lin_reg.predict(X_test_sc)
    R["y_pred_ar"] = arbol_mejor.predict(X_test)

    # ---------------- Preguntas de análisis ----------------
    # P20: evidencia de fuga
    Xd, yd = df_con_dup.drop(columns="quality"), df_con_dup["quality"]
    Xd_tr, Xd_te, _, _ = train_test_split(Xd, yd, test_size=0.2, random_state=seed, stratify=yd)
    m1 = Xd_te.merge(Xd_tr.drop_duplicates(), how="left", on=list(Xd.columns), indicator=True)
    m2 = X_test.merge(X_train.drop_duplicates(), how="left", on=list(X.columns), indicator=True)
    R["pct_leak_con"] = float((m1["_merge"] == "both").mean())
    R["pct_leak_sin"] = float((m2["_merge"] == "both").mean())

    # P21: experimento de outliers
    mk = pd.Series(True, index=X_train.index)
    for c in X_train.columns:
        q1, q3 = X_train[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        mk &= X_train[c].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    Xtr_f, ytr_f = X_train[mk], y_train[mk]
    sc_f = StandardScaler().fit(Xtr_f)
    lr_f = LinearRegression().fit(sc_f.transform(Xtr_f), ytr_f)
    ar_f = DecisionTreeRegressor(max_depth=mejor_prof, random_state=seed).fit(Xtr_f, ytr_f)
    R["out_train_n"], R["out_train_f"] = len(X_train), len(Xtr_f)
    R["comp_out"] = pd.DataFrame({
        "LR - conservando outliers": metricas(y_test, lin_reg.predict(X_test_sc)),
        "LR - sin outliers (train)": metricas(y_test, lr_f.predict(sc_f.transform(X_test))),
        "Árbol - conservando": metricas(y_test, arbol_mejor.predict(X_test)),
        "Árbol - sin outliers (train)": metricas(y_test, ar_f.predict(X_test)),
    }).T

    # P22: escalado
    lr_orig = LinearRegression().fit(X_train, y_train)
    R["comp_coef"] = pd.DataFrame({"coef. sin escalar": lr_orig.coef_,
                                   "coef. estandarizado": lin_reg.coef_}, index=X.columns)
    R["ok_lr_scale"] = bool(np.allclose(lr_orig.predict(X_test), lin_reg.predict(X_test_sc)))
    arbol_sc = DecisionTreeRegressor(max_depth=mejor_prof, random_state=seed).fit(X_train_sc, y_train)
    R["ok_ar_pred"] = bool(np.allclose(arbol_sc.predict(X_test_sc), arbol_mejor.predict(X_test)))
    R["ok_ar_imp"] = bool(np.allclose(arbol_sc.feature_importances_, arbol_mejor.feature_importances_))

    # P24: repetir sin eliminar duplicados
    R["comparacion"] = pd.concat([
        correr_experimento(df, "SIN duplicados (limpio)", seed),
        correr_experimento(df_con_dup, "CON duplicados", seed)])
    return R


# ======================================================================
# Carga de datos
# ======================================================================
st.title("🍷 Proceso KDD: Regresión Lineal vs. Árbol de Decisión")
st.caption("TP Evaluativo Nº 1 · Minería de Datos II · ITSE · Tecnicatura Superior en Ciencia de Datos e IA · Perez Carletti, Martin"
           
           "Dataset: Wine Quality (Vinho Verde tinto), UCI — Cortez et al. (2009)")

with st.sidebar:
    st.header("⚙️ Configuración")
    seed = int(st.number_input("Semilla aleatoria (random_state)", min_value=0, max_value=9999, value=42, step=1,
                               help="Controla la partición train/test y la validación cruzada. "
                                    "Cambiala para ver cuánto varían los resultados."))
    st.markdown("---")
    st.markdown("**Etapas KDD**\n\n1. Selección\n2. Preprocesamiento\n3. Transformación\n4. Minería de datos\n"
                "5. Interpretación / Evaluación\n6. Conocimiento / Modelo final\n7. Preguntas de análisis (20–24)")

df_raw, fuente = cargar_local_o_url()
if df_raw is None:
    st.warning("No se pudo descargar el dataset automáticamente. Subí el archivo `winequality-red.csv` "
               "(separado por `;`) o guardalo en `data/winequality-red.csv` dentro del repositorio.")
    subido = st.file_uploader("winequality-red.csv", type="csv")
    if subido is None:
        st.stop()
    df_raw, fuente = pd.read_csv(subido, sep=";"), "archivo subido"

st.sidebar.success(f"Datos cargados desde: {fuente}")
R = correr_pipeline(df_raw, seed)
df, X, y = R["df"], R["X"], R["y"]
mejor_prof = R["mejor_prof"]

tabs = st.tabs(["1 · Selección", "2 · Limpieza", "3 · Transformación", "4 · Minería de datos",
                "5 · Evaluación", "6 · Modelo final y conclusiones", "7 · Preguntas 20–24"])

# ======================================================================
# 1. SELECCIÓN
# ======================================================================
with tabs[0]:
    st.header("Etapa 1 — Selección de datos")
    st.markdown(
        "**Decisión:** se usa el dataset *Wine Quality (tinto)* indicado en la consigna: "
        f"**{len(df_raw):,} muestras**, 11 predictores numéricos continuos y un target (`quality`) "
        "que se trata como variable continua para plantear un problema de regresión.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", f"{df_raw.shape[0]:,}")
    c2.metric("Columnas", df_raw.shape[1])
    c3.metric("Tipos de dato", ", ".join(sorted(df_raw.dtypes.astype(str).unique())))

    st.subheader("Primeras filas")
    st.dataframe(df_raw.head(), use_container_width=True)
    st.subheader("Estadísticas descriptivas")
    st.dataframe(df_raw.describe().T.round(4), use_container_width=True)

    st.subheader("Distribución de `quality` y matriz de correlación")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(x="quality", data=df_raw, ax=ax[0], color=PURPLE)
    ax[0].set_title("Distribución de quality")
    for p in ax[0].patches:
        ax[0].annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                       ha="center", va="bottom", fontsize=9)
    sns.heatmap(df_raw.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[1],
                annot_kws={"size": 7}, cbar=False)
    ax[1].set_title("Matriz de correlación")
    plt.tight_layout()
    mostrar(fig)

    cq = df_raw.corr()["quality"].drop("quality").sort_values(ascending=False)
    p56 = df_raw["quality"].isin([5, 6]).mean()
    st.markdown(
        f"**Lectura del EDA**\n"
        f"- `quality` toma valores entre {int(df_raw['quality'].min())} y {int(df_raw['quality'].max())} y se concentra en 5 y 6 "
        f"({p56:.0%} de las muestras): los extremos están poco representados, lo que dificulta predecirlos.\n"
        f"- Mayor correlación positiva con `quality`: **{cq.index[0]}** (r = {cq.iloc[0]:+.2f}); "
        f"mayor correlación negativa: **{cq.index[-1]}** (r = {cq.iloc[-1]:+.2f}).\n"
        "- Hay **multicolinealidad** entre predictores (p. ej. `fixed acidity` con `density`, `citric acid` y `pH`), "
        "relevante al interpretar los coeficientes de la regresión.\n"
        "- Las escalas difieren mucho entre variables (p. ej. `total sulfur dioxide` vs. `chlorides`).")
    with st.expander("Correlación de cada variable con quality"):
        st.dataframe(cq.rename("correlación").to_frame().round(4))

# ======================================================================
# 2. LIMPIEZA
# ======================================================================
with tabs[1]:
    st.header("Etapa 2 — Preprocesamiento y limpieza")

    st.subheader("3 · Formato de nombres")
    st.markdown("Todos los nombres de columnas pasan a minúsculas y sin espacios (`fixed acidity` → `fixed_acidity`). "
                f"Columnas de tipo texto en el dataset: **{len(R['cols_texto'])}** (no hay valores categóricos que unificar).")
    st.code(", ".join(df.columns))

    st.subheader("4 · Valores nulos")
    n_nulos = int(R["nulos"].sum())
    st.markdown(f"Total de nulos: **{n_nulos}**. "
                + ("**Decisión:** no se imputa ni se eliminan filas por este motivo. " if n_nulos == 0
                   else "Se debería imputar (p. ej. mediana ajustada solo con train). ")
                + "Los únicos ceros están en `citric_acid` (" +
                (f"{int(R['ceros'].get('citric_acid', 0))} casos" if len(R["ceros"]) else "ninguno") +
                "), un valor fisicoquímicamente posible (vinos sin ácido cítrico agregado), no un faltante: se conservan.")

    st.subheader("5 · Filas duplicadas exactas")
    c1, c2, c3 = st.columns(3)
    c1.metric("Duplicadas encontradas", R["n_dup"], f"{R['n_dup'] / len(df_raw):.1%} del total", delta_color="off")
    c2.metric("Antes", f"{len(R['df_con_dup']):,} filas")
    c3.metric("Después", f"{len(df):,} filas")
    st.markdown("**Decisión:** se **eliminan antes de dividir** en train/test. Si se dejaran, una misma observación "
                "podría quedar en ambos conjuntos (*data leakage*) e inflar las métricas, sobre todo en modelos que "
                "memorizan, como un árbol profundo (ver pestaña 7, preguntas 20 y 24).")

    st.subheader("6 · Outliers (método IQR)")
    st.dataframe(R["tab_out"].round(3), use_container_width=True)
    st.markdown(f"Filas con al menos un outlier en alguna variable: **{R['n_any']:,} ({R['pct_any']:.1%})**.")
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for ax, c in zip(axes.ravel(), df.columns.drop("quality")):
        sns.boxplot(y=df[c], ax=ax, color="#d7bde2")
        ax.set_title(c)
    axes.ravel()[-1].axis("off")
    plt.tight_layout()
    mostrar(fig)
    st.info(
        "**Decisión: CONSERVAR los outliers.**\n\n"
        "1. Son valores **plausibles del dominio** (vinos con mucho azúcar, cloruros o sulfatos altos), no errores de medición.\n"
        "2. Eliminarlos descartaría una fracción muy grande de datos y justo a los vinos extremos, que más informan sobre calidad.\n"
        "3. El **árbol es robusto** a extremos (parte por umbrales); la regresión lineal es más sensible, por eso se verifica "
        "empíricamente en la pregunta 21.\n"
        "4. Alternativa considerada: transformar (`log1p` en `residual_sugar` y `chlorides`); no se aplica para mantener las "
        "unidades originales y queda como mejora.")

    st.subheader("7 · Unidades y formatos")
    for k, v in R["checks"].items():
        st.markdown(f"{'✅' if v else '⚠️'} {k}")
    st.markdown("Las unidades son homogéneas dentro de cada variable y el formato es numérico: **no se requiere conversión**. "
                "El escalado estadístico (Etapa 3) es una transformación para modelar, no una unificación de unidades.")

    st.subheader("8 · Tamaño final")
    st.markdown(f"- Dataset original: **{df_raw.shape[0]:,} × {df_raw.shape[1]}**\n"
                f"- Duplicadas eliminadas: **{R['n_dup']}** · Nulos tratados: **0** · Outliers eliminados: **0**\n"
                f"- Dataset final: **{df.shape[0]:,} × {df.shape[1]}**")

# ======================================================================
# 3. TRANSFORMACIÓN
# ======================================================================
with tabs[2]:
    st.header("Etapa 3 — Transformación de datos")
    st.subheader("9 · Separar X e y")
    st.markdown(f"`X`: {X.shape[0]:,} × {X.shape[1]} (11 predictores) · `y` = `quality`: {y.shape[0]:,}")

    st.subheader("10 · Partición 80/20")
    c1, c2 = st.columns(2)
    c1.metric("Train", f"{len(R['X_train']):,} filas")
    c2.metric("Test", f"{len(R['X_test']):,} filas")
    st.markdown(f"`random_state={seed}`, **estratificada por `quality`**: como las clases 3 y 8 son muy raras, la "
                "estratificación asegura la misma distribución del target en ambas particiones y una evaluación más estable.")
    st.dataframe(R["dist_split"].round(4), use_container_width=True)

    st.subheader("11 · Escalado")
    st.markdown(
        "- **Regresión Lineal → SÍ se escala** (`StandardScaler` ajustado solo con train para no filtrar información del test). "
        "Las predicciones de OLS no cambian, pero los coeficientes quedan en *unidades de desvío estándar* y se vuelven "
        "**comparables** entre variables.\n"
        "- **Árbol de Decisión → NO se escala**: elige umbrales por variable y es invariante a transformaciones monótonas.")
    st.dataframe(R["X_train_sc"].describe().loc[["mean", "std"]].round(3), use_container_width=True)
    st.caption("Train estandarizado: media ≈ 0 y desvío ≈ 1 en todas las variables.")

# ======================================================================
# 4. MINERÍA
# ======================================================================
with tabs[3]:
    st.header("Etapa 4 — Minería de datos")
    st.subheader("12 · Regresión Lineal Múltiple")
    st.markdown(f"Entrenada sobre las variables estandarizadas. Intercepto: **{R['lin_reg'].intercept_:.4f}** "
                "(≈ la media de `quality` en train).")

    st.subheader("13 · Árboles de decisión de regresión")
    st.markdown("Se entrenan con `max_depth` = 3, 5 y sin límite:")
    st.dataframe(R["info_arboles"], use_container_width=True)

    st.subheader("Elección de la profundidad por validación cruzada")
    st.markdown("Además de las tres profundidades pedidas, se explora el rango 1–20 y se elige la profundidad óptima por "
                "**validación cruzada de 5 folds sobre el conjunto de entrenamiento** (no mirando el test), para que el "
                "test siga siendo una evaluación honesta.")
    cv_ser = pd.Series(R["cv_r2"], name="R² medio CV")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(cv_ser.index, cv_ser.values, "o-", color=PURPLE)
    ax.axvline(mejor_prof, ls="--", color="gray", label=f"mejor = {mejor_prof}")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("R² medio (CV 5 folds)")
    ax.set_xticks(R["rango_prof"])
    ax.legend()
    plt.tight_layout()
    mostrar(fig)
    st.success(f"Mejor profundidad según CV: **{mejor_prof}** (R² CV = {R['cv_r2'][mejor_prof]:.4f})")

# ======================================================================
# 5. EVALUACIÓN
# ======================================================================
with tabs[4]:
    st.header("Etapa 5 — Interpretación y evaluación de resultados")
    st.subheader("14 · Regresión Lineal: MAE, RMSE y R²")
    st.dataframe(R["res_lr"].round(4), use_container_width=True)

    st.subheader("15 · Árbol de Decisión: train vs. test según la profundidad")
    st.dataframe(R["res_arbol"].round(4), use_container_width=True)

    curva = R["curva"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(curva.index, curva["R2_train"], "o-", label="R² train")
    ax.plot(curva.index, curva["R2_test"], "s-", label="R² test")
    ax.axvline(mejor_prof, ls="--", color="gray", label=f"mejor profundidad CV = {mejor_prof}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("R²")
    ax.set_xticks(R["rango_prof"])
    ax.set_title("Árbol de decisión: R² en train y test según la profundidad")
    ax.legend()
    plt.tight_layout()
    mostrar(fig)

    full = R["res_arbol"].loc["sin límite"]
    st.markdown(
        f"**Análisis de sobreajuste.** La brecha de R² (train − test) pasa de **{R['res_arbol'].loc[3, 'Brecha R2']:.3f}** "
        f"con profundidad 3 a **{R['res_arbol'].loc[5, 'Brecha R2']:.3f}** con 5 y a **{full['Brecha R2']:.3f}** sin límite "
        f"(R² train = {full['R2_train']:.3f}; R² test = {full['R2_test']:.3f}).\n\n"
        f"- Con profundidades bajas el árbol es simple y R² de train y test son parecidos y bajos → **subajuste**.\n"
        f"- El R² de test alcanza su máximo cerca de la profundidad **{R['prof_max_test']}**; a partir de ahí deja de mejorar "
        "mientras el de train sigue subiendo: ahí **empieza el sobreajuste**.\n"
        "- Sin límite, el árbol llega a una hoja por observación (R² train ≈ 1) y generaliza mal: **sobreajuste extremo**.")
    with st.expander("Ver tabla de la curva completa (profundidades 1–20)"):
        st.dataframe(curva.round(3).T, use_container_width=True)

    st.subheader("16 · Coeficientes (Regresión Lineal) e importancias (Árbol)")
    coef, imp = R["coef"], R["imp"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    coef.plot.barh(ax=ax[0], color=["#c0392b" if v < 0 else "#2e86c1" for v in coef])
    ax[0].set_title("Regresión Lineal — coeficientes (variables estandarizadas)")
    ax[0].axvline(0, color="black", lw=0.8)
    imp.plot.barh(ax=ax[1], color="#27ae60")
    ax[1].set_title(f"Árbol (max_depth={mejor_prof}) — feature_importances_")
    plt.tight_layout()
    mostrar(fig)

    top_coef = coef.reindex(coef.abs().sort_values(ascending=False).index).head(3)
    top_imp = imp.sort_values(ascending=False).head(3)
    comunes = sorted(set(top_coef.index) & set(top_imp.index))
    st.markdown(
        "- **Regresión Lineal** (mayor |coef| estandarizado): " + ", ".join(f"`{k}` ({v:+.2f})" for k, v in top_coef.items()) + "\n"
        "- **Árbol** (mayor importancia): " + ", ".join(f"`{k}` ({v:.2f})" for k, v in top_imp.items()) + "\n"
        + (f"- Coinciden en: {', '.join('`' + c + '`' for c in comunes)} → refuerza la conclusión.\n" if comunes else "")
        + "- Los coeficientes de variables correlacionadas entre sí (`fixed_acidity`, `density`, `citric_acid`, `ph`) deben "
          "leerse con cautela por la **multicolinealidad**. Las importancias del árbol no indican el *signo* del efecto.")

    with st.expander("Ver el árbol elegido (primeros 3 niveles)"):
        fig = plt.figure(figsize=(20, 8))
        plot_tree(R["arbol_mejor"], feature_names=list(X.columns), filled=True, rounded=True,
                  max_depth=3, fontsize=8, precision=2)
        mostrar(fig)

# ======================================================================
# 6. MODELO FINAL
# ======================================================================
with tabs[5]:
    st.header("Etapa 6 — Conocimiento: selección del modelo y conclusiones")
    st.subheader("17 · Tabla comparativa final (conjunto de prueba)")
    st.dataframe(R["tabla_final"].round(4), use_container_width=True)
    st.markdown("**Robustez / estabilidad (validación cruzada de 5 folds sobre train):**")
    st.dataframe(R["robustez"].round(4), use_container_width=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(R["y_test"], R["y_pred_lr"], alpha=0.5, label="Reg. Lineal")
    ax.scatter(R["y_test"], R["y_pred_ar"], alpha=0.5, label=f"Árbol d={mejor_prof}")
    ax.plot([3, 8], [3, 8], "k--", lw=1)
    ax.set_xlabel("quality real")
    ax.set_ylabel("quality predicho")
    ax.set_title("Real vs. predicho (test)")
    ax.legend()
    plt.tight_layout()
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        mostrar(fig)

    st.subheader("18 · Selección del modelo final")
    lr_te, ar_te = R["fila_lr"], R["fila_ar"]
    rob = R["robustez"]
    cv_lr_m, cv_ar_m = rob["R2 CV media"].iloc[0], rob["R2 CV media"].iloc[1]
    cv_lr_s, cv_ar_s = rob["R2 CV desvío"].iloc[0], rob["R2 CV desvío"].iloc[1]
    gap_lr, gap_ar = rob["Brecha R2 train-test"].iloc[0], rob["Brecha R2 train-test"].iloc[1]
    elige_arbol = R["elige_arbol"]
    modelo_final = f"Árbol de Decisión (max_depth={mejor_prof})" if elige_arbol else "Regresión Lineal Múltiple"
    sel = ar_te if elige_arbol else lr_te
    st.success(f"**Modelo final: {modelo_final}**")
    st.caption("Regla de decisión: se elige el árbol solo si mejora el R² de test en más de 0,02 **y** también la validación "
               "cruzada; si no, se prefiere el modelo más simple (parsimonia).")
    if elige_arbol:
        st.markdown(f"El árbol supera a la regresión en R² de test ({ar_te['R2']:.3f} vs. {lr_te['R2']:.3f}) y en validación "
                    f"cruzada ({cv_ar_m:.3f} vs. {cv_lr_m:.3f}); la mejora es consistente, por lo que se acepta su mayor "
                    "varianza y se mantiene podado para controlar el sobreajuste.")
    else:
        st.markdown(f"El árbol no supera de forma clara y consistente a la regresión (R² test {ar_te['R2']:.3f} vs. "
                    f"{lr_te['R2']:.3f}; R² CV {cv_ar_m:.3f} vs. {cv_lr_m:.3f}). Con métricas similares se prefiere la regresión por "
                    f"**parsimonia**: es más simple, **interpretable** (cada coeficiente da signo y magnitud del efecto), "
                    f"**más estable** (desvío en CV {cv_lr_s:.3f} vs. {cv_ar_s:.3f}) y con brecha train–test mínima "
                    f"({gap_lr:.3f} vs. {gap_ar:.3f}).")
    st.markdown(
        "| Criterio | Regresión Lineal | Árbol (podado) |\n|---|---|---|\n"
        "| Interpretabilidad | Alta: coeficiente = efecto marginal y signo | Alta en reglas, difícil de leer si es profundo |\n"
        "| Robustez | Estable; sensible a outliers y multicolinealidad | Alta varianza; muy sensible a `max_depth` |\n"
        "| No linealidades / interacciones | No las captura | Sí las captura |\n"
        "| Sobreajuste | Bajo | Riesgo alto si la profundidad no se controla |")

    st.subheader("19 · Conclusión")
    c1, c2, c3 = st.columns(3)
    c1.metric("R² (test)", f"{sel['R2']:.3f}")
    c2.metric("MAE (test)", f"{sel['MAE']:.3f}")
    c3.metric("RMSE (test)", f"{sel['RMSE']:.3f}")
    st.markdown(
        f"- Con **solo variables fisicoquímicas** la capacidad predictiva es **moderada**: el modelo final explica "
        f"aproximadamente el **{max(sel['R2'], 0):.0%}** de la variabilidad de `quality`.\n"
        "- La calidad sensorial depende también de factores **no medidos** (variedad de uva, añada, terroir, aromas, "
        "envejecimiento) y de la **subjetividad del panel de cata**; además `quality` es discreta y concentrada en 5–6.\n"
        "- Las variables más relevantes (`alcohol`, `volatile_acidity`, `sulphates`) coinciden en ambos modelos, lo que da "
        "consistencia al hallazgo.")
    st.markdown("**Posibles mejoras:** transformaciones (`log1p`) de variables asimétricas · interacciones/términos polinómicos · "
                "regularización (Ridge/Lasso) por la multicolinealidad · ensambles (Random Forest, Gradient Boosting) · "
                "plantear el problema como clasificación (baja/media/alta) · validación cruzada repetida · más variables "
                "(uva, añada, zona) y más datos en los extremos de la escala.")

# ======================================================================
# 7. PREGUNTAS
# ======================================================================
with tabs[6]:
    st.header("Preguntas de análisis")

    with st.expander("**Pregunta 20** — ¿Por qué eliminar duplicados exactos antes de dividir en train/test?", expanded=True):
        st.markdown(
            "Si se divide con duplicados, **una copia puede quedar en entrenamiento y otra en prueba**: el modelo ya vio ese "
            "dato, por lo que acierta sin haber generalizado. Es una **fuga de datos (*data leakage*)**. Consecuencias: "
            "(a) el test deja de ser independiente y las métricas resultan **optimistas**; (b) se **favorece artificialmente a "
            "los modelos que memorizan** (árbol profundo, k-NN); (c) las filas repetidas pesan más en el ajuste. "
            "Al deduplicar antes del split, el test contiene solo casos nunca vistos.")
        c1, c2 = st.columns(2)
        c1.metric("Filas de TEST con copia exacta en TRAIN (CON duplicados)", f"{R['pct_leak_con']:.1%}")
        c2.metric("Ídem tras deduplicar", f"{R['pct_leak_sin']:.1%}")
        st.caption("Matiz: en este dataset puede haber vinos distintos con mediciones idénticas por la baja resolución de "
                   "los instrumentos; aun así, la práctica conservadora es eliminarlos.")

    with st.expander("**Pregunta 21** — ¿Eliminar o conservar los outliers detectados por IQR?", expanded=True):
        st.markdown(
            f"**Se conservan.** (1) Quitarlos eliminaría **{R['pct_any']:.0%}** de las filas. (2) Son valores plausibles del "
            "dominio, no errores, y muchos corresponden a vinos de calidad extrema. (3) El árbol es robusto a extremos; la "
            "regresión lineal es más sensible, por eso se verifica empíricamente: se quitan los outliers **solo del train** "
            f"({R['out_train_n']:,} → {R['out_train_f']:,} filas) y se evalúa sobre el **mismo test**.")
        st.dataframe(R["comp_out"].round(4), use_container_width=True)
        co = R["comp_out"]
        mejora = max(co.loc["LR - sin outliers (train)", "R2"] - co.loc["LR - conservando outliers", "R2"],
                     co.loc["Árbol - sin outliers (train)", "R2"] - co.loc["Árbol - conservando", "R2"])
        st.markdown("La mejora de R² al quitarlos " + ("**no justifica** descartar datos." if mejora < 0.02 else
                    "existe pero **no compensa** la pérdida de información; alternativa: transformar (log) o usar modelos robustos (Huber)."))

    with st.expander("**Pregunta 22** — ¿Por qué escalar para interpretar los coeficientes pero no para el árbol?", expanded=True):
        st.markdown(
            "- **Regresión Lineal:** el coeficiente es el cambio en `quality` por **cada unidad** de la variable. Con escalas "
            "distintas (`density` varía ≈0,01 y `total_sulfur_dioxide` ≈280) los coeficientes **no son comparables**. Con "
            "`StandardScaler`, cada coeficiente es el cambio por **1 desvío estándar** y sus magnitudes se pueden comparar. "
            "El escalado no cambia predicciones ni métricas de OLS.\n"
            "- **Árbol:** cada nodo busca un umbral sobre una variable a la vez; las particiones son las mismas con o sin "
            "escalado (invarianza ante transformaciones monótonas).")
        st.dataframe(R["comp_coef"].round(4), use_container_width=True)
        st.markdown(f"¿Mismas predicciones de la regresión con y sin escalar? → **{R['ok_lr_scale']}**  \n"
                    f"¿Mismas predicciones del árbol con y sin escalar? → **{R['ok_ar_pred']}**  \n"
                    f"¿Mismas importancias del árbol? → **{R['ok_ar_imp']}**")

    with st.expander("**Pregunta 23** — ¿Qué significa un R² negativo?", expanded=True):
        full = R["res_arbol"].loc["sin límite"]
        st.latex(r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}}")
        st.markdown(
            "Compara el error del modelo con el de un **modelo trivial que siempre predice la media**. Un **R² negativo** "
            "indica que el modelo predice **peor que la media** (SS_res > SS_tot). Un árbol sin límite crece hasta una hoja por "
            "observación, **memoriza** el entrenamiento (R² train ≈ 1) y, si en test el R² es negativo o muy bajo, la brecha "
            "enorme delata **sobreajuste extremo** (varianza muy alta). Se controla con `max_depth`, `min_samples_leaf`, "
            "poda (`ccp_alpha`) o ensambles.")
        c1, c2, c3 = st.columns(3)
        c1.metric("R² train (sin límite)", f"{full['R2_train']:.3f}")
        c2.metric("R² test (sin límite)", f"{full['R2_test']:.3f}")
        c3.metric("Brecha", f"{full['Brecha R2']:.3f}")
        if full["R2_test"] >= 0:
            st.caption("En esta corrida el R² de test no resultó negativo, pero la brecha igualmente evidencia sobreajuste extremo.")

    with st.expander("**Pregunta 24** — Repetir la comparación final sin eliminar los duplicados", expanded=True):
        st.markdown("Se repite **todo el pipeline** (split estratificado, escalado solo para LR, profundidad por CV, "
                    "evaluación en test) con el dataset **con** duplicados y se compara contra el limpio.")
        comp = R["comparacion"]
        st.dataframe(comp.round(4), use_container_width=True)

        def g24(ds, key):
            sub = comp[comp["dataset"] == ds]
            return sub[sub.index.str.startswith(key)].iloc[0]

        L, D = "SIN duplicados (limpio)", "CON duplicados"
        win_L = "Regresión Lineal" if g24(L, "Regresión")["R2"] >= g24(L, "Árbol (")["R2"] else "Árbol podado"
        win_D = "Regresión Lineal" if g24(D, "Regresión")["R2"] >= g24(D, "Árbol (")["R2"] else "Árbol podado"
        full_L, full_D = g24(L, "Árbol sin")["R2"], g24(D, "Árbol sin")["R2"]
        lr_L, lr_D = g24(L, "Regresión")["R2"], g24(D, "Regresión")["R2"]
        st.markdown(
            f"El mejor modelo por R² de test es **{win_L}** con datos limpios y **{win_D}** con duplicados: la conclusión "
            + ("**no cambia**" if win_L == win_D else "**cambia**")
            + f". El R² del árbol sin límite pasa de {full_L:.3f} a {full_D:.3f} y el de la regresión de {lr_L:.3f} a {lr_D:.3f}"
            + (": los modelos que memorizan son los más beneficiados por la fuga, una ventaja artificial que no se sostendría "
               "en producción." if full_D > full_L else ": en esta partición el efecto no fue visible, aunque el riesgo de fuga persiste."))
        st.info("**Moraleja KDD:** una decisión temprana de limpieza puede alterar las métricas, el ranking de modelos y la "
                "confianza en las conclusiones; por eso se documenta y se hace *antes* de particionar. Como el split es "
                "aleatorio, conviene repetirlo con distintas semillas (probá cambiar la semilla en la barra lateral) o usar "
                "validación cruzada repetida para distinguir diferencias reales de ruido.")
