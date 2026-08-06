# =============================================================================
# MÓDULO 5 — EDA Amazon Products
# Curso: Diagnóstico y Predictibilidad | 92-0030
# Semana 5 | I Cuatrimestre 2026 | Prof. Robin Sequeira
#
# Dataset: amazon.csv — 1,466 productos de Amazon India
# Variable objetivo: rating (calificación del producto, de 1 a 5)
#
# ANTES DE EJECUTAR:
#   1. Confirma que el cluster de Databricks esté activo (esquina superior derecha)
#   2. Verifica que amazon.csv esté disponible en la ruta PATH de abajo
#   3. Ejecuta las celdas en orden: Run All o celda por celda con Shift+Enter
# =============================================================================


# =============================================================================
# CELDA 1 — Librerías
# Importamos todo lo que vamos a necesitar en una sola celda al inicio.
# Si seaborn no está disponible, el heatmap se imprime como tabla de texto.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

try:
    import seaborn as sns
    SEABORN_OK = True
except ImportError:
    SEABORN_OK = False

# Ajuste de visualización: mostrar todas las columnas sin truncar
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 140)

print("✅ Librerías cargadas correctamente.")


# =============================================================================
# CELDA 2 — Carga del dataset
# Leemos el CSV y guardamos una copia limpia en df.
# NUNCA trabajamos sobre df_raw directamente: si nos equivocamos al limpiar,
# podemos volver a hacer df = df_raw.copy() sin recargar el archivo.
# =============================================================================

df_raw = spark.table("workspace.default.amazon").toPandas()


df = df_raw.copy()

print(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
df.head()


# =============================================================================
# CELDA 3 — Normalizar nombres de columnas
# Convertimos todos los nombres a minúsculas y reemplazamos espacios por _.
# Esto evita errores como escribir df["Discounted Price"] en vez de
# df["discounted_price"]. Las mayúsculas y espacios causan muchos bugs.
# =============================================================================

df.columns = (
    df.columns
    .str.strip()          # elimina espacios al inicio y al final
    .str.lower()          # convierte a minúsculas
    .str.replace(" ", "_") # reemplaza espacios internos por _
)

print("Columnas disponibles:")
print(df.columns.tolist())


# =============================================================================
# CELDA 4 — Diagnóstico de calidad
# Antes de limpiar, entendemos qué tan sucio está el dataset.
# isna().mean() * 100  →  porcentaje de nulos por columna
# duplicated().sum()   →  filas completamente repetidas
# nunique()            →  valores únicos por columna
#                         (alto = posible columna ID que no aporta al modelo)
# =============================================================================

print("NULOS POR COLUMNA (%):")
nulos = df.isna().mean().sort_values(ascending=False) * 100
print(nulos[nulos > 0].round(2).to_string())

print(f"\nFilas duplicadas: {df.duplicated().sum()}")

print("\nVALORES ÚNICOS POR COLUMNA (cardinalidad):")
print(df.nunique().sort_values(ascending=False).to_string())


# =============================================================================
# CELDA 5 — Funciones de limpieza con regex
#
# PROBLEMA: Las columnas de precio llegan como texto, por ejemplo '₹2,499'.
# Python no puede hacer matemáticas con ese texto. Necesitamos el número 2499.0.
#
# SOLUCIÓN: Expresiones regulares (regex) para eliminar el texto no numérico.
#
# Cómo funciona re.sub(patrón, reemplazo, texto):
#   - Busca el patrón en el texto
#   - Reemplaza cada coincidencia con el reemplazo (aquí, string vacío '')
#   - Devuelve el texto limpio
#
# El patrón r"[^\d\.\-]" significa: "cualquier carácter que NO sea
# un dígito (0-9), un punto (.) o un signo negativo (-)"
#
# pd.to_numeric(..., errors='coerce'):
#   - Convierte el string limpio a número
#   - Si no puede convertir (valor raro o vacío), pone NaN en vez de dar error
# =============================================================================

def clean_currency(series):
    """
    Convierte columnas de precio: '₹2,499' → 2499.0
    Elimina símbolo de moneda, comas y cualquier texto.
    """
    s = series.astype("string")
    s = s.str.replace(r"[^\d\.\-]", "", regex=True)  # deja solo números y punto
    s = s.replace("", pd.NA)                          # vacíos → NaN
    return pd.to_numeric(s, errors="coerce")


def clean_percent(series):
    """
    Convierte columnas de porcentaje: '54%' → 54.0
    Elimina el símbolo % y cualquier texto adicional.
    """
    s = series.astype("string")
    s = s.str.replace("%", "", regex=False)           # elimina solo el %
    s = s.str.replace(r"[^\d\.\-]", "", regex=True)  # elimina cualquier otro texto
    s = s.replace("", pd.NA)
    return pd.to_numeric(s, errors="coerce")


def clean_float(series):
    """
    Convierte la columna rating: '3.9 out of 5 stars' → 3.9
    Extrae el primer número flotante que aparezca en el texto.
    """
    s = series.astype("string")
    s = s.str.replace(r"[^\d\.\-]", "", regex=True)  # deja solo el número
    s = s.replace("", pd.NA)
    return pd.to_numeric(s, errors="coerce")


print("✅ Funciones de limpieza definidas: clean_currency(), clean_percent(), clean_float()")


# =============================================================================
# CELDA 6 — Aplicar la limpieza
# apply(función) aplica la función fila por fila a toda la columna.
# Es equivalente a un for-loop sobre las filas, pero mucho más rápido.
# Después verificamos con dtypes: deben decir float64, no object.
# =============================================================================

df["discounted_price"]    = clean_currency(df["discounted_price"])
df["actual_price"]        = clean_currency(df["actual_price"])
df["discount_percentage"] = clean_percent(df["discount_percentage"])
df["rating"]              = clean_float(df["rating"])
df["rating_count"]        = clean_currency(df["rating_count"])

# Verificación: si alguna columna sigue siendo 'object', la limpieza falló
print("TIPOS DE DATOS DESPUÉS DE LIMPIAR:")
cols = ["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]
print(df[cols].dtypes.to_string())

print("\nMuestra de valores limpios (primeras 5 filas):")
df[cols].head()


# =============================================================================
# CELDA 7 — Análisis univariado: distribución del rating
# El rating es la variable objetivo. Siempre la analizamos primero.
# El histograma nos muestra si los ratings están sesgados hacia valores altos
# (sesgo positivo) o distribuidos de forma uniforme entre 1 y 5.
# La línea punteada marca el promedio para ver el sesgo visualmente.
# =============================================================================

plt.figure(figsize=(8, 4))
plt.hist(df["rating"].dropna(), bins=20, color="#E8820C", edgecolor="white")
media_rating = df["rating"].mean()
plt.axvline(media_rating, color="#3B1F5E", linestyle="--", linewidth=2,
            label=f"Promedio: {media_rating:.2f}")
plt.title("Distribución del Rating — Amazon Products")
plt.xlabel("Rating (1 a 5)")
plt.ylabel("Número de productos")
plt.legend()
plt.tight_layout()
plt.show()

print(df["rating"].describe().round(3).to_string())


# =============================================================================
# CELDA 8 — Análisis univariado: distribución del precio
# Los precios tienen distribución sesgada a la derecha: muchos productos baratos
# y pocos muy caros. Por eso usamos escala logarítmica en el eje X.
# El boxplot detecta outliers usando los bigotes del diagrama.
# Combinamos ambos gráficos para tener una vista completa.
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histograma con escala logarítmica
precios = df["actual_price"].dropna()
precios = precios[precios > 0]  # eliminamos ceros para que log no dé error
axes[0].hist(np.log1p(precios), bins=25, color="#3B1F5E", edgecolor="white")
axes[0].set_title("Distribución del Precio Real (escala log)")
axes[0].set_xlabel("log(1 + precio)")
axes[0].set_ylabel("Número de productos")

# Boxplot horizontal
axes[1].boxplot(precios, vert=False, patch_artist=True,
                boxprops=dict(facecolor="#3B1F5E", color="#E8820C"),
                medianprops=dict(color="#E8820C", linewidth=2),
                flierprops=dict(marker="o", color="#E8820C", alpha=0.4))
axes[1].set_title("Boxplot del Precio Real (outliers en naranja)")
axes[1].set_xlabel("Precio (₹)")

plt.tight_layout()
plt.show()

# Detección de outliers con la regla IQR del Módulo 2
q1  = df["actual_price"].quantile(0.25)
q3  = df["actual_price"].quantile(0.75)
iqr = q3 - q1
outliers = ((df["actual_price"] < q1 - 1.5 * iqr) |
            (df["actual_price"] > q3 + 1.5 * iqr)).sum()
print(f"Outliers detectados en precio: {outliers}")
print(f"Q1: ₹{q1:,.0f}  |  Q3: ₹{q3:,.0f}  |  IQR: ₹{iqr:,.0f}")


# =============================================================================
# CELDA 9 — Análisis bivariado: precio vs rating
# ¿Los productos más caros reciben mejor calificación?
# El scatter plot cruza dos variables numéricas.
# Cada punto es un producto. Eje X = precio, Eje Y = rating.
# Usamos escala logarítmica en X porque los precios tienen mucha dispersión.
# El coeficiente de correlación nos confirma si hay relación lineal.
# =============================================================================

df_plot1 = df[["actual_price", "rating"]].dropna()
df_plot1 = df_plot1[df_plot1["actual_price"] > 0]

plt.figure(figsize=(9, 5))
plt.scatter(df_plot1["actual_price"], df_plot1["rating"],
            alpha=0.3, color="#E8820C", s=20)
plt.xscale("log")
plt.xlabel("Precio real — escala logarítmica (₹)")
plt.ylabel("Rating")
plt.title("Precio vs Rating — ¿Los productos más caros se califican mejor?")
plt.tight_layout()
plt.show()

# Correlación de Pearson entre precio y rating
corr_precio = df_plot1.corr().loc["actual_price", "rating"]
print(f"Correlación precio vs rating: {corr_precio:.3f}")
print("Interpretación: cercano a 0 = no hay relación lineal entre precio y rating.")


# =============================================================================
# CELDA 10 — Análisis bivariado: descuento vs rating
# ¿Un mayor descuento genera mejor o peor calificación?
# Esta celda responde la hipótesis que planteamos al inicio de la clase.
# Misma lógica que la celda anterior, pero con discount_percentage en el eje X.
# =============================================================================

df_plot2 = df[["discount_percentage", "rating"]].dropna()

plt.figure(figsize=(9, 5))
plt.scatter(df_plot2["discount_percentage"], df_plot2["rating"],
            alpha=0.3, color="#3B1F5E", s=20)
plt.xlabel("Descuento (%)")
plt.ylabel("Rating")
plt.title("Descuento vs Rating — ¿El descuento alto mejora la percepción?")
plt.tight_layout()
plt.show()

corr_desc = df_plot2.corr().loc["discount_percentage", "rating"]
print(f"Correlación descuento vs rating: {corr_desc:.3f}")
print("Interpretación: si es cercano a 0, el descuento no predice el rating de forma lineal.")


# =============================================================================
# CELDA 11 — Heatmap de correlación (análisis multivariado)
# Visualizamos TODAS las correlaciones entre variables numéricas al mismo tiempo.
# Importante: select_dtypes(include='number') filtra solo columnas numéricas.
# Sin ese filtro, corr() da error si hay columnas con texto.
# La tabla al final muestra qué variable tiene más relación con el rating.
# =============================================================================

# Solo columnas numéricas
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr     = df[num_cols].corr()

if SEABORN_OK:
    plt.figure(figsize=(8, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Heatmap de Correlación — Amazon Products")
    plt.tight_layout()
    plt.show()
else:
    # Si seaborn no está disponible, imprimimos la matriz como tabla
    print("Matriz de correlación:")
    print(corr.round(2).to_string())

print("\nCorrelaciones con la variable objetivo 'rating' (de mayor a menor):")
print(corr["rating"].sort_values(ascending=False).round(3).to_string())


# =============================================================================
# CELDA 12 — Resumen de hallazgos
# Documentamos los resultados clave del EDA.
# Este bloque también sirve como guía para la presentación grupal:
# cada equipo debe poder responder estas preguntas con los números reales.
# =============================================================================

print("=" * 60)
print("RESUMEN EDA — Amazon Products | Módulo 5 Semana 5")
print("=" * 60)
print(f"  Productos analizados : {len(df)}")
print(f"  Rating promedio      : {df['rating'].mean():.2f} / 5.0")
print(f"  Rating mediano       : {df['rating'].median():.2f}")
print(f"  Precio promedio      : ₹{df['actual_price'].mean():,.0f}")
print(f"  Descuento promedio   : {df['discount_percentage'].mean():.1f}%")
print(f"  Outliers en precio   : {outliers}")
print()
print("Correlaciones con rating:")
print(corr["rating"].drop("rating").sort_values(ascending=False).round(3).to_string())
print()
print("Preguntas para la presentación grupal:")
print("  1. ¿El precio predice de forma lineal el rating?")
print("  2. ¿El descuento mejora o empeora la percepción del cliente?")
print("  3. ¿Qué variable tiene más correlación con el rating?")

# =============================================================================
# FIN DEL CÓDIGO — MÓDULO 5
# Próximo paso: Módulo 6 — Regresión Lineal con el dataset Insurance
# =============================================================================
