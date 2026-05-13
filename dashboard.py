# dashboard.py — Shield-Agro | Cacao Intelligence
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import joblib
import numpy as np

st.set_page_config(
    page_title="Shield-Agro | Cacao Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# ── Estilo general ────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMetric { background:#1e2530; border-radius:10px; padding:10px; }
</style>
""", unsafe_allow_html=True)

# ── Carga de datos ────────────────────────────────────────
@st.cache_data
def cargar_datos():
    df = pd.read_excel("Resultados_ShieldAgro (1).xlsx")
    # Convertir números de Excel a fechas (Excel base date: 1899-12-30)
    df["Fecha"] = pd.Timestamp("1899-12-30") + pd.to_timedelta(df["Fecha"], unit="D")
    return df.sort_values("Fecha").reset_index(drop=True)

@st.cache_resource
def cargar_modelos():
    rf_m = joblib.load("modelos/rf_monilia.pkl")
    rf_p = joblib.load("modelos/rf_phytophthora.pkl")
    le_m = joblib.load("modelos/le_monilia.pkl")
    le_p = joblib.load("modelos/le_phytophthora.pkl")
    return rf_m, rf_p, le_m, le_p

df = cargar_datos()
rf_m, rf_p, le_m, le_p = cargar_modelos()

FEATURES = [
    "humedad", "Temperatura Media", "Temperatura Minima",
    "Temperatura Maxima", "radiacion", "viento",
    "presion superficial", "precipitacion", "VPD",
    "Delta_Temp", "GDA_diario", "GDA_Acumulado",
    "NDVI_Sentinel", "REIP_Sentinel", "NDWI_Sentinel"
]

COLORES = {"ALTO":"#e74c3c", "MEDIO":"#f39c12", "BAJO":"#27ae60",
           "ALTA":"#e74c3c", "MODERADA":"#f39c12", "BAJA":"#27ae60"}
EMOJI = {"ALTO":"🔴", "MEDIO":"🟠", "BAJO":"🟢",
           "ALTA":"🔴", "MODERADA":"🟠", "BAJA":"🟢"}

# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("# 🛡️ Shield-Agro | Cacao Intelligence")
st.markdown("**Monitoreo fitosanitario con IA — Esmeraldas, Ecuador 2024–2025**")
st.divider()

# ════════════════════════════════════════════════════════════
# PÁGINAS
# ════════════════════════════════════════════════════════════
pag1, pag2, pag3 = st.tabs([
    "📊 Análisis de datos",
    "🎯 Predecir riesgo hoy",
    "🗺️ Mapa Satelital"
])

# ════════════════════════════════════════════════════════════
# PÁGINA 1 — ANÁLISIS
# ════════════════════════════════════════════════════════════
with pag1:

    # Métricas rápidas arriba
    ultimo = df.iloc[-1]
    X_ult = np.array([[ultimo[f] for f in FEATURES]])
    pred_m_hoy = le_m.inverse_transform(rf_m.predict(X_ult))[0]
    pred_p_hoy = le_p.inverse_transform(rf_p.predict(X_ult))[0]

    st.subheader("📅 Estado de hoy")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🍫 Moniliasis", f"{EMOJI[pred_m_hoy]} {pred_m_hoy}")
    c2.metric("🍄 Phytophthora", f"{EMOJI[pred_p_hoy]} {pred_p_hoy}")
    c3.metric("🌡️ Temperatura", f"{ultimo['Temperatura Media']:.1f} °C")
    c4.metric("💧 Humedad", f"{ultimo['humedad']:.1f} %")

    st.divider()

    # Gráfico 1 — Conteo de alertas
    st.subheader("¿Cuántos días hubo cada nivel de riesgo?")
    conteo = df["Alerta_Monilia"].value_counts()

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    fig1.patch.set_facecolor("#0d1117")
    ax1.set_facecolor("#0d1117")
    bars = ax1.bar(conteo.index, conteo.values,
                   color=[COLORES[k] for k in conteo.index],
                   width=0.5, edgecolor="none")
    for bar, val in zip(bars, conteo.values):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 2, str(val),
                 ha="center", color="white", fontsize=13, fontweight="bold")
    ax1.tick_params(colors="white", labelsize=12)
    ax1.set_ylabel("Número de días", color="white")
    ax1.spines[:].set_visible(False)
    ax1.grid(axis="y", alpha=0.15)
    st.pyplot(fig1)

    st.divider()

    # Gráfico 2 — Tendencia últimos 60 días
    st.subheader("Tendencia de riesgo — últimos 60 días")
    ultimos = df.tail(60)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    fig2.patch.set_facecolor("#0d1117")
    ax2.set_facecolor("#0d1117")
    ax2.plot(ultimos["Fecha"], ultimos["IFE_Monilia"],
             color="#1abc9c", linewidth=1.5, alpha=0.5)
    for nivel, color in COLORES.items():
        if nivel in ["ALTO","MEDIO","BAJO"]:
            mask = ultimos["Alerta_Monilia"] == nivel
            ax2.scatter(ultimos[mask]["Fecha"],
                        ultimos[mask]["IFE_Monilia"],
                        color=color, label=nivel, s=25, alpha=0.9)
    ax2.axhline(0.80, color="#e74c3c", linestyle="--",
                linewidth=1, label="Umbral ALTO")
    ax2.axhline(0.75, color="#f39c12", linestyle="--",
                linewidth=1, label="Umbral MEDIO")
    ax2.tick_params(colors="white")
    ax2.set_ylabel("Nivel de riesgo", color="white")
    ax2.legend(facecolor="#21262d", labelcolor="white", fontsize=9)
    ax2.spines[:].set_visible(False)
    ax2.grid(alpha=0.15)
    st.pyplot(fig2)

    st.divider()

    # Gráfico 3 — Lluvia vs Riesgo
    st.subheader("🌧️ ¿Cómo afecta la lluvia al riesgo?")
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    fig3.patch.set_facecolor("#0d1117")
    ax3.set_facecolor("#0d1117")
    for nivel, color in COLORES.items():
        if nivel in ["ALTO","MEDIO","BAJO"]:
            mask = df["Alerta_Monilia"] == nivel
            ax3.scatter(df[mask]["precipitacion"],
                        df[mask]["IFE_Monilia"],
                        color=color, label=nivel, alpha=0.6, s=18)
    ax3.tick_params(colors="white")
    ax3.set_xlabel("Lluvia del día (mm)", color="white")
    ax3.set_ylabel("Nivel de riesgo", color="white")
    ax3.legend(facecolor="#21262d", labelcolor="white")
    ax3.spines[:].set_visible(False)
    ax3.grid(alpha=0.15)
    st.pyplot(fig3)

# ════════════════════════════════════════════════════════════
# PÁGINA 2 — PREDICCIÓN
# ════════════════════════════════════════════════════════════
with pag2:
    st.subheader("🎯 Ingresa los datos de tu campo")
    st.caption("Mueve los controles y el modelo predice el riesgo en tiempo real")

    ultimo = df.iloc[-1]
    col1, col2 = st.columns(2)

    with col1:
        humedad = st.slider("💧 Humedad del aire (%)",
                            60.0, 100.0, float(ultimo["humedad"]))
        lluvia = st.slider("🌧️ Lluvia (mm)",
                            0.0, 50.0, float(ultimo["precipitacion"]))
        temp = st.slider("🌡️ Temperatura (°C)",
                            20.0, 35.0, float(ultimo["Temperatura Media"]))
    with col2:
        viento = st.slider("💨 Viento (m/s)",
                            0.5, 5.0, float(ultimo["viento"]))
        vpd = st.slider("🌫️ Estrés hídrico (VPD)",
                            0.1, 1.5, float(ultimo["VPD"]))
        gda = st.slider("📊 GDA Acumulado",
                            0.0, 550.0, float(ultimo["GDA_Acumulado"]))

    X = np.array([[
        humedad, temp,
        float(ultimo["Temperatura Minima"]),
        float(ultimo["Temperatura Maxima"]),
        float(ultimo["radiacion"]), viento,
        float(ultimo["presion superficial"]), lluvia,
        vpd, float(ultimo["Delta_Temp"]),
        float(ultimo["GDA_diario"]), gda,
        float(ultimo["NDVI_Sentinel"]),
        float(ultimo["REIP_Sentinel"]),
        float(ultimo["NDWI_Sentinel"])
    ]])

    pred_m = le_m.inverse_transform(rf_m.predict(X))[0]
    pred_p = le_p.inverse_transform(rf_p.predict(X))[0]

    mensajes = {
        "ALTO": "⚠️ Revisa tus mazorcas hoy mismo.",
        "MEDIO": "👁️ Monitorea cada 2 días.",
        "BAJO": "✅ Condiciones favorables hoy."
    }
    colores_bg = {
        "ALTO": "#e74c3c", "MEDIO": "#f39c12", "BAJO": "#27ae60",
        "ALTA": "#e74c3c", "MODERADA": "#f39c12", "BAJA": "#27ae60"
    }

    st.divider()
    r1, r2 = st.columns(2)

    with r1:
        color = colores_bg.get(pred_m, "#888")
        st.markdown(f"""
        <div style="background:{color}22; border-left:5px solid {color};
                    border-radius:10px; padding:20px;">
            <h3 style="color:{color}; margin:0">
                🍫 Moniliasis — {EMOJI[pred_m]} {pred_m}
            </h3>
            <p style="color:white; margin-top:10px; font-size:16px">
                {mensajes.get(pred_m,'')}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        color2 = colores_bg.get(pred_p, "#888")
        st.markdown(f"""
        <div style="background:{color2}22; border-left:5px solid {color2};
                    border-radius:10px; padding:20px;">
            <h3 style="color:{color2}; margin:0">
                🍄 Phytophthora — {EMOJI[pred_p]} {pred_p}
            </h3>
            <p style="color:white; margin-top:10px; font-size:16px">
                {mensajes.get(pred_p,'')}
            </p>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PÁGINA 3 — MAPA
# ════════════════════════════════════════════════════════════
with pag3:
    st.subheader("🗺️ Mapa Satelital — Esmeraldas, Ecuador")
    st.caption("Análisis de índices espectrales NDVI · NDWI · REIP con Sentinel-2")

    try:
        with open("mapa_satelital.html", "r", encoding="utf-8") as f:
            mapa_html = f.read()
        st.components.v1.html(mapa_html, height=580, scrolling=False)
    except:
        st.error("No se encontró el archivo mapa_satelital.html")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption("🛡️ Shield-Agro Precisión | Cacao Intelligence · Esmeraldas, Ecuador · 2025")
