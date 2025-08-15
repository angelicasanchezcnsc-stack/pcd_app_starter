import os
import io
import pandas as pd
import plotly.express as px
import plotly.io as pio

# ======= Paleta institucional =======
PALETTE = ["#FFFFD9", "#EDF8B1", "#C7E9B4", "#7FCDBB",
           "#41B6C4", "#1D91C0", "#225EA8", "#0C2C84"]

# Configuración de colores para todos los gráficos Plotly
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE
pio.templates.default = "plotly_white"
import streamlit as st
from core.pipeline import run_pipeline, PipelineConfig

st.set_page_config(page_title="Analizador de Cuota PcD", layout="wide")

st.title("🔎 Analizador Ejecutivo de Cuota PcD — Entorno Web")
st.write(
    "Sube tus archivos, elige el proveedor de IA y ejecuta el análisis. "
    "Esta es una plantilla para migrar tu notebook de Colab a una app web real."
)

# --- API keys desde variables de entorno (configúralas en Render) ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Configuración")
    proveedor_ia = st.radio("Proveedor de IA", ["Ninguno", "OpenAI", "Gemini"], index=0)
    top_n = st.number_input("Top N empleos por territorio", min_value=1, max_value=50, value=10)
    score_minimo = st.slider("Puntaje mínimo de compatibilidad", 0, 100, 60)
    generar_pdf = st.checkbox("Generar reporte PDF", value=True)

    st.markdown("---")
    st.caption("Variables de entorno detectadas:")
    st.write({"OPENAI_API_KEY": bool(OPENAI_API_KEY), "GOOGLE_API_KEY": bool(GOOGLE_API_KEY)})

st.subheader("📂 Carga de archivos")
c1, c2 = st.columns(2)
with c1:
    f1 = st.file_uploader("Tabla 1 (personas/discapacidad) — Excel", type=["xls", "xlsx"])
with c2:
    f2 = st.file_uploader("Tabla 2 (empleos/vacantes) — Excel", type=["xls", "xlsx"])

if st.button("🚀 Ejecutar análisis", type="primary"):
    if not f1 or not f2:
        st.error("Por favor sube **ambos** archivos de Excel.")
        st.stop()

    try:
        df1 = pd.read_excel(f1)
        df2 = pd.read_excel(f2)
    except Exception as e:
        st.exception(e)
        st.stop()

    cfg = PipelineConfig(
        proveedor_ia=proveedor_ia.lower(),
        openai_api_key=OPENAI_API_KEY,
        google_api_key=GOOGLE_API_KEY,
        top_n=int(top_n),
        score_minimo=int(score_minimo),
        generar_pdf=generar_pdf,
    )

    with st.spinner("Procesando…"):
        result = run_pipeline(df1=df1, df2=df2, config=cfg)

    st.success("¡Listo! ✅")

    if "df_resultado" in result and result["df_resultado"] is not None:
        st.subheader("📊 Resultado")
        st.dataframe(result["df_resultado"], use_container_width=True)
        csv_bytes = result["df_resultado"].to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV", csv_bytes, file_name="resultado.csv", mime="text/csv")

    if result.get("fig"):
        st.subheader("📈 Visualización")
        st.plotly_chart(result["fig"], use_container_width=True)

    if result.get("pdf_bytes"):
        st.subheader("📄 Reporte PDF")
        st.download_button("Descargar PDF", result["pdf_bytes"], file_name="reporte.pdf", mime="application/pdf")

st.markdown("---")
st.caption("Plantilla creada por tu asistente. Sustituye `run_pipeline` con tu lógica del notebook.")
