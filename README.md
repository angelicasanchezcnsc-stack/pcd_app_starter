# Analizador Ejecutivo de Cuota PcD — Plantilla Web (Render)

Esta plantilla convierte tu notebook de Colab en una **app web real** usando **Streamlit** y despliegue en **Render**.

## Estructura
```
.
├── app.py                   # UI Streamlit
├── core/
│   ├── pipeline.py          # 🔁 Pega aquí tu lógica (reemplaza run_pipeline)
│   └── report.py            # Generación de PDF básica (fpdf2)
├── requirements.txt
├── render.yaml              # Config para despliegue en Render
└── .streamlit/config.toml
```

## Paso a paso (rápido)
1. **Pega tu lógica** del notebook en `core/pipeline.py` dentro de `run_pipeline(...)`.
2. **Configura variables** de entorno (no las subas al repo): `OPENAI_API_KEY` y/o `GOOGLE_API_KEY` si usas IA.
3. **Prueba local**:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   # abre http://localhost:8501
   ```
4. **Sube a GitHub** (nuevo repo).
5. **Render → New → Web Service**: conecta tu repo.
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - Añade `OPENAI_API_KEY` y `GOOGLE_API_KEY` en Variables de entorno.
6. **Deploy** y comparte la URL pública con tu equipo.

## Consejos de migración desde Colab
- Quita `!pip ...` y `from google.colab import ...`.
- Reemplaza `ipywidgets` por widgets de Streamlit (`st.selectbox`, `st.slider`, etc.).
- Lee archivos subidos con `st.file_uploader(...)` (ejemplo en `app.py`).
- Guarda resultados para descarga con `st.download_button(...)`.
- Si usas archivos locales persistentes en Render, necesitarás un **Render Disk** o un bucket (S3/GCS).

¡Éxitos con tu despliegue! 🚀
