# core/report.py
from io import BytesIO
from datetime import datetime
import pandas as pd
import plotly.express as px
from plotly.io import to_html
from fpdf import FPDF

# ---------- PDF opcional (arreglado para Python 3.13) ----------
def build_simple_pdf(title: str = "Reporte", resumen: str = "") -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, txt=title, ln=True, align="L")
    pdf.set_font("Arial", size=12)
    if resumen:
        pdf.multi_cell(0, 8, txt=f"Resumen: {resumen}")
    
    out = pdf.output(dest="S")  # en fpdf2 puede ser bytearray o str
    if isinstance(out, bytearray):
        return bytes(out)
    else:
        return out.encode("latin1")

# ---------- XLSX (datos) ----------
def df_to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "resultado") -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf.getvalue()

# ---------- HTML (reporte ejecutivo) ----------
def make_html_report(
    titulo: str,
    resumen_html: str,
    tablas_html: list[str] = None,
    figuras_html: list[str] = None,
) -> bytes:
    tablas_html = tablas_html or []
    figuras_html = figuras_html or []
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M")

    secciones_tablas = "\n".join(tablas_html)
    secciones_figs = "\n".join(figuras_html)

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{titulo}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px}}
 h1{{margin:.2rem 0 0}}
 .muted{{color:#666;margin:0 0 1rem}}
 section{{margin:1.2rem 0}}
 table{{border-collapse:collapse;width:100%;font-size:14px}}
 th,td{{border:1px solid #e5e5e5;padding:6px;text-align:left}}
 th{{background:#f7f7f7}}
 .kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}}
 .card{{border:1px solid #eee;padding:12px;border-radius:12px;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
</style>
</head>
<body>
  <header>
    <h1>{titulo}</h1>
    <p class="muted">Generado: {ahora}</p>
  </header>

  <section>
    {resumen_html}
  </section>

  <section>
    {secciones_figs}
  </section>

  <section>
    {secciones_tablas}
  </section>
</body>
</html>"""
    return html.encode("utf-8")

def fig_to_inline_html(fig) -> str:
    """Convierte una figura Plotly en HTML embebible (sin página completa)."""
    return to_html(fig, include_plotlyjs="cdn", full_html=False, default_width="100%", default_height="420px")
