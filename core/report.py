from fpdf import FPDF
from typing import Optional

def build_simple_pdf(title: str = "Reporte", resumen: str = "") -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, txt=title, ln=True, align='L')
    pdf.set_font("Arial", size=12)
    if resumen:
        pdf.multi_cell(0, 8, txt=f"Resumen: {resumen}")
    return pdf.output(dest='S').encode('latin1')
