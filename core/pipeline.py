from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import os
import pandas as pd
import numpy as np
import plotly.express as px

# ========= CONSTANTES MÍNIMAS (de tu notebook v22.0) =========
COL_OPEC = 'NO. OPEC'
COL_DENOMINACION = 'DENOMINACION'
COL_PROPOSITO = 'PROPOSITO'
COL_FUNCIONES = 'FUNCIONES'
COL_VACANTES = 'CANTIDAD DE VACANTES'
COL_MODALIDAD = 'MODALIDAD'
COL_NIVEL = 'NIVEL JERARQUICO'
COL_FUNCIONES_LIMPIAS = 'FUNCIONES_LIMPIAS'
COL_NIVEL_EDUCATIVO = 'NIVEL EDUCATIVO'
COL_MUNICIPIO = 'MUNICIPIO'

# ========= HELPERS BÁSICOS =========
def limpiar_texto_funciones(txt: str) -> str:
    if txt is None:
        return ""
    t = str(txt).replace("\n", " ").replace("\r", " ").strip()
    return " ".join(t.split())

# ========= FUNCIONES TRAÍDAS DEL NOTEBOOK =========
def normalizar_texto_compatible(texto: str) -> str:
    """
    Normaliza texto eliminando tildes y caracteres especiales.
    Compatible con sistemas que no tienen unidecode.
    """
    if not isinstance(texto, str):
        return ""

    if UNIDECODE_AVAILABLE:
        return unidecode(texto.upper().strip())
    else:
        # Normalización básica sin unidecode
        replacements = {
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'á': 'A', 'é': 'E', 'í': 'I', 'ó': 'O', 'ú': 'U',
            'Ñ': 'N', 'ñ': 'N'
        }
        texto_normalizado = texto.upper().strip()
        for old, new in replacements.items():
            texto_normalizado = texto_normalizado.replace(old, new)
        return texto_normalizado


def validar_estructura_minsalud(df_minsalud: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Valida que el archivo MinSalud tenga la estructura correcta"""
    columnas_requeridas = {
        COL_CONDICION, COL_ESCOLARIDAD_MINSALUD,
        COL_DEPTO, COL_MUNICIPIO_MINSALUD, COL_CANTIDAD_PCD
    }

    errores = []
    if df_minsalud is None or df_minsalud.empty:
        errores.append("Archivo MinSalud vacío o no cargado")
        return False, errores

    # Normalizar nombres de columnas
    df_minsalud.columns = [str(c).upper().strip() for c in df_minsalud.columns]

    columnas_faltantes = columnas_requeridas - set(df_minsalud.columns)
    if columnas_faltantes:
        errores.append(f"Faltan columnas en MinSalud: {', '.join(columnas_faltantes)}")
        return False, errores

    return True, []

print("✅ Funciones de normalización territorial configuradas")

def limpiar_y_validar_dataframe(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Valida la estructura y contenido del DataFrame de entrada"""
    df_limpio = df.copy()
    errores = []

    columnas_requeridas = {
        COL_OPEC, COL_PROPOSITO, COL_FUNCIONES, COL_VACANTES,
        COL_MODALIDAD, COL_NIVEL, COL_DENOMINACION
    }

    # Normalizar nombres de columnas
    df_limpio.columns = [str(c).upper().strip() for c in df_limpio.columns]

    # Verificar columnas faltantes
    faltantes = columnas_requeridas - set(df_limpio.columns)
    if faltantes:
        errores.append(f"Faltan columnas: {', '.join(faltantes)}")
        return None, errores

    # Limpiar y validar datos
    df_limpio[COL_NIVEL] = df_limpio[COL_NIVEL].astype(str).str.upper().str.strip()
    df_limpio[COL_MODALIDAD] = df_limpio[COL_MODALIDAD].astype(str).str.strip().str.upper()
    df_limpio[COL_DENOMINACION] = df_limpio[COL_DENOMINACION].astype(str).str.upper().str.strip()
    df_limpio[COL_OPEC] = df_limpio[COL_OPEC].astype(str).str.strip()
    df_limpio[COL_VACANTES] = pd.to_numeric(df_limpio[COL_VACANTES], errors='coerce').fillna(0).astype(int)

    # Validaciones
    if (df_limpio[COL_VACANTES] < 0).any():
        errores.append("Valores negativos en cantidad de vacantes")

    if (df_limpio[COL_PROPOSITO].astype(str).str.strip() == '').any():
        errores.append("Filas con PROPOSITO vacío")

    # Filtrar solo vacantes válidas
    df_valido = df_limpio[df_limpio[COL_VACANTES] > 0].copy()
    df_valido[COL_FUNCIONES_LIMPIAS] = df_valido[COL_FUNCIONES].apply(limpiar_texto_funciones)

    return df_valido, errores


def analizar_empleos_con_ia_territorial(
    df_unique_jobs: pd.DataFrame,
    df_minsalud: pd.DataFrame,
    provider: str,
    model_name: str,
    municipio: str,
    tipo_entidad: str,
    batch_size: int = 3
) -> List[Dict]:
    """Función principal para análisis territorial con IA"""

    if provider == 'Gemini':
        modelo = configurar_gemini_territorial(model_name)
        if not modelo:
            return []
    elif provider == 'OpenAI':
        cliente = configurar_openai_territorial(model_name)
        if not cliente:
            return []
    else:
        display(Markdown(f"❌ Proveedor '{provider}' no reconocido"))
        return []

    # Calcular contexto territorial
    predominancia = calcular_predominancia_territorial(df_minsalud, municipio)

    # Identificar empleos con provisionalidad (si existe la columna)
    empleos_provisionalidad = []
    empleos_no_provistos = []

    if COL_ESTADO_PROVISION in df_unique_jobs.columns:
        empleos_provisionalidad = df_unique_jobs[
            df_unique_jobs[COL_ESTADO_PROVISION].str.contains('PROVISIONAL', na=False)
        ].to_dict('records')
        empleos_no_provistos = df_unique_jobs[
            df_unique_jobs[COL_ESTADO_PROVISION].str.contains('NO_PROVISTO|SIN_PROVEER', na=False)
        ].to_dict('records')

    all_results = []
    num_lotes = math.ceil(len(df_unique_jobs) / batch_size)

    descripcion_progreso = f"🤖 Analizando con {provider} (Territorial)"
    barra_progreso = tqdm(range(num_lotes), desc=descripcion_progreso) if WIDGETS_AVAILABLE else range(num_lotes)

    for i in barra_progreso:
        if WIDGETS_AVAILABLE:
            barra_progreso.set_description(f"🤖 {provider} Territorial (Lote {i+1}/{num_lotes})")
        else:
            display(Markdown(f"🤖 Analizando con {provider}... Lote {i+1}/{num_lotes}"))

        start_index = i * batch_size
        end_index = start_index + batch_size
        batch = df_unique_jobs.iloc[start_index:end_index]

        # Generar prompt territorial
        prompt_territorial = generar_prompt_territorial_gemini(
            municipio, tipo_entidad, predominancia, {},
            empleos_provisionalidad, empleos_no_provistos
        )

        # Preparar datos del lote
        empleos_texto = "\n".join([
            f"- OPEC: {row[COL_OPEC]}, Denominación: {row[COL_DENOMINACION]}, "
            f"Propósito: {row[COL_PROPOSITO]}, Funciones: {row[COL_FUNCIONES_LIMPIAS]}, "
            f"Nivel Educativo: {row.get(COL_NIVEL_EDUCATIVO, 'No especificado')}"
            for _, row in batch.iterrows()
        ])

        try:
            if provider == 'Gemini':
                response = modelo.generate_content(f"{prompt_territorial}\n\nAnaliza estos empleos:\n{empleos_texto}")
                json_text = response.text

                # Limpiar respuesta de Gemini
                json_match = re.search(r'```json\s*(.*?)\s*```', json_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)

                parsed_json = json.loads(json_text)
                all_results.extend(parsed_json.get("analisis_empleos", []))

            elif provider == 'OpenAI':
                response = cliente.chat.completions.create(
                    model=model_name,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": prompt_territorial},
                        {"role": "user", "content": f"Analiza estos empleos:\n{empleos_texto}"}
                    ],
                    temperature=0.1
                )

                json_text = response.choices[0].message.content
                parsed_json = json.loads(json_text)
                all_results.extend(parsed_json.get("analisis_empleos", []))

            time.sleep(2)  # Respetar límites de API

        except Exception as e:
            display(Markdown(f"❌ Error en lote {i+1}: {str(e)[:100]}..."))
            continue

    return all_results

print("✅ Cliente OpenAI territorial configurado")

def calcular_score_compatibilidad_territorial(analisis_detallado: List[Dict]) -> float:
    """Calcula score de compatibilidad considerando contexto territorial"""
    if not isinstance(analisis_detallado, list):
        return 0.0

    score = 0.0
    for detalle in analisis_detallado:
        afinidad = detalle.get('afinidad', 'Baja')
        contexto_territorial = detalle.get('contexto_territorial', '')

        # Score base por afinidad
        if afinidad == 'Alta':
            score_base = 3.0
        elif afinidad == 'Media':
            score_base = 1.5
        else:
            score_base = 0.0

        # Bonificación territorial
        bonificacion = 0.0
        if 'disponibilidad alta' in contexto_territorial.lower():
            bonificacion = 0.5
        elif 'predominante' in contexto_territorial.lower():
            bonificacion = 0.3
        elif 'disponible' in contexto_territorial.lower():
            bonificacion = 0.2

        score += score_base + bonificacion

    return min(score, 14.0)  # Máximo posible: 7 tipos * 2 puntos


def crear_grafico_predominancia_territorial(predominancia: Dict[str, float], municipio: str) -> str:
    """Crea gráfico de barras de predominancia territorial"""
    if not VIS_AVAILABLE or not predominancia:
        return "<p>Gráfico no disponible (datos insuficientes)</p>"

    try:
        fig = go.Figure(data=[
            go.Bar(
                x=list(predominancia.keys()),
                y=list(predominancia.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'],
                text=[f"{v:.1f}%" for v in predominancia.values()],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title=f"Predominancia de Discapacidad en {municipio}",
            xaxis_title="Tipo de Discapacidad",
            yaxis_title="Porcentaje (%)",
            height=400,
            font=dict(family="Arial, sans-serif", size=12)
        )

        return fig.to_html(include_plotlyjs='cdn', div_id="grafico_predominancia")
    except Exception as e:
        return f"<p>Error generando gráfico: {str(e)}</p>"


def mapear_discapacidad_a_legal(condicion_minsalud: str) -> str:
    """Mapea subcategorías de MinSalud a las 7 categorías legales"""
    condicion_normalizada = normalizar_texto_compatible(str(condicion_minsalud))
    return MAPEO_DISCAPACIDAD.get(condicion_normalizada, "Múltiple")


def detectar_tipo_entidad_automatico(nombre_entidad: str) -> str:
    """Detecta automáticamente si es entidad nacional o territorial"""
    entidades_nacionales = [
        'ICBF', 'SENA', 'DIAN', 'INVIAS', 'INPEC', 'POLICIA',
        'EJERCITO', 'ARMADA', 'FUERZA AEREA', 'FISCALIA',
        'PROCURADURIA', 'CONTRALORIA', 'DANE', 'COLCIENCIAS',
        'MINTIC', 'MINSALUD', 'MINEDUCACION', 'MINTRABAJO',
        'SUPERINTENDENCIA', 'AGENCIA NACIONAL', 'INSTITUTO NACIONAL'
    ]

    nombre_upper = normalizar_texto_compatible(nombre_entidad)
    for entidad in entidades_nacionales:
        if entidad in nombre_upper:
            return "NACIONAL"
    return "TERRITORIAL"


def ejecutar_asignacion_estratificada(df_merged: pd.DataFrame, cuotas_por_nivel_y_modalidad: pd.DataFrame, total_reserva: int) -> pd.DataFrame:
    """Ejecuta la asignación estratificada final"""
    asignados_list = []
    id_vacantes_asignadas = set()
    cuotas_restantes = cuotas_por_nivel_y_modalidad.copy()

    # Asignación por estrato
    for nivel in cuotas_restantes.index:
        for modalidad in ['ABIERTO', 'ASCENSO']:
            if modalidad not in cuotas_restantes.columns:
                continue

            cuota_requerida = cuotas_restantes.loc[nivel, modalidad]
            if cuota_requerida <= 0:
                continue

            df_segmento = df_merged[
                (df_merged[COL_NIVEL] == nivel) &
                (df_merged[COL_MODALIDAD] == modalidad) &
                (~df_merged['ID_VACANTE_UNICA'].isin(id_vacantes_asignadas))
            ]

            vacantes_asignar = df_segmento.head(int(cuota_requerida))
            for _, vacante in vacantes_asignar.iterrows():
                vacante_dict = vacante.to_dict()
                vacante_dict['Estado_Asignacion'] = f'Asignado (Territorial - {nivel})'
                asignados_list.append(vacante_dict)
                id_vacantes_asignadas.add(vacante['ID_VACANTE_UNICA'])

    # Completar cuota total si es necesario
    cuota_faltante = total_reserva - len(id_vacantes_asignadas)
    if cuota_faltante > 0:
        df_restantes = df_merged[~df_merged['ID_VACANTE_UNICA'].isin(id_vacantes_asignadas)]
        vacantes_complementarias = df_restantes.head(cuota_faltante)

        for _, vacante in vacantes_complementarias.iterrows():
            vacante_dict = vacante.to_dict()
            vacante_dict['Estado_Asignacion'] = 'Asignado (Territorial - Complementario)'
            asignados_list.append(vacante_dict)

    return pd.DataFrame(asignados_list) if asignados_list else pd.DataFrame()

print("✅ Lógica de asignación territorial configurada")

def asignar_vacantes_territoriales(
    df_vacancies: pd.DataFrame,
    ai_recs: List[Dict],
    df_minsalud: pd.DataFrame,
    cuotas_por_nivel_y_modalidad: pd.DataFrame,
    total_reserva: int,
    municipio: str
) -> pd.DataFrame:
    """Asigna vacantes considerando contexto territorial completo"""

    if not ai_recs or total_reserva == 0:
        df_fallback = df_vacancies.head(min(total_reserva, len(df_vacancies))).copy()
        if not df_fallback.empty:
            df_fallback['Estado_Asignacion'] = 'Asignado (Sin Análisis Territorial)'
            df_fallback['Score_Final'] = 0.5
        return df_fallback

    # Preparar DataFrame de recomendaciones
    df_recs = pd.DataFrame(ai_recs)
    if df_recs.empty:
        return pd.DataFrame()

    if 'opec' in df_recs.columns:
        df_recs.rename(columns={'opec': COL_OPEC}, inplace=True)

    # Calcular scores territoriales
    def calcular_scores_completos(row):
        analisis = row.get('analisis_detallado', [])
        nivel_edu = row.get(COL_NIVEL_EDUCATIVO, '')

        # Score IA territorial
        score_ia = calcular_score_compatibilidad_territorial(analisis) / 14.0

        # Obtener tipos de discapacidad con afinidad Alta/Media
        tipos_compatibles = [
            d.get('discapacidad', '') for d in analisis
            if d.get('afinidad') in ['Alta', 'Media']
        ]

        # Calcular scoring territorial completo
        scores = calcular_score_territorial_completo(
            score_ia, nivel_edu, municipio, df_minsalud, tipos_compatibles
        )

        return pd.Series({
            'score_final': scores['score_final'],
            'score_ia': scores['score_ia'],
            'score_educativo': scores['score_educativo'],
            'score_predominancia': scores['score_predominancia'],
            'tipos_compatibles': tipos_compatibles
        })

    # Aplicar cálculos de scoring
    scores_df = df_recs.apply(calcular_scores_completos, axis=1)
    df_recs = pd.concat([df_recs, scores_df], axis=1)

    # Preparar vacancies con información completa
    df_vacancies[COL_OPEC] = df_vacancies[COL_OPEC].astype(str)
    df_recs[COL_OPEC] = df_recs[COL_OPEC].astype(str)

    # Merge con datos territoriales
    df_merged = pd.merge(df_vacancies, df_recs, on=COL_OPEC, how='left')
    df_merged['score_volumen'] = np.log1p(df_merged[COL_VACANTES])

    # Rellenar valores faltantes
    df_merged.fillna({
        'score_final': 0.0,
        'score_ia': 0.0,
        'score_educativo': 0.0,
        'score_predominancia': 0.0,
        'score_volumen': 0.0,
        'tipos_compatibles': []
    }, inplace=True)

    # Prioridad especial para provisionalidad
    df_merged['prioridad_provision'] = 0.0
    if COL_ESTADO_PROVISION in df_merged.columns:
        # Máxima prioridad para empleos con PcD provisional
        mask_provisional = df_merged[COL_ESTADO_PROVISION].str.contains('PROVISIONAL', na=False)
        df_merged.loc[mask_provisional, 'prioridad_provision'] = 1.0

        # Segunda prioridad para empleos no provistos
        mask_no_provisto = df_merged[COL_ESTADO_PROVISION].str.contains('NO_PROVISTO|SIN_PROVEER', na=False)
        df_merged.loc[mask_no_provisto, 'prioridad_provision'] = 0.5

    # Score final ponderado con provisionalidad
    df_merged['score_final_ponderado'] = (
        df_merged['score_final'] * 0.7 +  # Score territorial
        df_merged['prioridad_provision'] * 0.3  # Prioridad provisionalidad
    )

    # Ordenar por prioridad territorial completa
    df_merged.sort_values(
        by=['prioridad_provision', 'score_final_ponderado', 'score_volumen', 'ID_VACANTE_UNICA'],
        ascending=[False, False, False, True],
        inplace=True
    )

    return ejecutar_asignacion_estratificada(df_merged, cuotas_por_nivel_y_modalidad, total_reserva)


def ejecutar_asignacion_territorial_completa(
    df_vacancies: pd.DataFrame,
    ai_recs: List[Dict],
    df_minsalud: pd.DataFrame,
    cuotas_por_nivel_y_modalidad: pd.DataFrame,
    total_reserva: int,
    municipio: str
) -> pd.DataFrame:
    """Ejecuta la asignación territorial completa con todos los factores"""

    # Preparar recomendaciones de IA
    df_recs = pd.DataFrame(ai_recs)
    if 'opec' in df_recs.columns:
        df_recs.rename(columns={'opec': COL_OPEC}, inplace=True)

    # Calcular scores para cada empleo
    def calcular_scores_empleo(row):
        analisis = row.get('analisis_detallado', [])

        # Score IA (simplificado)
        score_ia = 0.5  # Default
        if analisis:
            score_ia = sum([1 if d.get('afinidad') == 'Alta' else 0.5 if d.get('afinidad') == 'Media' else 0
                           for d in analisis]) / len(analisis)

        # Score educativo y territorial
        nivel_edu = row.get(COL_NIVEL_EDUCATIVO, '')
        score_educativo = calcular_score_educativo(nivel_edu, df_minsalud, municipio)

        # Tipos compatibles
        tipos_compatibles = [d.get('discapacidad', '') for d in analisis
                           if d.get('afinidad') in ['Alta', 'Media']]

        # Score predominancia
        predominancia = calcular_predominancia_territorial(df_minsalud, municipio)
        score_predominancia = 0.0
        if predominancia and tipos_compatibles:
            for tipo in tipos_compatibles:
                score_predominancia += predominancia.get(tipo, 0) / 100
            score_predominancia = min(1.0, score_predominancia)

        # Score final compuesto
        score_final = (score_ia * 0.7) + (score_educativo * 0.2) + (score_predominancia * 0.1)

        return pd.Series({
            'score_final': score_final,
            'score_ia': score_ia,
            'score_educativo': score_educativo,
            'score_predominancia': score_predominancia,
            'tipos_compatibles': tipos_compatibles
        })

    # Aplicar cálculos
    scores_df = df_recs.apply(calcular_scores_empleo, axis=1)
    df_recs = pd.concat([df_recs, scores_df], axis=1)

    # Merge con vacancies
    df_vacancies[COL_OPEC] = df_vacancies[COL_OPEC].astype(str)
    df_recs[COL_OPEC] = df_recs[COL_OPEC].astype(str)

    df_merged = pd.merge(df_vacancies, df_recs, on=COL_OPEC, how='left')

    # Rellenar valores faltantes
    df_merged.fillna({
        'score_final': 0.3,
        'score_ia': 0.3,
        'score_educativo': 0.3,
        'score_predominancia': 0.0,
        'tipos_compatibles': []
    }, inplace=True)

    # Ordenar por score y tomar los mejores
    df_merged.sort_values(['score_final', COL_VACANTES], ascending=[False, False], inplace=True)

    # Asignar las mejores vacantes
    df_asignacion = df_merged.head(total_reserva).copy()
    df_asignacion['Estado_Asignacion'] = 'Asignado (Análisis Territorial)'

    return df_asignacion

print("✅ Función de asignación territorial corregida")

# ========= CONFIG PIPELINE =========
@dataclass
class PipelineConfig:
    proveedor_ia: str = "ninguno"          # 'openai' | 'gemini' | 'ninguno'
    openai_api_key: str = ""
    google_api_key: str = ""
    top_n: int = 10
    score_minimo: int = 60
    generar_pdf: bool = True

def _setup_ia(config: PipelineConfig) -> None:
    if config.proveedor_ia == "openai":
        try:
            import openai
            openai.api_key = config.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
            if not openai.api_key:
                raise ValueError("Falta OPENAI_API_KEY")
        except Exception as e:
            raise RuntimeError(f"No se pudo inicializar OpenAI: {e}")
    elif config.proveedor_ia == "gemini":
        try:
            import google.generativeai as genai
            api_key = config.google_api_key or os.environ.get("GOOGLE_API_KEY", "")
            if not api_key:
                raise ValueError("Falta GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"No se pudo inicializar Gemini: {e}")

# ========= PIPELINE PRINCIPAL =========
def run_pipeline(df1: pd.DataFrame, df2: pd.DataFrame, config: PipelineConfig) -> Dict[str, Any]:
    """
    df1: Datos Minsalud (territorio/población PcD) – esperado que tenga MUNICIPIO y columnas de conteo
    df2: OPEC (vacantes) – esperado que tenga PROPOSITO, FUNCIONES, MODALIDAD, NIVEL, DENOMINACION, CANTIDAD DE VACANTES
    """
    # Normalizar encabezados
    df1 = df1.copy(); df1.columns = [str(c).upper().strip() for c in df1.columns]
    df2 = df2.copy(); df2.columns = [str(c).upper().strip() for c in df2.columns]

    # 1) Validaciones/límites suaves: intentamos usar tus funciones si existen
    errores: List[str] = []
    df_opec_val = df2
    try:
        if 'limpiar_y_validar_dataframe' in globals():
            df_opec_val, _errs = limpiar_y_validar_dataframe(df2)  # valida OPEC
            errores += _errs or []
        else:
            # fallback: columnas mínimas + limpieza básica
            req = [COL_PROPOSITO, COL_FUNCIONES, COL_VACANTES]
            faltan = [c for c in req if c not in df2.columns]
            if faltan:
                errores.append(f"Faltan columnas en OPEC: {faltan}")
            df_opec_val[COL_FUNCIONES_LIMPIAS] = df_opec_val.get(COL_FUNCIONES, "").apply(limpiar_texto_funciones)
            df_opec_val[COL_VACANTES] = pd.to_numeric(df_opec_val.get(COL_VACANTES, 0), errors='coerce').fillna(0).astype(int)
            df_opec_val = df_opec_val[df_opec_val[COL_VACANTES] > 0].copy()
    except Exception as e:
        errores.append(f"Error validando OPEC: {e}")
        df_opec_val = df2.copy()

    try:
        if 'validar_estructura_minsalud' in globals():
            _ok, _msg = validar_estructura_minsalud(df1)
            if not _ok:
                errores.append(f"Minsalud: {_msg}")
    except Exception as e:
        errores.append(f"Error validando Minsalud: {e}")

    # 2) Recomendaciones IA (opcional)
    ai_recs: List[Dict] = []
    if config.proveedor_ia.lower() in ("openai", "gemini") and 'analizar_empleos_con_ia_territorial' in globals():
        try:
            _setup_ia(config)
            ai_recs = analizar_empleos_con_ia_territorial(
                df_vacantes=df_opec_val.head(config.top_n * 2),  # limitamos por demo
                df_minsalud=df1,
                provider=config.proveedor_ia.lower(),
                model_name="gpt-4o-mini" if config.proveedor_ia.lower()=="openai" else "gemini-1.5-pro"
            ) or []
        except Exception as e:
            errores.append(f"IA desactivada por error: {e}")

    # 3) Scoring básico: intentamos usar tus funciones, con fallback
    df_res = df_opec_val.copy()
    def _safe_score(row):
        try:
            if 'calcular_score_compatibilidad_territorial' in globals():
                return calcular_score_compatibilidad_territorial(
                    row.to_dict(), df_minsalud=df1, municipio=row.get(COL_MUNICIPIO, None)
                )
        except Exception:
            pass
        # Fallback simple
        base = 70
        if isinstance(row.get(COL_MODALIDAD, ""), str) and "ABIERTA" in row[COL_MODALIDAD].upper():
            base += 5
        if isinstance(row.get(COL_NIVEL, ""), str) and "PROFESIONAL" in row[COL_NIVEL].upper():
            base += 5
        return min(100, base)

    df_res["SCORE_COMPATIBILIDAD"] = df_res.apply(_safe_score, axis=1)
    df_res = df_res[df_res["SCORE_COMPATIBILIDAD"] >= int(config.score_minimo)].copy()
    df_res = df_res.sort_values("SCORE_COMPATIBILIDAD", ascending=False).head(int(config.top_n))

    # 4) Visualización sencilla
    x_col = COL_MUNICIPIO if COL_MUNICIPIO in df_res.columns else (COL_NIVEL if COL_NIVEL in df_res.columns else df_res.columns[0])
    fig = px.histogram(df_res, x=x_col, title="Distribución de vacantes seleccionadas")

    # 5) PDF básico
    from .report import build_simple_pdf
    resumen = f"Filas: {len(df_res)} | IA_recs: {len(ai_recs)} | Errores: {'; '.join(errores) if errores else 'Ninguno'}"
    pdf_bytes = build_simple_pdf(title="Reporte Analizador PcD", resumen=resumen)

    return {
        "df_resultado": df_res,
        "fig": fig,
        "pdf_bytes": pdf_bytes if config.generar_pdf else None,
        "errores": errores,
        "ai_recs": ai_recs,
    }
