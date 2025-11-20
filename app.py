import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt  # por si acaso
import streamlit as st
import altair as alt

# =========================================================
#  TEMA CLARO PARA ALTAIR
# =========================================================
def _light_theme():
    return {
        "config": {
            "background": "white",
            "view": {"stroke": "transparent"},
            "axis": {
                "labelColor": "#333333",
                "titleColor": "#333333",
            },
            "legend": {
                "labelColor": "#333333",
                "titleColor": "#333333",
            },
        }
    }

alt.themes.register("light_theme", _light_theme)
alt.themes.enable("light_theme")

# =========================================================
#  CONFIG & CONSTANTES
# =========================================================

PROGRAM_PATTERNS = ["programa"]
YEAR_PATTERNS = [
    "añoegreso",
    "año egreso",
    "anoegreso",
    "ano egreso",
    "año de egreso",
    "ano de egreso",
]


def load_config(path: str = "config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm(s: str) -> str:
    """Normaliza texto: minúsculas, sin acentos, sin dobles espacios."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).replace("\u00A0", " ").strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_beneficios(texto: str) -> str:
    """
    Agrupa respuestas libres de:
    '¿Qué beneficios te gustaría obtener de una asociación de egresados?'
    en categorías más generales.
    """
    t = norm(texto)
    if not t:
        return ""

    # --- NINGUNO / NO CONOCE / NO APLICA ---
    if (
        "no aplica" in t
        or "ninguno" in t
        or "ningun " in t
        or "ninguno por el momento" in t
        or "no tengo conocimiento" in t
        or "no conozco los beneficios" in t
        or "no conozco los beneficios actuales" in t
    ):
        return "Ninguno / no conoce beneficios"

    # --- DESCUENTOS / BENEFICIOS ECONÓMICOS ---
    if "descuento" in t or "caminatas ecologicas" in t or "caminatas ecológicas" in texto.lower():
        return "Descuentos y beneficios económicos"

    # --- VINCULACIÓN LABORAL / EMPLEABILIDAD / BOLSA DE EMPLEO ---
    if (
        "empleabilidad" in t
        or "vinculacion laboral" in t
        or "vinculación laboral" in texto.lower()
        or "bolsa de empleo" in t
        or "vinculacion a la bolsa de empleo" in t
    ):
        return "Vinculación laboral / empleabilidad"

    # --- INFORMACIÓN / COMUNICACIÓN ---
    if "informacion" in t or "información" in texto.lower():
        return "Información y comunicación con egresados"

    # --- ACTIVIDADES / DEPORTIVAS / CULTURALES / RECREATIVAS ---
    if (
        "actividades deportivas" in t
        or "actividades culturales" in t
        or "recreativas" in t
    ):
        return "Actividades deportivas, culturales y recreativas"

    # --- CUANDO PONEN "TODOS" ---
    if t == "todos":
        return "Todos los beneficios"

    return texto.strip()

# =========================================================
#  CARGA DE DATOS + DETECCIÓN DE ENCABEZADOS
# =========================================================

@st.cache_data
def load_data_with_header_detection(csv_source) -> pd.DataFrame:
    """
    csv_source puede ser:
      - ruta (str / Path) a un .csv
      - archivo subido (UploadedFile) de Streamlit

    No asume que la primera fila sea encabezado; busca fila con 'programa' y 'año egreso'.
    """
    df_raw = pd.read_csv(
        csv_source,
        header=None,
        dtype=str,
        encoding="utf-8",
        keep_default_na=False,
    )

    n_rows, _ = df_raw.shape
    header_row_idx = None
    max_scan = min(n_rows, 50)

    for i in range(max_scan):
        row_norm = [norm(v) for v in df_raw.iloc[i].tolist()]
        has_prog = any(any(p in c for p in PROGRAM_PATTERNS) for c in row_norm)
        has_year = any(any(p in c for p in YEAR_PATTERNS) for c in row_norm)
        if has_prog and has_year:
            header_row_idx = i
            break

    if header_row_idx is None:
        # fallback: asumimos primera fila como encabezado
        df = pd.read_csv(csv_source, encoding="utf-8")
        df.columns = [c.replace("\u00A0", " ").strip() for c in df.columns]
        return df

    headers = df_raw.iloc[header_row_idx].tolist()
    df = df_raw.iloc[header_row_idx + 1:].reset_index(drop=True)
    df.columns = [str(h).replace("\u00A0", " ").strip() for h in headers]
    return df


def detect_program_year_columns(df: pd.DataFrame):
    """Busca en nombres de columna algo con 'programa' y algo con 'año egreso'."""
    prog_col = None
    year_col = None

    for c in df.columns:
        n = norm(c)
        if prog_col is None and any(p in n for p in PROGRAM_PATTERNS):
            prog_col = c
        if year_col is None and any(p in n for p in YEAR_PATTERNS):
            year_col = c

    if prog_col is None or year_col is None:
        raise KeyError(
            "No pude encontrar columnas de Programa/AñoEgreso.\n"
            f"Encabezados encontrados: {list(df.columns)}"
        )

    return prog_col, year_col

# =========================================================
#  FILTROS (SIDEBAR)
# =========================================================

def get_program_year_filters(df: pd.DataFrame):
    prog_col, year_col = detect_program_year_columns(df)

    # ==============================
    # LIMPIAR COLUMNA PROGRAMA
    # ==============================
    prog_series = df[prog_col].astype(str).str.strip()

    # Valores que queremos tratar como "vacíos"
    prog_series = prog_series.replace(
        ["", "nan", "NaN", "NULL", "null"], pd.NA
    )

    # Lista de programas válidos (sin null, sin vacíos)
    programas = (
        prog_series
        .dropna()
        .unique()
        .tolist()
    )
    programas = sorted(programas)

    # ==============================
    # AÑOS (igual que antes)
    # ==============================
    years_raw = df[year_col].dropna().astype(str).str.strip()
    years = sorted(
        {
            m.group(0)
            for val in years_raw
            if (m := re.search(r"(19\d{2}|20\d{2}|2100)", val))
        },
        key=int,
    )

    st.sidebar.header("Filtros")

    # ---- PROGRAMA: ahora con "Todos los programas" ----
    opciones_prog = ["Todos los programas"] + programas
    programa_sel = st.sidebar.selectbox(
        "Programa",
        opciones_prog,
        key="sidebar_programa",
    )

    # ---- AÑO ----
    year_sel = st.sidebar.selectbox(
        "AñoEgreso (opcional)",
        ["Todos"] + years,
        key="sidebar_year",
    )

    # ==============================
    # CONSTRUIR EL MASK DE FILTRO
    # ==============================
    # Empezamos con todo True y vamos filtrando
    mask = pd.Series(True, index=df.index)

    # Si NO es "Todos los programas", filtramos por programa
    if programa_sel != "Todos los programas":
        mask = mask & prog_series.eq(programa_sel)

    # Si NO es "Todos", filtramos por año
    if year_sel != "Todos":
        mask = mask & years_raw.str.contains(year_sel)

    df_filtrado = df[mask].copy()
    return df_filtrado, programa_sel, year_sel

# =========================================================
#  UTIL MULTI-RESPUESTA
# =========================================================

def split_multi_values(series: pd.Series) -> pd.Series:
    """Para preguntas MULTI (texto con ; , /): separa y apila."""
    all_vals = []
    for val in series.dropna():
        for part in re.split(r"[;,/]", str(val)):
            part = part.strip()
            if part:
                all_vals.append(part)
    return pd.Series(all_vals)

# =========================================================
#  GRÁFICAS
# =========================================================

def chart_binary(df: pd.DataFrame, col: str, label: str):
    serie = df[col].dropna().astype(str).str.strip()
    if serie.empty:
        st.info("Sin datos para esta pregunta.")
        return

    yes_pattern = re.compile(r"^(si|sí|true|1|x)$", re.IGNORECASE)
    no_pattern = re.compile(r"^(no|false|0)$", re.IGNORECASE)

    mapped = []
    for v in serie:
        if yes_pattern.match(v):
            mapped.append("Sí")
        elif no_pattern.match(v):
            mapped.append("No")
        else:
            mapped.append(v)

    counts = pd.Series(mapped).value_counts().reset_index()
    counts.columns = ["Respuesta", "Conteo"]

    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("Respuesta:N", sort="-y", title="Respuesta"),
            y=alt.Y("Conteo:Q"),
            color=alt.Color("Respuesta:N", legend=None),
            tooltip=["Respuesta", "Conteo"],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def chart_categorical(df: pd.DataFrame, col: str, label: str, top_k: int):
    serie = df[col].dropna().astype(str).str.strip()

    # Normalización especial para beneficios de asociación
    if col.strip() == "¿Qué beneficios te gustaría obtener de una asociación de egresados?":
        serie = serie.apply(normalize_beneficios)

    if serie.empty:
        st.info("Sin datos para esta pregunta.")
        return

    counts = serie.value_counts().head(top_k).reset_index()
    counts.columns = ["Categoría", "Conteo"]

    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("Conteo:Q"),
            y=alt.Y("Categoría:N", sort="-x"),
            color=alt.Color("Categoría:N", legend=None),
            tooltip=["Categoría", "Conteo"],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def chart_multi(df: pd.DataFrame, col: str, label: str, top_k: int):
    serie = split_multi_values(df[col])
    if serie.empty:
        st.info("Sin datos para esta pregunta.")
        return
    counts = serie.value_counts().head(top_k).reset_index()
    counts.columns = ["Selección", "Conteo"]

    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("Conteo:Q"),
            y=alt.Y("Selección:N", sort="-x"),
            color=alt.Color("Selección:N", legend=None),
            tooltip=["Selección", "Conteo"],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def chart_likert(df: pd.DataFrame, col: str, label: str):
    serie = df[col].dropna()
    if serie.empty:
        st.info("Sin datos para esta pregunta.")
        return

    numeric = []
    for v in serie:
        try:
            n = float(str(v).replace(",", "."))
            numeric.append(n)
        except ValueError:
            continue

    if not numeric:
        st.info("No se pudo interpretar la escala 1–5 en esta pregunta.")
        return

    s = pd.Series(numeric)
    dist = s.round().astype(int).value_counts().sort_index()
    df_dist = dist.reindex([1, 2, 3, 4, 5], fill_value=0).reset_index()
    df_dist.columns = ["Valor", "Conteo"]

    chart = (
        alt.Chart(df_dist)
        .mark_bar()
        .encode(
            x=alt.X("Valor:N"),
            y=alt.Y("Conteo:Q"),
            color=alt.Color("Valor:N", legend=None),
            tooltip=["Valor", "Conteo"],
        )
    )
    st.altair_chart(chart, use_container_width=True)

    avg = s.mean()
    st.caption(f"Promedio: {avg:.2f}")


def chart_multi_from_cols(df: pd.DataFrame, option_cols, label: str, top_k: int):
    """
    Para preguntas donde la pregunta está en una columna vacía
    y las opciones están en varias columnas con 0/1.
    """
    if not option_cols:
        st.info("Sin columnas de opción configuradas para esta pregunta.")
        return

    data = []

    for c in option_cols:
        if c not in df.columns:
            continue

        col_obj = df[c]

        if isinstance(col_obj, pd.DataFrame):
            col_series = col_obj.iloc[:, 0]
        else:
            col_series = col_obj

        series = pd.to_numeric(col_series, errors="coerce").fillna(0)
        count = (series != 0).sum()
        data.append((c, int(count)))

    if not data:
        st.info("Sin datos seleccionados para esta pregunta.")
        return

    df_counts = pd.DataFrame(data, columns=["Selección", "Conteo"])
    df_counts = df_counts.sort_values("Conteo", ascending=False).head(top_k)

    chart = (
        alt.Chart(df_counts)
        .mark_bar()
        .encode(
            x=alt.X("Conteo:Q"),
            y=alt.Y("Selección:N", sort="-x"),
            color=alt.Color("Selección:N", legend=None),
            tooltip=["Selección", "Conteo"],
        )
    )
    st.altair_chart(chart, use_container_width=True)

# =========================================================
#  APP PRINCIPAL
# =========================================================

def main():
    st.set_page_config(
        page_title="Informe Egresados UTP",
        layout="wide",
        page_icon="📊",
    )

    # ---- Estado de visibilidad del sidebar ----
    if "show_filters" not in st.session_state:
        st.session_state["show_filters"] = True

    show_filters = st.session_state["show_filters"]

    sidebar_display_css = "" if show_filters else """
        [data-testid="stSidebar"]{
            display: none !important;
        }
    """

    # ===== CSS GENERAL =====
    st.markdown(
        f"""
        <style>  
        [data-testid="stAppViewContainer"]{{
            background-color: #ffffff;
        }}
        [data-testid="stAppViewContainer"] *{{
            color: #111111;
            font-family: "Segoe UI", system-ui, sans-serif;
        }}
        .block-container {{
            padding-top: 3rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }}
        [data-testid="stHeader"] {{
            background-color: #ffffff;
        }}
        [data-testid="stHeader"] div {{
            height: 0px;
        }}
        [data-testid="stSidebar"]{{
            background-color: #2f3542;
        }}
        [data-testid="stSidebar"] *{{
            color: #f5f6fa !important;
            font-family: "Segoe UI", system-ui, sans-serif;
        }}
        {sidebar_display_css}
        .portada-box{{
            border-radius: 16px;
            padding: 1.5rem 2rem;
            background: linear-gradient(120deg, #00b894, #0984e3);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.25);
        }}
        .portada-box h1{{
            margin-bottom: 0.3rem;
            font-size: 2rem;
        }}
        .portada-box h3{{
            margin-top: 0.2rem;
            margin-bottom: 0.8rem;
            font-weight: 500;
            font-size: 1.1rem;
        }}
        .portada-meta{{
            font-size: 0.95rem;
            opacity: 0.95;
        }}
        h2, h3 {{
            color: #111111 !important;
        }}
        .logo-row{{
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin: 1.2rem 0 0.4rem 0;
        }}
        .logo-row img{{
            max-height: 70px;
            width: auto;
            height: auto;
            object-fit: contain;
        }}
        .stAlert {{
            border-radius: 10px;
        }}
        .print-hide {{
        /* elementos que no deben salir en el PDF */
        }}
        /* No separar títulos de lo que viene después */
h2, h3 {{
    page-break-after: avoid;
    break-after: avoid;
}}

/* Evitar que las gráficas se partan entre páginas */
.element-container,
.stAltairChart,
.stPlotlyChart {{
    page-break-inside: avoid !important;
    break-inside: avoid !important;
}}

        @page {{
            size: A4;
            margin: 8mm;
        }}
        @media print {{
        /* Ocultar sidebar, header y cosas de Streamlit */
        [data-testid="stSidebar"],
        [data-testid="stHeader"],
        .print-hide {{
            display: none !important;
        }}

        /* Quitar márgenes extra del contenido */
        .block-container {{
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            max-width: 100% !important;
        }}

        /* Intentar que cada gráfico no se parta entre páginas */
        .element-container,
        .stAltairChart,
        .stPlotlyChart {{
            break-inside: avoid !important;
            page-break-inside: avoid !important;
        }}

        
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- Botón hamburguesa ----
    def toggle_filters():
        st.session_state["show_filters"] = not st.session_state["show_filters"]

    top_cols = st.columns([0.25, 0.75])
    with top_cols[0]:
        st.button(
            "☰ Filtros",
            on_click=toggle_filters,
            help="Mostrar/ocultar panel de filtros",
            key="btn_toggle_filtros",
        )

    # ===== CONFIG (solo preguntas) =====
    cfg = load_config()
    questions = cfg["questions"]

    # ===== SUBIR ARCHIVO (OBLIGATORIO) =====
    st.sidebar.subheader("Archivo de datos")
    uploaded = st.sidebar.file_uploader(
        "Sube el archivo CSV de la encuesta",
        type=["csv"],
        key="uploader_csv",
    )

    # 👉 Si NO hay archivo subido, no mostramos nada del informe
    if uploaded is None:
        st.markdown("## Informe de Encuesta de Egresados")
        st.info("Por favor sube el archivo CSV de la encuesta en la barra lateral para visualizar el informe.")
        return

    # Si sí hay archivo, lo cargamos
    df = load_data_with_header_detection(uploaded)

    # ===== APLICAR FILTROS POR PROGRAMA / AÑO =====
    df_filt, programa, year = get_program_year_filters(df)
    total = len(df_filt)

    # ===== LOGOS =====
    st.markdown(
        """
    <div class="logo-row">
      <img src="https://media2.utp.edu.co/programas/8/utp.png" alt="Logo UTP">
      <img src="https://comunicaciones.utp.edu.co/wp-content/uploads/sites/2/LOGO-COLOR-05.png" alt="Logo ASEUTP">
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ===== PORTADA =====
    st.markdown(
        f"""
    <div class="portada-box">
      <h1>Informe de Encuesta de Egresados Pregrado</h1>
      <h3>Universidad Tecnológica de Pereira · ASEUTP</h3>
      <div class="portada-meta">
        <p><strong>Programa:</strong> {programa}</p>
        <p><strong>Año de egreso:</strong> {year if year != 'Todos' else 'Todos los años'}</p>
        <p><strong>Respuestas analizadas:</strong> {total}</p>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if total == 0:
        st.warning("No hay respuestas para ese filtro. Cambia Programa/AñoEgreso.")
        return

    # ===== CONTENIDO (PARTES A–E) =====
    sections = ["A", "B", "C", "D", "E"]
    # Títulos visibles por sección
    section_titles = {
        "A": "Situación Laboral",
        "B": "Emprendimiento",
        "C": "Satisfacción con los recursos ofrecidos por la Universidad",
        "D": "Autoevaluación",
        "E": "Acreditación Institucional ",
    }

    for sec in sections:
        preguntas_sec = [q for q in questions if q.get("section", "").upper() == sec]
        if not preguntas_sec:
            continue

        titulo = section_titles.get(sec, f"Parte {sec}")
        st.markdown(f"## {titulo}")

        for q in preguntas_sec:
            qtype = q["type"].upper()
            label = q.get("label", "")
            col = q.get("column")

            if qtype != "MULTI_COLS":
                if not col or col not in df_filt.columns:
                    st.warning(f"La columna no existe en los datos: {col}")
                    continue

            st.markdown(f"### {label}")

            if qtype == "BINARIA":
                chart_binary(df_filt, col, label)
            elif qtype == "CATEGORICA":
                chart_categorical(
                    df_filt, col, label, cfg.get("top_k_categories", 10)
                )
            elif qtype == "MULTI":
                chart_multi(df_filt, col, label, cfg.get("top_k_categories", 10))
            elif qtype == "LIKERT":
                chart_likert(df_filt, col, label)
            elif qtype == "MULTI_COLS":
                chart_multi_from_cols(
                    df_filt,
                    q.get("columns", []),
                    label,
                    cfg.get("top_k_categories", 10),
                )
            else:
                st.info(f"Tipo no soportado: {qtype}")

    # ===== CONCLUSIONES DEL INFORME =====
    st.markdown("## Conclusiones")

    conclusiones = st.text_area(
        "",
        height=200,
        key="conclusiones_texto",
        help="Este texto se guardará mientras la página esté abierta y saldrá al imprimir/guardar en PDF.",
    )

    # Vista previa en formato “bonito” para el PDF
    if conclusiones.strip():
        st.markdown("### Conclusiones")
        # Reemplazamos saltos de línea por saltos de línea en Markdown
        st.markdown(conclusiones.replace("\n", "  \n"))



if __name__ == "__main__":
    main()
