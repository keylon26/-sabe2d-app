# --- SABE 2D | Software de Análisis de Bidimensional de Estructuras
# --- Desarrollado por Keylon D'costa & Dilan Nemocon 
# --- Universidad de La Guajira | Facultad de Ingeniería Civil
# --------------------------------------------------------------------------------------

# --- LIBRERÍAS ---
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.linalg import eigh
import streamlit as st

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="SABE 2D - Universidad de La Guajira", layout="wide")

# --- ESTILOS VISUALES (CSS) ---
st.markdown(
    """
    <style>
    .stApp { background-color: #005B64; }
    h1 { font-size: 35px !important; }
    h3 { font-size: 22px !important; margin-top: 10px !important; }
    [data-testid="stSidebar"] h1 { font-size: 18px !important; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size: 18px !important; color: #FFFFFF; }
    [data-testid="stSidebar"] div[data-testid="stSelectbox"] label { font-size: 12px !important; color: #FFFFFF !important; font-weight: bold !important; }
    [data-testid="stSidebar"] div[data-testid="stToggle"] label { font-size: 12px !important; color: #E0E0E0 !important; }            
    [data-testid="stSidebar"] { background-color: #003238; }
    [data-testid="stHeader"] { background-color: #002529; }
    [data-testid="stHeader"]::before {
        content: "SABE 2D | Software de Análisis de Bidimensional de Estructuras | Versión 1.1";
        color: #00FFFF; position: absolute; left: 50%; transform: translateX(-50%);
        top: 16px; font-size: 18px; font-weight: bold; letter-spacing: 1px; white-space: nowrap;
    }
    .plotly-notifier, .annotation-text { backdrop-filter: blur(4px); -webkit-backdrop-filter: blur(4px); }

div[data-testid="stFormSubmitButton"] > button {
    background-color: #002529 !important;
    color: #00FFFF !important;
    border: 2px solid #00FFFF !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    letter-spacing: 1px !important;
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.4), inset 0 0 10px rgba(0, 255, 255, 0.2) !important;
    transition: all 0.3s ease-in-out !important;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #00FFFF !important;
    color: #002529 !important;
    border-color: #FFFFFF !important;
    box-shadow: 0 0 20px #00FFFF, 0 0 35px #00FFFF, inset 0 0 15px #00FFFF !important;
    transform: translateY(-2px);
}
    </style>
""",
    unsafe_allow_html=True,
)

st.title("📐 Análisis Bidimensional de Estructuras")
st.markdown("**Universidad de La Guajira** | Facultad de Ingeniería Civil")

# --- SISTEMA MAESTRO DE CONVERSIÓN DE UNIDADES ---
FACTORES_BASE = {
    "SI (m, kN)": {"L": 1.0, "F": 1.0},
    "MKS (m, tonf)": {"L": 1.0, "F": 0.101972},
    "Inglés (ft, kip)": {"L": 3.28084, "F": 0.224809},
}

# --- GESTIÓN DE MEMORIA Y ESTADOS (DATOS INICIALES POR DEFECTO) ---
if "last_system" not in st.session_state:
    st.session_state.last_system = "SI (m, kN)"

if "gamma_val" not in st.session_state:
    st.session_state.gamma_val = 24.5200

if "df_nudos" not in st.session_state:
    st.session_state.df_nudos = pd.DataFrame({
        "Nudo": [1, 2, 3, 4],
        "X": [0.00, 0.00, 4.00, 4.00],
        "Y": [0.00, 3.00, 3.00, 0.00]
    })

if "df_barras" not in st.session_state:
    st.session_state.df_barras = pd.DataFrame({
        "Barra": [1, 2, 3],
        "Nudo_Ini": [1, 2, 3],
        "Nudo_Fin": [2, 3, 4],
        "Area": [0.1600, 0.1600, 0.1600],
        "Inercia": [0.002133, 0.002133, 0.002133],
        "E": [22000000.0, 22000000.0, 22000000.0]
    })

if "df_apoyos_v4" not in st.session_state:
    st.session_state.df_apoyos_v4 = pd.DataFrame({
        "Nudo": [1, 4],
        "Dx": [1, 1],
        "Dy": [1, 1],
        "Giro": [1, 1],
        "Orientación": ["Centro", "Centro"]
    })

if "df_puntuales" not in st.session_state:
    st.session_state.df_puntuales = pd.DataFrame({
        "Nudo": [2],
        "Fx": [10.00],
        "Fy": [0.00],
        "M": [0.00],
        "Tipo": ["D"]
    })

if "df_distribuidas" not in st.session_state:
    st.session_state.df_distribuidas = pd.DataFrame({
        "Barra": pd.Series(dtype="int"),
        "q1": pd.Series(dtype="float"),
        "q2": pd.Series(dtype="float"),
        "Tipo": pd.Series(dtype="str")
    })

def ejecutar_conversion_unidades():
    sistema_viejo = st.session_state.last_system
    sistema_nuevo = st.session_state.current_system
    if sistema_viejo == sistema_nuevo:
        return
    
    fL = FACTORES_BASE[sistema_nuevo]["L"] / FACTORES_BASE[sistema_viejo]["L"]
    fF = FACTORES_BASE[sistema_nuevo]["F"] / FACTORES_BASE[sistema_viejo]["F"]
    
    fA, fI, fM, fq, fE = fL**2, fL**4, fF * fL, fF / fL, fF / (fL**2)
    fGamma = fF / (fL**3)
    st.session_state.gamma_val = float(st.session_state.gamma_val * fGamma)

    if not st.session_state.df_nudos.empty:
        df = st.session_state.df_nudos.copy()
        df[["X", "Y"]] *= fL
        st.session_state.df_nudos = df

    if not st.session_state.df_barras.empty:
        df = st.session_state.df_barras.copy()
        df["Area"] *= fA
        df["Inercia"] *= fI
        df["E"] *= fE
        st.session_state.df_barras = df

    if not st.session_state.df_puntuales.empty:
        df = st.session_state.df_puntuales.copy()
        df[["Fx", "Fy"]] *= fF
        df["M"] *= fM
        st.session_state.df_puntuales = df

    if not st.session_state.df_distribuidas.empty:
        df = st.session_state.df_distribuidas.copy()
        df[["q1", "q2"]] *= fq
        st.session_state.df_distribuidas = df

    st.session_state.last_system = sistema_nuevo

# --- BARRA LATERAL DE CONTROLES ---
col_sidebar_img1, col_sidebar_img2, col_sidebar_img3 = st.sidebar.columns([1, 1, 1])
with col_sidebar_img2:
    try:
        st.image("UNIGUAJIRA.jpg", width=80)
    except:
        pass

with st.sidebar.expander("📏 Sistema de unidades", expanded=False):
    sistema_unidades = st.selectbox("Seleccionar unidades:", ["SI (m, kN)", "MKS (m, tonf)", "Inglés (ft, kip)"], key="current_system", on_change=ejecutar_conversion_unidades)

dicc_u = {
    "SI (m, kN)": {"L": "m", "F": "kN", "M": "kNm", "q": "kN/m", "A": "m²", "I": "m⁴", "E": "kN/m²"},
    "MKS (m, tonf)": {"L": "m", "F": "tonf", "M": "tonf·m", "q": "tonf/m", "A": "m²", "I": "m⁴", "E": "tonf/m²"},
    "Inglés (ft, kip)": {"L": "ft", "F": "kip", "M": "kip·ft", "q": "kip/ft", "A": "ft²", "I": "ft⁴", "E": "kip/ft²"},
}
u = dicc_u[sistema_unidades]

with st.sidebar.expander("🏗️ Propiedades y Análisis", expanded=False):
    gamma = st.number_input(f"Peso Específico del material (γ)", value=float(st.session_state.gamma_val), format="%.4f", key="gamma_val", help=f"Peso específico en {u['F']}/{u['L']}³")
    st.markdown("---")
    inc_pp = st.toggle("Considerar Peso Propio", value=True)
    inc_sismo = st.toggle("Considerar Sismo", value=True)

tipos_cargas = ["D", "L", "E", "W", "H", "T", "Lr", "G", "F", "Fa", "Ed", "Fs", "Le", "L0"]

pct_D_sismo, pct_L_sismo, pct_Lr_sismo, pct_Le_sismo, pct_L0_sismo = 100.0, 25.0, 25.0, 25.0, 25.0

if inc_sismo:
    with st.sidebar.expander("🌋 Espectro de Diseño (NSR-10)", expanded=False):
        tipo_analisis_sismico = st.radio("Método de Análisis Sísmico", ["Dinámico Modal Espectral", "Fuerza Horizontal Equivalente (FHE)"])
        st.markdown("---")
        Aa = st.number_input("Aa", 0.0, 1.0, 0.20, 0.05)
        Av = st.number_input("Av", 0.0, 1.0, 0.20, 0.05)
        Fa = st.number_input("Fa", 0.0, 3.0, 1.40, 0.1)
        Fv = st.number_input("Fv", 0.0, 3.0, 1.20, 0.1)
        R_factor = st.number_input("Coef. Reducción (R)", 1.0, 8.0, 1.0, 0.5)
        I_imp = st.number_input("Importancia (I)", 1.0, 1.5, 1.0, 0.1)
        dir_sismo = st.radio("Dirección del Sismo", ["X", "Y"])
        st.markdown("---")
        st.markdown("**Porcentajes de Carga para Masa Sísmica**")
        pct_D_sismo = st.slider("% Carga Muerta (D)", 0.0, 100.0, 100.0, 5.0)
        pct_L_sismo = st.slider("% Carga Viva (L)", 0.0, 100.0, 25.0, 5.0)
        pct_Lr_sismo = st.slider("% Carga Viva Cubierta (Lr)", 0.0, 100.0, 25.0, 5.0)
        pct_Le_sismo = st.slider("% Carga Viva Acumulación (Le)", 0.0, 100.0, 25.0, 5.0)
        pct_L0_sismo = st.slider("% Carga Viva Especial (L0)", 0.0, 100.0, 25.0, 5.0)
        st.markdown("---")

with st.sidebar.expander("⚖️ Combinaciones de Carga", expanded=False):
    st.markdown("Defina los factores de mayoración de cargas:")
    col_c1, col_c2 = st.columns(2)
    factores = {}
    for i, t_c in enumerate(tipos_cargas):
        val_def = 1.0
        if i % 2 == 0:
            factores[t_c] = col_c1.number_input(f"Factor {t_c}", value=float(val_def), step=0.1)
        else:
            factores[t_c] = col_c2.number_input(f"Factor {t_c}", value=float(val_def), step=0.1)

with st.sidebar.expander("⬇️ Visualizar cargas", expanded=False):
    grafPunt = st.toggle("Mostrar Cargas Puntuales y momentos", value=True)
    grafSismoPunt = st.toggle("Mostrar Cargas Sísmicas Nodales", value=False)
    grafDistri = st.toggle("Mostrar Cargas Distribuidas Externas", value=True)

with st.sidebar.expander("💢 Visualizar solicitaciones", expanded=False):
    grafN = st.toggle("Mostrar Fuerzas Axiales", value=False)
    grafV = st.toggle("Mostrar Cortante", value=False)
    grafM = st.toggle("Mostrar Momento Flector", value=False)
    grafDef = st.toggle("Mostrar Deformada", value=False)
    grafReac = st.toggle("Mostrar Reacciones", value=False)
    grafTC = st.toggle("Mostrar Tensión/Compresión", value=False)

with st.sidebar.expander("🎚️ Escalas Gráficas", expanded=False):
    st.markdown("**Escalas de Cargas Externas**")
    escPuntuales = st.slider("Escala Cargas Puntuales", 0.1, 3.0, 0.6)
    escDistri = st.slider("Escala Cargas Distribuidas", 0.01, 1.0, 0.1)
    st.markdown("---")
    st.markdown("**Escalas de Fuerzas Internas**")
    escDef = st.slider("Escala Deformación", 100, 1000, 500)
    escM = st.slider("Escala Momentos", 0.01, 0.1, 0.02)
    escV = st.slider("Escala Cortantes", 0.01, 0.1, 0.05)
    escN = st.slider("Escala Axiales", 0.01, 0.1, 0.01)

with st.sidebar.expander("🎨 Estilos Visuales", expanded=False):
    escSizeNodos = st.slider("Tamaño de Nodos", 4, 30, 10)
    escWidthBarras = st.slider("Grosor de Barras", 1, 15, 4)
    escTextNodos = st.slider("Tamaño Letras (Nodos y Barras)", 8, 30, 12)
    escTextFuerzas = st.slider("Tamaño Letras (Valores, Fuerzas y Resultados)", 8, 30, 12)

cfg_nudos = {
    "Nudo": st.column_config.NumberColumn("ID Nodo", required=True, step=1),
    "X": st.column_config.NumberColumn("Nodo X", required=True, format="%.2f"),
    "Y": st.column_config.NumberColumn("Nodo Y", required=True, format="%.2f"),
}
cfg_barras = {
    "Barra": st.column_config.NumberColumn("ID Barra", required=True, step=1),
    "Nudo_Ini": st.column_config.NumberColumn("Nodo Inicio", required=True, step=1),
    "Nudo_Fin": st.column_config.NumberColumn("Nodo Fin", required=True, step=1),
    "Area": st.column_config.NumberColumn("Area", required=True, format="%.4f"),
    "Inercia": st.column_config.NumberColumn("Inercia", required=True, format="%.6f"),
    "E": st.column_config.NumberColumn("E", required=True),
}
cfg_apoyos = {
    "Nudo": st.column_config.NumberColumn("Nodo", required=True, step=1),
    "Dx": st.column_config.NumberColumn("Dx", required=True, min_value=0, max_value=1, step=1),
    "Dy": st.column_config.NumberColumn("Dy", required=True, min_value=0, max_value=1, step=1),
    "Giro": st.column_config.NumberColumn("Giro", required=True, min_value=0, max_value=1, step=1),
    "Orientación": st.column_config.SelectboxColumn("Orientar", options=["Centro", "Izquierda", "Derecha"], required=True, default="Centro"),
}
cfg_puntuales = {
    "Nudo": st.column_config.NumberColumn("Nodo", required=True, step=1),
    "Fx": st.column_config.NumberColumn("Fx", required=True, format="%.2f"),
    "Fy": st.column_config.NumberColumn("Fy", required=True, format="%.2f"),
    "M": st.column_config.NumberColumn("M", required=True, format="%.2f"),
    "Tipo": st.column_config.SelectboxColumn("Tipo", options=tipos_cargas, required=True, default="D"),
}
cfg_distribuidas = {
    "Barra": st.column_config.NumberColumn("Barra", required=True, step=1),
    "q1": st.column_config.NumberColumn("q1", required=True, format="%.2f"),
    "q2": st.column_config.NumberColumn("q2", required=True, format="%.2f"),
    "Tipo": st.column_config.SelectboxColumn("Tipo", options=tipos_cargas, required=True, default="D"),
}

# --- INTERFAZ DE ENTRADAS ---
col1, espacio, col2 = st.columns([0.39, 0.01, 0.6])

with col1:
    st.subheader("📑 Ingreso de datos")
    with st.form("form_ingreso_datos"):
        st.write(f"**Coordenadas de Nudos ({u['L']})**")
        nudos_tmp = st.data_editor(st.session_state.df_nudos, num_rows="dynamic", use_container_width=True, height=200, hide_index=True, column_config=cfg_nudos, key="editor_nudos")
        
        st.write(f"**Barras y Secciones (Area={u['A']}), (Inercia={u['I']}), (E={u['E']})**")
        barras_tmp = st.data_editor(st.session_state.df_barras, num_rows="dynamic", use_container_width=True, height=200, hide_index=True, column_config=cfg_barras, key="editor_barras")
        
        st.write("**Apoyos (Restringido=1, Libre=0)**")
        apoyos_tmp = st.data_editor(st.session_state.df_apoyos_v4, num_rows="dynamic", use_container_width=True, height=150, hide_index=True, column_config=cfg_apoyos, key="editor_apoyos")
        
        st.markdown("---")
        st.subheader("⚖️ Cargas y Masas")
        st.write(f"**Puntuales ({u['F']})**")
        puntuales_tmp = st.data_editor(st.session_state.df_puntuales, num_rows="dynamic", use_container_width=True, height=150, hide_index=True, column_config=cfg_puntuales, key="editor_puntuales")
        
        st.write(f"**Distribuidas ({u['q']})**")
        distribuidas_tmp = st.data_editor(st.session_state.df_distribuidas, num_rows="dynamic", use_container_width=True, height=150, hide_index=True, column_config=cfg_distribuidas, key="editor_distribuidas")
        
        st.markdown("---")
        btn_calcular = st.form_submit_button("🚀 Calcular Estructura", use_container_width=True)

    if btn_calcular:
        st.session_state.df_nudos = nudos_tmp.fillna(0)
        st.session_state.df_barras = barras_tmp.fillna(0)
        st.session_state.df_apoyos_v4 = apoyos_tmp.fillna({"Orientación": "Centro"}).fillna(0)
        
        if "Tipo" not in puntuales_tmp.columns: puntuales_tmp["Tipo"] = "D"
        st.session_state.df_puntuales = puntuales_tmp.fillna({"Tipo": "D"})
        
        if "Tipo" not in distribuidas_tmp.columns: distribuidas_tmp["Tipo"] = "D"
        st.session_state.df_distribuidas = distribuidas_tmp.fillna({"Tipo": "D"})

df_nudos = st.session_state.df_nudos
df_barras = st.session_state.df_barras
df_apoyos = st.session_state.df_apoyos_v4
df_puntuales = st.session_state.df_puntuales
df_distribuidas = st.session_state.df_distribuidas

if len(df_nudos) < 2 or len(df_barras) < 1:
    with col2:
        st.subheader("📊 Visualización Dinámica")
        st.info("👋 **¡Bienvenido a SABE 2D!** \n\nEl lienzo está limpio. Añade al menos **2 nudos, 1 barra y 1 apoyo**.")
    st.stop() 

# --- CÁLCULO MATRICIAL ---
try:
    nudos = df_nudos.to_numpy(dtype=float)
    barras_full = df_barras.to_numpy(dtype=float)
    barras = barras_full[:, 0:5]
    propSeccion = np.column_stack((barras_full[:, 0], barras_full[:, 3:6]))

    mapeo_rotacion = {"Centro": 0.0, "Izquierda": -90.0, "Derecha": 90.0}
    df_apoyos_calc = df_apoyos.copy()
    if "Orientación" in df_apoyos_calc.columns:
        df_apoyos_calc["Orientación"] = df_apoyos_calc["Orientación"].map(mapeo_rotacion).fillna(0.0)
    else:
        df_apoyos_calc["Orientación"] = 0.0
    apoyos = df_apoyos_calc.to_numpy(dtype=float)

    noNudos = nudos.shape[0]
    noBarras = barras.shape[0]

    longBarras = np.zeros(noBarras)
    angulosBarras = np.zeros(noBarras)
    for i in range(noBarras):
        nudoInicio = int(barras[i, 1]) - 1
        nudoFin = int(barras[i, 2]) - 1
        xi, yi = nudos[nudoInicio, 1], nudos[nudoInicio, 2]
        xf, yf = nudos[nudoFin, 1], nudos[nudoFin, 2]
        longBarras[i] = math.sqrt((xf - xi) ** 2 + (yf - yi) ** 2)
        angulosBarras[i] = math.degrees(math.atan2(yf - yi, xf - xi))

    kLocalp = np.zeros((2, 2, noBarras))
    for i in range(noBarras):
        EAL = propSeccion[i, 3] * propSeccion[i, 1] / longBarras[i]
        kLocalp[:, :, i] = EAL * np.array([[1, -1], [-1, 1]])

    T_2x4 = np.zeros((2, 4, noBarras))
    for i in range(noBarras):
        rad = np.radians(angulosBarras[i])
        cs, sn = np.cos(rad), np.sin(rad)
        T_2x4[:, :, i] = np.array([[cs, sn, 0, 0], [0, 0, cs, sn]])

    kLocal = np.zeros((6, 6, noBarras))
    for i in range(noBarras):
        elas, area, inercia, long = propSeccion[i, 3], propSeccion[i, 1], propSeccion[i, 2], longBarras[i]
        EAL = elas * area / long
        EI12 = 12 * elas * inercia / long**3
        EI6 = 6 * elas * inercia / long**2
        EI4 = 4 * elas * inercia / long
        EI2 = 2 * elas * inercia / long
        kLocal[:, :, i] = np.array([
            [EAL, 0, 0, -EAL, 0, 0],
            [0, EI12, EI6, 0, -EI12, EI6],
            [0, EI6, EI4, 0, -EI6, EI2],
            [-EAL, 0, 0, EAL, 0, 0],
            [0, -EI12, -EI6, 0, EI12, -EI6],
            [0, EI6, EI2, 0, -EI6, EI4],
        ])

    kGlobal = np.zeros((noBarras, 6, 6))
    for i in range(noBarras):
        rad = np.radians(angulosBarras[i])
        cs, sn = np.cos(rad), np.sin(rad)
        T = np.array([
            [cs, sn, 0, 0, 0, 0], [-sn, cs, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
            [0, 0, 0, cs, sn, 0], [0, 0, 0, -sn, cs, 0], [0, 0, 0, 0, 0, 1],
        ])
        kGlobal[i] = np.dot(np.dot(T.T, kLocal[:, :, i]), T)
        kGlobal[i] = np.where(np.abs(kGlobal[i]) < 1e-6, 0, kGlobal[i])

    GDLbarras = np.zeros((noBarras, 7), dtype=int)
    for i in range(noBarras):
        GDLbarras[i, 0] = barras[i, 0]
        GDLbarras[i, 1:4] = [barras[i, 1] * 3 - 2, barras[i, 1] * 3 - 1, barras[i, 1] * 3]
        GDLbarras[i, 4:7] = [barras[i, 2] * 3 - 2, barras[i, 2] * 3 - 1, barras[i, 2] * 3]

    kEstructura = np.zeros((noNudos * 3, noNudos * 3))
    for i in range(noBarras):
        gdl = GDLbarras[i, 1:7] - 1
        for row in range(6):
            for col in range(6):
                kEstructura[gdl[row], gdl[col]] += kGlobal[i][row, col]

    # --- APLICACIÓN DE FACTORES DE COMBINACIÓN LRFD ---
    Q = np.zeros((noNudos * 3, 1))
    if len(df_puntuales) > 0:
        for idx, row in df_puntuales.iterrows():
            nudo = int(row["Nudo"])
            tipo = str(row["Tipo"]) if "Tipo" in df_puntuales.columns else "D"
            factor = factores.get(tipo, 1.0)
            Q[nudo * 3 - 3, 0] += float(row["Fx"]) * factor
            Q[nudo * 3 - 2, 0] += float(row["Fy"]) * factor
            Q[nudo * 3 - 1, 0] += float(row["M"]) * factor

    f = np.zeros((noNudos * 3, 1))
    fLocal = np.zeros((6, 1, noBarras))
    q1_ext, q2_ext = np.zeros(noBarras), np.zeros(noBarras)

    if len(df_distribuidas) > 0:
        for idx, row in df_distribuidas.iterrows():
            noB = int(row["Barra"]) - 1
            tipo = str(row["Tipo"]) if "Tipo" in df_distribuidas.columns else "D"
            factor = factores.get(tipo, 1.0)
            q1_ext[noB] += float(row["q1"]) * factor
            q2_ext[noB] += float(row["q2"]) * factor

    q1_tot, q2_tot = q1_ext.copy(), q2_ext.copy()
    px_tot = np.zeros(noBarras) # <-- NUEVO: Para guardar la componente axial de cargas

    if inc_pp:
        for i in range(noBarras):
            A = propSeccion[i, 1]
            wg = np.array([0.0, -gamma * A])
            rad = np.radians(angulosBarras[i])
            c, s = np.cos(rad), np.sin(rad)
            wl = np.array([[c, s], [-s, c]]) @ wg
            factor_pp = factores.get("D", 1.2)
            px_tot[i] += wl[0] * factor_pp  # <-- NUEVO: Componente axial del peso propio
            q1_tot[i] += wl[1] * factor_pp
            q2_tot[i] += wl[1] * factor_pp

    for noBarra in range(1, noBarras + 1):
        q1, q2 = q1_tot[noBarra - 1], q2_tot[noBarra - 1]
        px = px_tot[noBarra - 1] # <-- NUEVO: Rescatamos la carga axial

        if q1 == 0 and q2 == 0 and px == 0: continue # <-- MODIFICADO: Continuar si hay cortante o axial

        nudoInicio, nudoFin, L = int(barras[noBarra - 1, 1]), int(barras[noBarra - 1, 2]), longBarras[noBarra - 1]

        if q1 < q2:
            f2, f3 = 3 / 20 * (q2 - q1) * L + q1 * L / 2, (q2 - q1) / 30 * L**2 + q1 * L**2 / 12
            f5, f6 = 7 / 20 * (q2 - q1) * L + q1 * L / 2, -(q2 - q1) / 20 * L**2 - q1 * L**2 / 12
        elif q1 == q2:
            f2, f3 = q1 * L / 2, q1 * L**2 / 12
            f5, f6 = f2, -f3
        else:
            f2, f3 = 7 / 20 * (q1 - q2) * L + q2 * L / 2, (q1 - q2) / 20 * L**2 + q2 * L**2 / 12
            f5, f6 = 3 / 20 * (q1 - q2) * L + q2 * L / 2, -(q1 - q2) / 30 * L**2 - q2 * L**2 / 12

        # <-- NUEVO: Cálculo de fuerzas de empotramiento perfecto axiales
        f1 = px * L / 2
        f4 = px * L / 2

        # <-- MODIFICADO: Reemplazamos los "0.0" por f1 y f4
        fLocal[:, :, noBarra - 1] = np.array([f1, f2, f3, f4, f5, f6]).reshape((6, 1))
        
        ang = angulosBarras[noBarra - 1]
        cs, sn = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        T_mat = np.array([
            [cs, sn, 0, 0, 0, 0], [-sn, cs, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
            [0, 0, 0, cs, sn, 0], [0, 0, 0, -sn, cs, 0], [0, 0, 0, 0, 0, 1],
        ])
        fGlobal = np.dot(T_mat.T, fLocal[:, :, noBarra - 1]).flatten()
        fGlobal[np.abs(fGlobal) < 1e-5] = 0

        f[int((nudoInicio - 1) * 3):int((nudoInicio - 1) * 3) + 3, 0] += fGlobal[0:3]
        f[int((nudoFin - 1) * 3):int((nudoFin - 1) * 3) + 3, 0] += fGlobal[3:6]

    gdlRestringidos = []
    if len(apoyos) > 0:
        for i in range(apoyos.shape[0]):
            nudo = apoyos[i, 0]
            if apoyos[i, 1] == 1: gdlRestringidos.append(int(nudo * 3 - 2))
            if apoyos[i, 2] == 1: gdlRestringidos.append(int(nudo * 3 - 1))
            if apoyos[i, 3] == 1: gdlRestringidos.append(int(nudo * 3))

    gdlRestringidos = np.array(gdlRestringidos) - 1
    kRed = np.delete(np.delete(np.copy(kEstructura), gdlRestringidos, axis=0), gdlRestringidos, axis=1)
    gdlLibres = np.setdiff1d(np.arange(noNudos * 3), gdlRestringidos)

    Q_sismo = np.zeros((noNudos * 3, 1))
    df_sismo_modos, df_derivas, df_reporte_pisos = None, None, None
    df_fuerzas_nodales_sismo, df_fhe = None, None

    # --- ANÁLISIS SÍSMICO MODAL / FHE ---
    if inc_sismo:
        M_matrix = np.zeros((noNudos * 3, noNudos * 3))
        g_val = 9.80665 if "SI" in sistema_unidades or "MKS" in sistema_unidades else 32.174
        M_struct_nodal = np.zeros(noNudos)
        
        factores_masa = {
            "D": pct_D_sismo / 100.0,
            "L": pct_L_sismo / 100.0,
            "Lr": pct_Lr_sismo / 100.0,
            "Le": pct_Le_sismo / 100.0,
            "L0": pct_L0_sismo / 100.0
        }
        
        if inc_pp:
            factor_m_pp = factores_masa.get("D", 1.0)
            for i in range(noBarras):
                m_bar = (gamma * propSeccion[i, 1] * longBarras[i]) / g_val * factor_m_pp
                M_struct_nodal[int(barras[i, 1]) - 1] += m_bar / 2.0
                M_struct_nodal[int(barras[i, 2]) - 1] += m_bar / 2.0

        if len(df_puntuales) > 0:
            for idx, row in df_puntuales.iterrows():
                nudo = int(row["Nudo"]) - 1
                tipo = str(row["Tipo"]) if "Tipo" in df_puntuales.columns else "D"
                if tipo in factores_masa:
                    peso_puntual = abs(float(row["Fy"]))
                    M_struct_nodal[nudo] += (peso_puntual / g_val) * factores_masa[tipo]

        if len(df_distribuidas) > 0:
            for idx, row in df_distribuidas.iterrows():
                noB = int(row["Barra"]) - 1
                tipo = str(row["Tipo"]) if "Tipo" in df_distribuidas.columns else "D"
                if tipo in factores_masa:
                    q1_v, q2_v = float(row["q1"]), float(row["q2"])
                    res_v = abs((q1_v + q2_v) / 2.0 * longBarras[noB])
                    ni = int(barras[noB, 1]) - 1
                    nf = int(barras[noB, 2]) - 1
                    M_struct_nodal[ni] += ((res_v / 2.0) / g_val) * factores_masa[tipo]
                    M_struct_nodal[nf] += ((res_v / 2.0) / g_val) * factores_masa[tipo]

        for n in range(noNudos):
            M_matrix[n * 3, n * 3] += M_struct_nodal[n]
            M_matrix[n * 3 + 1, n * 3 + 1] += M_struct_nodal[n]

        niveles_y = np.unique(np.round(nudos[:, 2], 3))
        reporte_pisos = []
        for y_val in niveles_y:
            nodos_en_piso = [n for n in range(noNudos) if abs(nudos[n, 2] - y_val) < 1e-3]
            peso_piso = sum(M_struct_nodal[n] * g_val for n in nodos_en_piso)
            reporte_pisos.append({"Nivel (Y)": y_val, "Peso Total Piso": round(peso_piso, 3), "Masa Activa Piso": round(peso_piso / g_val, 3)})
        df_reporte_pisos = pd.DataFrame(reporte_pisos)

        M_red = np.delete(np.delete(M_matrix, gdlRestringidos, axis=0), gdlRestringidos, axis=1)
        active_dofs = np.where(np.diag(M_red) > 1e-9)[0]
        omitted_dofs = np.where(np.diag(M_red) <= 1e-9)[0]

        if len(omitted_dofs) > 0 and len(active_dofs) > 0:
            K_aa = kRed[np.ix_(active_dofs, active_dofs)]
            K_ao = kRed[np.ix_(active_dofs, omitted_dofs)]
            K_oa = kRed[np.ix_(omitted_dofs, active_dofs)]
            K_oo = kRed[np.ix_(omitted_dofs, omitted_dofs)]
            M_aa = M_red[np.ix_(active_dofs, active_dofs)]  
            K_oo_inv = np.linalg.inv(K_oo)
            K_cond = K_aa - K_ao @ K_oo_inv @ K_oa
            w2_a, Phi_a = eigh(K_cond, M_aa)
            idx_ord = np.argsort(w2_a)
            w2_a, Phi_a = w2_a[idx_ord], Phi_a[:, idx_ord]
            w2 = w2_a
            Phi = np.zeros((len(gdlLibres), len(w2_a)))
            Phi[active_dofs, :] = Phi_a
            Phi[omitted_dofs, :] = -K_oo_inv @ K_oa @ Phi_a
        else:
            w2, Phi = eigh(kRed, M_red)
            idx_ord = np.argsort(w2)
            w2, Phi = w2[idx_ord], Phi[:, idx_ord]

        r_vec = np.zeros(len(gdlLibres))
        for k_idx, gdl in enumerate(gdlLibres):
            if dir_sismo == "X" and gdl % 3 == 0: r_vec[k_idx] = 1.0
            elif dir_sismo == "Y" and gdl % 3 == 1: r_vec[k_idx] = 1.0

        total_mass = np.dot(r_vec.T, np.dot(M_red, r_vec))
        datos_modos = []
        u_modos, f_modos = np.zeros((len(gdlLibres), len(w2))), np.zeros((len(gdlLibres), len(w2)))

        for m_idx in range(len(w2)):
            if w2[m_idx] <= 0: continue
            omega = np.sqrt(w2[m_idx])
            T_p = 2 * np.pi / omega
            phi_m = Phi[:, m_idx]
            phi_m = phi_m / np.sqrt(np.dot(phi_m.T, np.dot(M_red, phi_m)))
            Phi[:, m_idx] = phi_m

            Gamma_n = np.dot(phi_m.T, np.dot(M_red, r_vec))
            mass_part = ((Gamma_n**2) / total_mass) * 100 if total_mass > 0 else 0
            Tc = 0.48 * Av * Fv / (Aa * Fa) if (Aa * Fa) > 0 else 0.01
            T0, Tl = 0.1 * Tc, 2.4 * Fv
            if T_p < T0: Sa_g = Aa * Fa * I_imp * (1.0 + 1.5 * T_p / T0)
            elif T_p <= Tc: Sa_g = 2.5 * Aa * Fa * I_imp
            elif T_p <= Tl: Sa_g = 1.2 * Av * Fv * I_imp / T_p
            else: Sa_g = 1.2 * Av * Fv * Tl * I_imp / (T_p**2)
            
            Sa_g_red = Sa_g / R_factor
            Sa_real = Sa_g_red * g_val

            datos_modos.append({"Modo": m_idx + 1, "Periodo (s)": round(T_p, 3), "Frecuencia (Hz)": round(1.0/T_p, 3), "Masa Partic. (%)": round(mass_part, 3), "Sa elástica (g)": round(Sa_g, 3), "Sa de Diseño (g)": round(Sa_g_red, 3), "Γ (Participación)": round(Gamma_n, 3)})
            u_modos[:, m_idx] = phi_m * (Gamma_n * Sa_real / (omega**2))
            f_modos[:, m_idx] = np.dot(M_red, phi_m) * Gamma_n * Sa_real

        if len(datos_modos) > 0:
            df_sismo_modos = pd.DataFrame(datos_modos)
            df_sismo_modos["Masa Acum. (%)"] = np.round(np.cumsum(df_sismo_modos["Masa Partic. (%)"].values), 3)

        if tipo_analisis_sismico == "Fuerza Horizontal Equivalente (FHE)":
            participaciones = [(idx, row["Masa Partic. (%)"], row["Periodo (s)"]) for idx, row in df_sismo_modos.iterrows()]
            participaciones.sort(key=lambda x: x[1], reverse=True)
            T_1 = participaciones[0][2] if participaciones else 0.1
            
            if T_1 < T0: Sa_g_fhe = Aa * Fa * I_imp * (1.0 + 1.5 * T_1 / T0)
            elif T_1 <= Tc: Sa_g_fhe = 2.5 * Aa * Fa * I_imp
            elif T_1 <= Tl: Sa_g_fhe = 1.2 * Av * Fv * I_imp / T_1
            else: Sa_g_fhe = 1.2 * Av * Fv * Tl * I_imp / (T_1**2)
            
            Sa_d = Sa_g_fhe / R_factor
            W_total = sum(p["Peso Total Piso"] for p in reporte_pisos)
            V_s = Sa_d * W_total 
            k_exp = 1.0 if T_1 <= 0.5 else (2.0 if T_1 > 2.5 else 0.75 + 0.5 * T_1)
            sum_Whk = sum(p["Peso Total Piso"] * (p["Nivel (Y)"] ** k_exp) for p in reporte_pisos)
            
            fhe_data = []
            for p in reporte_pisos:
                W_i, h_i = p["Peso Total Piso"], p["Nivel (Y)"]
                C_vx = (W_i * (h_i ** k_exp)) / sum_Whk if sum_Whk > 0 else 0
                F_i = C_vx * V_s
                fhe_data.append({"Nivel (Y)": h_i, "Peso (W)": round(W_i, 3), "Cvx": round(C_vx, 4), "Fuerza Sísmica (F)": round(F_i, 3)})
                nodos_en_piso = [n for n in range(noNudos) if abs(nudos[n, 2] - h_i) < 1e-3]
                if len(nodos_en_piso) > 0:
                    f_nodo = F_i / len(nodos_en_piso)
                    for n in nodos_en_piso:
                        if dir_sismo == "X": Q_sismo[n*3, 0] += f_nodo
                        else: Q_sismo[n*3+1, 0] += f_nodo
            df_fhe = pd.DataFrame(fhe_data)
            U_sismo, F_sismo = np.zeros(noNudos * 3), np.zeros(noNudos * 3)
            F_sismo[gdlLibres] = Q_sismo[gdlLibres].flatten()
        else:
            U_sismo, F_sismo = np.zeros(noNudos * 3), np.zeros(noNudos * 3)
            U_sismo[gdlLibres] = np.sqrt(np.sum(u_modos**2, axis=1))
            F_sismo[gdlLibres] = np.sqrt(np.sum(f_modos**2, axis=1))
            for idx_gdl in gdlLibres:
                if dir_sismo == "X" and idx_gdl % 3 == 0: Q_sismo[idx_gdl, 0] = F_sismo[idx_gdl]
                elif dir_sismo == "Y" and idx_gdl % 3 == 1: Q_sismo[idx_gdl, 0] = F_sismo[idx_gdl]

        derivas_data, prev_disp = [], 0.0
        for i in range(len(niveles_y)):
            y_val = niveles_y[i]
            nodos_nivel = np.where(np.round(nudos[:, 2], 3) == y_val)[0]
            disp_avg = np.mean(U_sismo[nodos_nivel * 3]) if dir_sismo == "X" else np.mean(U_sismo[nodos_nivel * 3 + 1])
            f_total = np.sum(F_sismo[nodos_nivel * 3]) if dir_sismo == "X" else np.sum(F_sismo[nodos_nivel * 3 + 1])

            h, drift = (y_val, disp_avg) if i == 0 else (y_val - niveles_y[i - 1], disp_avg - prev_disp)
            pct_deriva = ((drift * R_factor / I_imp) / h) * 100 if h > 0 else 0
            check = "CUMPLE" if pct_deriva <= 1.0 and h > 0 else ("BASE" if h == 0 else "NO CUMPLE")
            derivas_data.append({"Nivel (Y)": y_val, "Fuerza Sismo Lateral": round(f_total, 3),  "Desplazamiento Elástico": round(disp_avg, 5), "Deriva de Diseño (%)": round(pct_deriva, 3), "Límite NSR-10 (<1.0%)": check})
            prev_disp = disp_avg
        df_derivas = pd.DataFrame(derivas_data)
        
        nodos_sismo = []
        for i in range(noNudos):
            fx_s, fy_s = Q_sismo[i*3, 0], Q_sismo[i*3 + 1, 0]
            if abs(fx_s) > 1e-4 or abs(fy_s) > 1e-4:
                nodos_sismo.append({"Nudo": i + 1, f"Fuerza Sismo X ({u['F']})": round(fx_s, 3), f"Fuerza Sismo Y ({u['F']})": round(fy_s, 3)})
        df_fuerzas_nodales_sismo = pd.DataFrame(nodos_sismo)

        Q_sismo = Q_sismo * factores.get("E", 1.0)

    # --- RESOLUCIÓN SISTEMA COMBINADO ESTÁTICO Y SÍSMICO ---
    fRed_G = np.delete(f + Q, gdlRestringidos, axis=0)
    uRed_G = np.linalg.solve(kRed, fRed_G)
    U_G = np.zeros((noNudos * 3, 1))
    U_G[gdlLibres] = uRed_G.reshape(-1, 1)

    U_E = np.zeros((noNudos * 3, 1))
    if inc_sismo:
        fRed_E = np.delete(Q_sismo, gdlRestringidos, axis=0)
        uRed_E = np.linalg.solve(kRed, fRed_E)
        U_E[gdlLibres] = uRed_E.reshape(-1, 1)

    U = U_G + U_E 
    R_G = np.dot(kEstructura, U_G) - f - Q
    R_G[np.abs(R_G) < 1e-5] = 0
    R_E = np.dot(kEstructura, U_E) - Q_sismo
    R_E[np.abs(R_E) < 1e-5] = 0

    qq1, qq2 = q1_tot.copy(), q2_tot.copy()
    N1_G, V2_G, M3_G, N4_G, V5_G, M6_G = np.zeros(noBarras), np.zeros(noBarras), np.zeros(noBarras), np.zeros(noBarras), np.zeros(noBarras), np.zeros(noBarras)
    C1_G, C2_G = np.zeros(noBarras), np.zeros(noBarras)
    N1_E, V2_E, M3_E, N4_E, V5_E, M6_E = np.zeros(noBarras), np.zeros(noBarras), np.zeros(noBarras), np.zeros(noBarras), np.zeros(noBarras), np.zeros(noBarras)
    C1_E, C2_E = np.zeros(noBarras), np.zeros(noBarras)
    uLocal_G = np.zeros((6, noBarras))

    for i in range(noBarras):
        gdlBarra = GDLbarras[i, 1:7] - 1
        elas, inercia = propSeccion[i, 3], propSeccion[i, 2]
        EI = elas * inercia
        rad = np.radians(angulosBarras[i])
        cs, sn = np.cos(rad), np.sin(rad)
        T_mat = np.array([
            [cs, sn, 0, 0, 0, 0], [-sn, cs, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
            [0, 0, 0, cs, sn, 0], [0, 0, 0, -sn, cs, 0], [0, 0, 0, 0, 0, 1],
        ])

        uLocG = np.dot(T_mat, U_G[gdlBarra]).flatten()
        uLocal_G[:, i] = uLocG
        fuerzasBarra_G = np.dot(kLocal[:, :, i], uLocG) - fLocal[:, :, i].flatten()
        
        N1_G[i], V2_G[i], M3_G[i] = -fuerzasBarra_G[0], fuerzasBarra_G[1], -fuerzasBarra_G[2]
        N4_G[i], V5_G[i], M6_G[i] = fuerzasBarra_G[3], -fuerzasBarra_G[4], fuerzasBarra_G[5]
        
        L = longBarras[i]
        C2_G[i] = M3_G[i]
        C1_G[i] = M6_G[i] / L + (qq2[i] - qq1[i]) * L / 6 + qq1[i] * L / 2 - M3_G[i] / L

        if inc_sismo:
            uLocE = np.dot(T_mat, U_E[gdlBarra]).flatten()
            fuerzasBarra_E = np.dot(kLocal[:, :, i], uLocE)
            N1_E[i], V2_E[i], M3_E[i] = -fuerzasBarra_E[0], fuerzasBarra_E[1], -fuerzasBarra_E[2]
            N4_E[i], V5_E[i], M6_E[i] = fuerzasBarra_E[3], -fuerzasBarra_E[4], fuerzasBarra_E[5]
            C2_E[i] = M3_E[i]
            C1_E[i] = M6_E[i] / L - M3_E[i] / L

    # --- CANVAS GRÁFICO DINÁMICO ---
    with col2:
        st.subheader("📊 Visualización Dinámica")
        st.write("**Gráfica Interactiva (Usa el scroll para Zoom y arrastra para Paneo)**")
        fig = go.Figure()

        color_cargas = {
            "D": ("#FFD100", "rgba(255, 209, 0, 0.4)"),    
            "L": ("#33C3F0", "rgba(51, 195, 240, 0.4)"),   
            "E": ("#FF5733", "rgba(255, 87, 51, 0.4)"),    
            "W": ("#8E44AD", "rgba(142, 68, 173, 0.4)"),   
            "H": ("#27AE60", "rgba(39, 174, 96, 0.4)"),    
            "T": ("#E67E22", "rgba(230, 126, 34, 0.4)"),   
            "Lr": ("#1ABC9C", "rgba(26, 188, 156, 0.4)"),  
            "G": ("#F1C40F", "rgba(241, 196, 15, 0.4)"),    
            "F": ("#34495E", "rgba(52, 73, 94, 0.4)"),     
            "Fa": ("#7F8C8D", "rgba(127, 140, 141, 0.4)"),  
            "Ed": ("#C0392B", "rgba(192, 57, 43, 0.4)"),    
            "Fs": ("#D35400", "rgba(211, 84, 0, 0.4)"),    
            "Le": ("#2980B9", "rgba(41, 128, 185, 0.4)"),  
            "L0": ("#16A085", "rgba(22, 160, 133, 0.4)")   
        }

        added_legend = set()
        def add_legend(name, line_color, fill_color=None, dash=None, marker="square"):
            if name not in added_legend:
                if dash:
                    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=line_color, width=2, dash=dash), name=name, showlegend=True))
                elif fill_color:
                    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(symbol=marker, size=14, color=fill_color, line=dict(color=line_color, width=2)), name=name, showlegend=True))
                else:
                    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=line_color, width=3), name=name, showlegend=True))
                added_legend.add(name)

        if grafTC:
            add_legend(f"Tensión (T) [{u['F']}]", "#0900AA", dash="solid")
            add_legend(f"Compresión (C) [{u['F']}]", "#DE0000", dash="solid")
        
        if grafN:
            if inc_sismo:
                add_legend(f"Env. Axial (+Sismo) [{u['F']}]", "#FF0000", fill_color="rgba(255, 133, 133, 0.3)")
                add_legend(f"Env. Axial (-Sismo) [{u['F']}]", "#0000FF", fill_color="rgba(0, 0, 255, 0.3)")
            else:
                add_legend(f"Fuerza Axial (N) [{u['F']}]", "#8E44AD", fill_color="rgba(142, 68, 173, 0.3)")
        
        if grafV:
            if inc_sismo:
                add_legend(f"Env. Cortante (+Sismo) [{u['F']}]", "#FF0000", fill_color="rgba(255, 0, 0, 0.3)")
                add_legend(f"Env. Cortante (-Sismo) [{u['F']}]", "#0000FF", fill_color="rgba(0, 0, 255, 0.3)")
            else:
                add_legend(f"Cortante (V) [{u['F']}]", "#00843D", fill_color="rgba(0, 199, 46, 0.8)")
        
        if grafM:
            if inc_sismo:
                add_legend(f"Env. Momento (+Sismo) [{u['M']}]", "#FF0000", fill_color="rgba(255, 0, 0, 0.3)")
                add_legend(f"Env. Momento (-Sismo) [{u['M']}]", "#0000FF", fill_color="rgba(0, 0, 255, 0.3)")
            else:
                add_legend(f"Momento (M) [{u['M']}]", "#00FFFF", fill_color="rgba(30, 255, 255, 0.8)")
                
        if grafDef:
            add_legend(f"Deformada [{u['L']}]", "Gray", dash="dash")
            
        if grafReac and len(apoyos) > 0:
            add_legend(f"Reacciones [{u['F']}]", "#FC4BB5", dash="solid")

        if grafDistri and len(df_distribuidas) > 0:
            for t in df_distribuidas["Tipo"].unique():
                c_l, c_f = color_cargas.get(str(t), ("#F2FF00", "rgba(255, 209, 0, 0.4)"))
                add_legend(f"C. Distribuida ({t}) [{u['q']}]", c_l, fill_color=c_f)
        
        if grafPunt and len(df_puntuales) > 0:
            for t in df_puntuales["Tipo"].unique():
                c_l, c_f = color_cargas.get(str(t), ("#FF9933", "rgba(255, 87, 51, 0.4)"))
                add_legend(f"C. Puntual ({t}) [{u['F']}]", c_l, dash="solid")

        if inc_sismo and grafPunt and grafSismoPunt:
            add_legend(f"Fuerza Sísmica Lateral (E) [{u['F']}]", color_cargas["E"][0], dash="solid")

        for i in range(noBarras):
            nudoInicio = int(barras[i, 1]) - 1
            nudoFin = int(barras[i, 2]) - 1
            xi, yi = nudos[nudoInicio, 1], nudos[nudoInicio, 2]
            xf, yf = nudos[nudoFin, 1], nudos[nudoFin, 2]

            fig.add_trace(go.Scatter(
                x=[xi, xf], y=[yi, yf], mode="lines",
                line=dict(color="#005B64", width=escWidthBarras), hoverinfo="text",
                text=f"Barra {i+1}<br>L = {longBarras[i]:.3f} {u['L']}", showlegend=False,
            ))

            if grafTC:
                avg_G = (N1_G[i] + N4_G[i]) / 2
                if inc_sismo:
                    avg_E = (abs(N1_E[i]) + abs(N4_E[i])) / 2
                    val_axial = avg_G - avg_E if avg_G < 0 else avg_G + avg_E
                else:
                    val_axial = avg_G
                
                magnitud = abs(val_axial)
                color_barra = "#DE0000" if val_axial < 0 else "#0900AA"
                estado = "C" if val_axial < 0 else "T"
                fig.add_trace(go.Scatter(x=[xi, xf], y=[yi, yf], mode="lines", line=dict(color=color_barra, width=escWidthBarras + 1), showlegend=False))
                xc, yc = (xi + xf) / 2, (yi + yf) / 2
                fig.add_annotation(
                    x=xc, y=yc, text=f"<b>{magnitud:.3f} {u['F']} ({estado})</b>",
                    showarrow=False, font=dict(color=color_barra, size=escTextFuerzas),
                    bgcolor="rgba(255,255,255,0.85)", bordercolor=color_barra, borderwidth=1,
                )
            else:
                xc, yc = (xi + xf) / 2, (yi + yf) / 2
                fig.add_annotation(x=xc, y=yc, text=f"B{i+1}", showarrow=False, font=dict(color="#FFFFFF", size=escTextNodos))

        for i in range(noNudos):
            xx, yy = nudos[i, 1], nudos[i, 2]
            fig.add_trace(go.Scatter(
                x=[xx], y=[yy], mode="markers+text", marker=dict(color="#002529", size=escSizeNodos, line=dict(color="white", width=1)),
                text=[f"N{i+1}"], textposition="bottom right", textfont=dict(color="#002529", size=escTextNodos, family="Arial Black"),
                hoverinfo="text", hovertext=f"Nodo {i+1}<br>X: {xx:.3f} {u['L']} | Y: {yy:.3f} {u['L']}", showlegend=False,
            ))

        color_apoyo = "#2C3E50"
        size_ap = 0.25
        if len(apoyos) > 0:
            for i in range(apoyos.shape[0]):
                nudo_idx = int(apoyos[i, 0]) - 1
                dx, dy, giro, rot_deg = apoyos[i, 1], apoyos[i, 2], apoyos[i, 3], apoyos[i, 4] if apoyos.shape[1] > 4 else 0.0
                xo, yo = nudos[nudo_idx, 1], nudos[nudo_idx, 2]
                cos_r, sin_r = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))

                def trans_p(x_arr, y_arr): return (xo + x_arr * cos_r - y_arr * sin_r, yo + x_arr * sin_r + y_arr * cos_r)

                if dx == 1 and dy == 1 and giro == 1:
                    xg, yg = trans_p(np.array([-size_ap, size_ap]), np.array([0.0, 0.0]))
                    fig.add_trace(go.Scatter(x=xg, y=yg, mode="lines", line=dict(color=color_apoyo, width=4), showlegend=False, hoverinfo="skip"))
                    for step in np.linspace(-size_ap, size_ap, 5):
                        xg_h, yg_h = trans_p(np.array([step, step - 0.08]), np.array([0.0, -0.12]))
                        fig.add_trace(go.Scatter(x=xg_h, y=yg_h, mode="lines", line=dict(color=color_apoyo, width=1.5), showlegend=False, hoverinfo="skip"))
                elif dx == 1 and dy == 1 and giro == 0:
                    xg, yg = trans_p(np.array([0.0, -size_ap, size_ap, 0.0]), np.array([0.0, -size_ap, -size_ap, 0.0]))
                    fig.add_trace(go.Scatter(x=xg, y=yg, mode="lines", fill="toself", fillcolor="rgba(44, 62, 80, 0.2)", line=dict(color=color_apoyo, width=2), showlegend=False, hoverinfo="skip"))
                    xg_b, yg_b = trans_p(np.array([-size_ap - 0.1, size_ap + 0.1]), np.array([-size_ap, -size_ap]))
                    fig.add_trace(go.Scatter(x=xg_b, y=yg_b, mode="lines", line=dict(color=color_apoyo, width=2), showlegend=False, hoverinfo="skip"))
                elif (dx == 0 and dy == 1) or (dx == 1 and dy == 0):
                    xg, yg = trans_p(np.array([0.0, -size_ap, size_ap, 0.0]), np.array([0.0, -size_ap, -size_ap, 0.0]))
                    fig.add_trace(go.Scatter(x=xg, y=yg, mode="lines", fill="toself", fillcolor="rgba(44, 62, 80, 0.05)", line=dict(color=color_apoyo, width=2), showlegend=False, hoverinfo="skip"))
                    xw1, yw1 = trans_p(-size_ap / 2, -size_ap - 0.04)
                    xw2, yw2 = trans_p(size_ap / 2, -size_ap - 0.04)
                    fig.add_trace(go.Scatter(x=[xw1, xw2], y=[yw1, yw2], mode="markers", marker=dict(color=color_apoyo, size=7, symbol="circle", line=dict(color="white", width=0.5)), showlegend=False, hoverinfo="skip"))

        if grafDef:
            nudosDesp = np.zeros_like(nudos)
            for i in range(noNudos):
                dx_val, dy_val = U[i * 3].item(), U[i * 3 + 1].item()
                nudosDesp[i, 1] = nudos[i, 1] + dx_val * escDef
                nudosDesp[i, 2] = nudos[i, 2] + dy_val * escDef
                if abs(dx_val) > 1e-6 or abs(dy_val) > 1e-6:
                    fig.add_trace(go.Scatter(
                        x=[nudosDesp[i, 1]], y=[nudosDesp[i, 2]], mode="markers+text",
                        marker=dict(color="Gray", size=6),
                        text=[
                            f"<span style='text-shadow: 0px 0px 8px rgba(0, 0, 0, 0.8), 0px 0px 12px rgba(0, 0, 0, 0.8);'>"
                            f"Δx: {dx_val:.6f} {u['L']}<br>Δy: {dy_val:.6f} {u['L']}"
                            f"</span>"
                        ],
                        textposition="top center", textfont=dict(color="White", size=escTextFuerzas), showlegend=False,
                    ))
            
            for i in range(noBarras):
                nudoInicio = int(barras[i, 1]) - 1
                nudoFin = int(barras[i, 2]) - 1
                xo, yo = nudos[nudoInicio, 1], nudos[nudoInicio, 2]
                ang_rad = np.radians(angulosBarras[i])
                L = longBarras[i]

                # Extraer desplazamientos totales de los nudos de la barra usando U global
                gdlBarra = GDLbarras[i, 1:7] - 1
                cs, sn = np.cos(ang_rad), np.sin(ang_rad)
                T_mat = np.array([
                    [cs, sn, 0, 0, 0, 0], [-sn, cs, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, cs, sn, 0], [0, 0, 0, -sn, cs, 0], [0, 0, 0, 0, 0, 1],
                ])
                uLocTotal = np.dot(T_mat, U[gdlBarra]).flatten()

                u1, v1, th1 = uLocTotal[0], uLocTotal[1], uLocTotal[2]
                u2, v2, th2 = uLocTotal[3], uLocTotal[4], uLocTotal[5]

                x_vals = np.linspace(0, L, 21)
                xi = x_vals / L

                N1_h = 1 - 3*xi**2 + 2*xi**3
                N2_h = L * (xi - 2*xi**2 + xi**3)
                N3_h = 3*xi**2 - 2*xi**3
                N4_h = L * (-xi**2 + xi**3)

                u_loc = (1 - xi) * u1 + xi * u2
                v_loc = N1_h * v1 + N2_h * th1 + N3_h * v2 + N4_h * th2

                u_plot = u_loc * escDef
                v_plot = v_loc * escDef

                x_loc_def = x_vals + u_plot
                y_loc_def = v_plot

                x_glob_def = x_loc_def * np.cos(ang_rad) - y_loc_def * np.sin(ang_rad) + xo
                y_glob_def = x_loc_def * np.sin(ang_rad) + y_loc_def * np.cos(ang_rad) + yo

                fig.add_trace(go.Scatter(x=x_glob_def, y=y_glob_def, mode="lines", line=dict(color="Gray", width=2, dash="dash"), showlegend=False))

        if grafM:
            for i in range(noBarras):
                nudoInicio = int(barras[i, 1]) - 1
                xo, yo = nudos[nudoInicio, 1], nudos[nudoInicio, 2]
                ang_rad, L = np.radians(angulosBarras[i]), longBarras[i]
                x = np.linspace(0, L, 21)
                
                M_lineal = C2_G[i] + (M6_G[i] - C2_G[i]) * (x / L)
                M_cargas = - (qq1[i] / 2) * x * (L - x) - ((qq2[i] - qq1[i]) / (6 * L)) * x * (L**2 - x**2)
                y_G = M_lineal + M_cargas
                
                if inc_sismo:
                    y_E = C1_E[i] * x + C2_E[i]
                    envelopes = [
                        (y_G + np.abs(y_E), '#FF0000', 'rgba(255, 0, 0, 0.2)'),
                        (y_G - np.abs(y_E), '#0000FF', 'rgba(0, 0, 255, 0.2)')
                    ]
                else:
                    envelopes = [(y_G, "#03B8B8", 'rgba(39, 245, 221, 0.5)')]

                for y_vals, c_line, c_fill in envelopes:
                    yEsc = -y_vals * escM
                    xRot, yRot = x * np.cos(ang_rad) - yEsc * np.sin(ang_rad), x * np.sin(ang_rad) + yEsc * np.cos(ang_rad)
                    xVigaRot, yVigaRot = x * np.cos(ang_rad), x * np.sin(ang_rad)
                    xTras, yTras = xRot + xo, yRot + yo
                    xVigaTras, yVigaTras = xVigaRot + xo, yVigaRot + yo
                    x_poly, y_poly = np.concatenate([xTras, xVigaTras[::-1]]), np.concatenate([yTras, yVigaTras[::-1]])

                    fig.add_trace(go.Scatter(x=x_poly, y=y_poly, fill='toself', fillcolor=c_fill, mode='lines', line=dict(color=c_line, width=1.5), showlegend=False, hoverinfo='skip'))
                    m_ini, m_fin = y_vals[0], y_vals[-1]
                    if abs(m_ini) > 1e-4: fig.add_annotation(x=xTras[0], y=yTras[0], text=f"<b>{m_ini:.3f} {u['M']}</b>", showarrow=False, font=dict(color=c_line, size=escTextFuerzas), bgcolor="rgba(255,255,255,0.8)")
                    if abs(m_fin) > 1e-4: fig.add_annotation(x=xTras[-1], y=yTras[-1], text=f"<b>{m_fin:.3f} {u['M']}</b>", showarrow=False, font=dict(color=c_line, size=escTextFuerzas), bgcolor="rgba(255,255,255,0.8)")

        if grafV:
            for i in range(noBarras):
                nudoInicio = int(barras[i, 1]) - 1
                xo, yo = nudos[nudoInicio, 1], nudos[nudoInicio, 2]
                ang_rad, L = np.radians(angulosBarras[i]), longBarras[i]
                v_ini_G, v_fin_G = V2_G[i], V5_G[i]
                
                if inc_sismo:
                    v_i_E, v_f_E = abs(V2_E[i]), abs(V5_E[i])
                    envelopes = [
                        (v_ini_G + v_i_E, v_fin_G + v_f_E, '#FF0000', 'rgba(255, 0, 0, 0.2)'),
                        (v_ini_G - v_i_E, v_fin_G - v_f_E, '#0000FF', 'rgba(0, 0, 255, 0.2)')
                    ]
                else:
                    envelopes = [(v_ini_G, v_fin_G, "#025F2D", 'rgba(0, 199, 46, 0.5)')]

                for v_ini, v_fin, c_line, c_fill in envelopes:
                    if abs(v_ini) < 1e-4 and abs(v_fin) < 1e-4: continue
                    x_arr, y_arr = np.array([0, 0, L, L]), np.array([0, -v_ini * escV, -v_fin * escV, 0])
                    xRot, yRot = x_arr * np.cos(ang_rad) - y_arr * np.sin(ang_rad) + xo, x_arr * np.sin(ang_rad) + y_arr * np.cos(ang_rad) + yo
                    fig.add_trace(go.Scatter(x=xRot, y=yRot, fill="toself", fillcolor=c_fill, mode="lines", line=dict(color=c_line, width=1), showlegend=False))
                    xt1, yt1 = 0 * np.cos(ang_rad) - (-v_ini * escV) * np.sin(ang_rad) + xo, 0 * np.sin(ang_rad) + (-v_ini * escV) * np.cos(ang_rad) + yo
                    fig.add_annotation(x=xt1, y=yt1, text=f"<b>{v_ini:.3f} {u['F']}</b>", showarrow=False, font=dict(color=c_line, size=escTextFuerzas), bgcolor="rgba(255,255,255,0.8)")
                    xt2, yt2 = L * np.cos(ang_rad) - (-v_fin * escV) * np.sin(ang_rad) + xo, L * np.sin(ang_rad) + (-v_fin * escV) * np.cos(ang_rad) + yo
                    fig.add_annotation(x=xt2, y=yt2, text=f"<b>{v_fin:.3f} {u['F']}</b>", showarrow=False, font=dict(color=c_line, size=escTextFuerzas), bgcolor="rgba(255,255,255,0.8)")

        if grafN:
            for i in range(noBarras):
                nudoInicio = int(barras[i, 1]) - 1
                xo, yo = nudos[nudoInicio, 1], nudos[nudoInicio, 2]
                ang_rad, L = np.radians(angulosBarras[i]), longBarras[i]
                n_ini_G, n_fin_G = N1_G[i], N4_G[i]
                
                if inc_sismo:
                    n_i_E, n_f_E = abs(N1_E[i]), abs(N4_E[i])
                    envelopes = [
                        (n_ini_G + n_i_E, n_fin_G + n_f_E, '#FF0000', 'rgba(255, 0, 0, 0.2)'),
                        (n_ini_G - n_i_E, n_fin_G - n_f_E, '#0000FF', 'rgba(0, 0, 255, 0.2)')
                    ]
                else:
                    c_line = "#8E44AD"
                    c_fill = "rgba(142, 68, 173, 0.5)"
                    envelopes = [(n_ini_G, n_fin_G, c_line, c_fill)]

                for item in envelopes:
                    n_ini, n_fin = item[0], item[1]
                    c_line, c_fill = item[2], item[3]
                    
                    if abs(n_ini) < 1e-4 and abs(n_fin) < 1e-4: continue
                    x_arr, y_arr = np.array([0, 0, L, L]), np.array([0, -n_ini * escN, -n_fin * escN, 0])
                    xRot, yRot = x_arr * np.cos(ang_rad) - y_arr * np.sin(ang_rad) + xo, x_arr * np.sin(ang_rad) + y_arr * np.cos(ang_rad) + yo
                    fig.add_trace(go.Scatter(x=xRot, y=yRot, fill="toself", fillcolor=c_fill, mode="lines", line=dict(color=c_line, width=1), showlegend=False))
                    xt1, yt1 = 0 * np.cos(ang_rad) - (-n_ini * escN) * np.sin(ang_rad) + xo, 0 * np.sin(ang_rad) + (-n_ini * escN) * np.cos(ang_rad) + yo
                    fig.add_annotation(x=xt1, y=yt1, text=f"<b>{n_ini:.3f} {u['F']}</b>", showarrow=False, font=dict(color=c_line, size=escTextFuerzas), bgcolor="rgba(255,255,255,0.8)")
                    xt2, yt2 = L * np.cos(ang_rad) - (-n_fin * escN) * np.sin(ang_rad) + xo, L * np.sin(ang_rad) + (-n_fin * escN) * np.cos(ang_rad) + yo
                    fig.add_annotation(x=xt2, y=yt2, text=f"<b>{n_fin:.3f} {u['F']}</b>", showarrow=False, font=dict(color=c_line, size=escTextFuerzas), bgcolor="rgba(255,255,255,0.8)")

        if grafDistri and len(df_distribuidas) > 0:
            for idx, row in df_distribuidas.iterrows():
                noBarra_ext = int(row["Barra"]) - 1
                q1_ex, q2_ex = float(row["q1"]), float(row["q2"])
                tipo_c = str(row["Tipo"]) if "Tipo" in df_distribuidas.columns else "D"
                
                c_line, c_fill = color_cargas.get(tipo_c, ("#FFD100", "rgba(255, 209, 0, 0.4)"))
                
                if abs(q1_ex) > 0 or abs(q2_ex) > 0:
                    nud_ini_idx = int(barras[noBarra_ext, 1]) - 1
                    xo, yo = nudos[nud_ini_idx, 1], nudos[nud_ini_idx, 2]
                    ang_rad, L = np.radians(angulosBarras[noBarra_ext]), longBarras[noBarra_ext]
                    escq = escDistri
                    
                    x_loc, y_loc = np.array([0, 0, L, L]), np.array([0, q1_ex * escq, q2_ex * escq, 0])
                    xRot, yRot = x_loc * np.cos(ang_rad) - y_loc * np.sin(ang_rad) + xo, x_loc * np.sin(ang_rad) + y_loc * np.cos(ang_rad) + yo
                    fig.add_trace(go.Scatter(x=xRot, y=yRot, fill="toself", fillcolor=c_fill, mode="lines", line=dict(color=c_line, width=2), showlegend=False))
                    
                    x_q1, y_q1 = 0 * np.cos(ang_rad) - (q1_ex * escq) * np.sin(ang_rad) + xo, 0 * np.sin(ang_rad) + (q1_ex * escq) * np.cos(ang_rad) + yo
                    x_q2, y_q2 = L * np.cos(ang_rad) - (q2_ex * escq) * np.sin(ang_rad) + xo, L * np.sin(ang_rad) + (q2_ex * escq) * np.cos(ang_rad) + yo
                    
                    if abs(q1_ex) > 0:
                        fig.add_annotation(x=x_q1, y=y_q1, text=f"<b>{q1_ex:.2f} {u['q']}</b>", showarrow=False, font=dict(color="#000000", size=escTextFuerzas), bgcolor=c_fill)
                    if abs(q2_ex) > 0:
                        fig.add_annotation(x=x_q2, y=y_q2, text=f"<b>{q2_ex:.2f} {u['q']}</b>", showarrow=False, font=dict(color="#000000", size=escTextFuerzas), bgcolor=c_fill)

        if grafPunt and len(df_puntuales) > 0:
            for idx, row in df_puntuales.iterrows():
                nudo = int(row["Nudo"]) - 1
                fx, fy, mz = float(row["Fx"]), float(row["Fy"]), float(row["M"])
                tipo_p = str(row["Tipo"]) if "Tipo" in df_puntuales.columns else "D"
                
                c_line, c_fill = color_cargas.get(tipo_p, ("#FF5733", "rgba(255, 87, 51, 0.85)"))
                xo, yo = nudos[nudo, 1], nudos[nudo, 2]
                
                if abs(fx) > 0.1 or abs(fy) > 0.1:
                    ax_val = xo - np.sign(fx) * escPuntuales if fx != 0 else xo
                    ay_val = yo - np.sign(fy) * escPuntuales if fy != 0 else yo
                    fig.add_annotation(
                        x=xo, y=yo, ax=ax_val, ay=ay_val,
                        xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=c_line,
                    )
                    tx_val = xo - np.sign(fx) * (escPuntuales + 0.1) if fx != 0 else xo
                    ty_val = yo - np.sign(fy) * (escPuntuales + 0.1) if fy != 0 else yo
                    fig.add_annotation(x=tx_val, y=ty_val, text=f"<b>({fx:.2f}, {fy:.2f}) {u['F']}</b>", showarrow=False, font=dict(color=c_line, size=escTextFuerzas), bgcolor="rgba(255,255,255,0.8)")
                if abs(mz) > 0.1:
                    fig.add_annotation(x=xo, y=yo + escPuntuales, text=f"<b>↻ M={mz:.2f} {u['M']}</b>", showarrow=False, font=dict(color=c_line, size=escTextFuerzas), bgcolor="rgba(255,255,255,0.8)")

        if inc_sismo and grafPunt and grafSismoPunt:
            for i in range(noNudos):
                fx_s, fy_s = Q_sismo[i*3, 0], Q_sismo[i*3 + 1, 0]
                if abs(fx_s) > 0.1 or abs(fy_s) > 0.1:
                    xo, yo = nudos[i, 1], nudos[i, 2]
                    c_line_sismo = color_cargas["E"][0]
                    ax_val = xo - np.sign(fx_s) * escPuntuales if fx_s != 0 else xo
                    ay_val = yo - np.sign(fy_s) * escPuntuales if fy_s != 0 else yo
                    fig.add_annotation(
                        x=xo, y=yo, ax=ax_val, ay=ay_val,
                        xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=3, arrowcolor=c_line_sismo,
                    )
                    tx_val = xo - np.sign(fx_s) * (escPuntuales + 0.15) if fx_s != 0 else xo
                    ty_val = yo - np.sign(fy_s) * (escPuntuales + 0.15) if fy_s != 0 else yo
                    fig.add_annotation(x=tx_val, y=ty_val, text=f"<b>E: ({fx_s:.2f}, {fy_s:.2f}) {u['F']}</b>", showarrow=False, font=dict(color=c_line_sismo, size=escTextFuerzas), bgcolor="rgba(255,255,255,0.8)")

        if grafReac and len(apoyos) > 0:
            escReacGraf = 0.02
            for i in range(apoyos.shape[0]):
                nudo = int(apoyos[i, 0]) - 1
                rx, ry = R_G[nudo * 3].item(), R_G[nudo * 3 + 1].item()
                if inc_sismo:
                    rx_E, ry_E = abs(R_E[nudo * 3].item()), abs(R_E[nudo * 3 + 1].item())
                    texto_rx = f"Rx: {rx:.2f} ± {rx_E:.2f}"
                    texto_ry = f"Ry: {ry:.2f} ± {ry_E:.2f}"
                else:
                    texto_rx = f"Rx: {rx:.3f}"
                    texto_ry = f"Ry: {ry:.3f}"

                xo, yo = nudos[nudo, 1], nudos[nudo, 2]
                if abs(rx) > 1e-4 or abs(ry) > 1e-4:
                    fig.add_annotation(
                        x=xo, y=yo, ax=xo - rx * escReacGraf, ay=yo - ry * escReacGraf,
                        xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="#FC4BB5",
                    )
                    fig.add_annotation(x=xo - rx * escReacGraf - 0.2, y=yo - ry * escReacGraf - 0.2, text=f"<b>{texto_rx}<br>{texto_ry} {u['F']}</b>", showarrow=False, font=dict(color="#F942B6", size=escTextFuerzas), bgcolor="rgba(255,255,255,0.8)")

        fig.update_layout(
            plot_bgcolor="#AED1E8", paper_bgcolor="#005B64",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(0, 37, 41, 0.8)", bordercolor="#00FFFF", borderwidth=1,
                font=dict(color="white", size=12)
            ),
            xaxis=dict(title=f'Coordenadas X ({u["L"]})', color="#FFFFFF", gridcolor="rgba(0, 0, 0, 0.15)", gridwidth=1, showgrid=True, zerolinecolor="#00FFFF", zerolinewidth=1.5),
            yaxis=dict(title=f'Coordenadas Y ({u["L"]})', color="#FFFFFF", gridcolor="rgba(0, 0, 0, 0.15)", gridwidth=1, showgrid=True, zerolinecolor="#2D2D2D", zerolinewidth=1.5, scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, t=40, b=20), height=650, dragmode="pan",
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

        # --- VISTA DETALLADA AISLADA POR BARRA ---
        st.markdown("---")
        st.subheader("🔍 Vista Detallada por Barra")
        barra_sel = st.selectbox("Seleccione la barra a detallar (Axial, Cortante y Momento):", range(1, noBarras + 1))
        idx_b = barra_sel - 1
        L_b = longBarras[idx_b]
        x_iso = np.linspace(0, L_b, 100)
        
        q1_b, q2_b = qq1[idx_b], qq2[idx_b]

        N_x_G = np.linspace(N1_G[idx_b], N4_G[idx_b], len(x_iso))
        
        M_lineal = C2_G[idx_b] + (M6_G[idx_b] - C2_G[idx_b]) * (x_iso / L_b)
        M_cargas = - (q1_b / 2) * x_iso * (L_b - x_iso) - ((q2_b - q1_b) / (6 * L_b)) * x_iso * (L_b**2 - x_iso**2)
        M_x_G = M_lineal + M_cargas
            
        fig_iso = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=(
                f"Diagrama Axial ({u['F']}) - Barra {barra_sel}", 
                f"Diagrama de Cortante ({u['F']}) - Barra {barra_sel}", 
                f"Diagrama de Momento Flector ({u['M']}) - Barra {barra_sel}"
            )
        )
        
        x_poly = np.concatenate([x_iso, x_iso[::-1]])
        zeros = np.zeros_like(x_iso)

        if inc_sismo:
            N_E_abs = np.linspace(abs(N1_E[idx_b]), abs(N4_E[idx_b]), len(x_iso))
            V_E_abs = np.linspace(abs(V2_E[idx_b]), abs(V5_E[idx_b]), len(x_iso))
            M_E_abs = np.abs(C1_E[idx_b] * x_iso + C2_E[idx_b])

            n_red = N_x_G + N_E_abs
            n_blue = N_x_G - N_E_abs
            
            fig_iso.add_trace(go.Scatter(x=x_poly, y=np.concatenate([n_red, zeros]), fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', mode='lines', line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
            fig_iso.add_trace(go.Scatter(x=x_poly, y=np.concatenate([n_blue, zeros]), fill='toself', fillcolor='rgba(0, 0, 255, 0.2)', mode='lines', line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
            fig_iso.add_trace(go.Scatter(x=x_iso, y=n_red, mode="lines", line=dict(color="#FF0000", width=2.5), name="Env. Axial (+Sismo)"), row=1, col=1)
            fig_iso.add_trace(go.Scatter(x=x_iso, y=n_blue, mode="lines", line=dict(color="#0000FF", width=2.5), name="Env. Axial (-Sismo)"), row=1, col=1)
            
            V_x_G_plot = np.linspace(V2_G[idx_b], V5_G[idx_b], len(x_iso))
            v_red = -(V_x_G_plot + V_E_abs)
            v_blue = -(V_x_G_plot - V_E_abs)
            
            fig_iso.add_trace(go.Scatter(x=x_poly, y=np.concatenate([v_red, zeros]), fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', mode='lines', line=dict(width=0), hoverinfo="skip", showlegend=False), row=2, col=1)
            fig_iso.add_trace(go.Scatter(x=x_poly, y=np.concatenate([v_blue, zeros]), fill='toself', fillcolor='rgba(0, 0, 255, 0.2)', mode='lines', line=dict(width=0), hoverinfo="skip", showlegend=False), row=2, col=1)
            fig_iso.add_trace(go.Scatter(x=x_iso, y=v_red, mode="lines", line=dict(color="#FF0000", width=2.5), name="Env. Cortante (+Sismo)"), row=2, col=1)
            fig_iso.add_trace(go.Scatter(x=x_iso, y=v_blue, mode="lines", line=dict(color="#0000FF", width=2.5), name="Env. Cortante (-Sismo)"), row=2, col=1)

            m_red = -(M_x_G + M_E_abs)
            m_blue = -(M_x_G - M_E_abs)
            
            fig_iso.add_trace(go.Scatter(x=x_poly, y=np.concatenate([m_red, zeros]), fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', mode='lines', line=dict(width=0), hoverinfo="skip", showlegend=False), row=3, col=1)
            fig_iso.add_trace(go.Scatter(x=x_poly, y=np.concatenate([m_blue, zeros]), fill='toself', fillcolor='rgba(0, 0, 255, 0.2)', mode='lines', line=dict(width=0), hoverinfo="skip", showlegend=False), row=3, col=1)
            fig_iso.add_trace(go.Scatter(x=x_iso, y=m_red, mode="lines", line=dict(color="#FF0000", width=2.5), name="Env. Momento (+Sismo)"), row=3, col=1)
            fig_iso.add_trace(go.Scatter(x=x_iso, y=m_blue, mode="lines", line=dict(color="#0000FF", width=2.5), name="Env. Momento (-Sismo)"), row=3, col=1)

        else:
            c_line_N = "#8E44AD"
            c_fill_N = 'rgba(142, 68, 173, 0.4)'        
            
            fig_iso.add_trace(go.Scatter(x=x_poly, y=np.concatenate([N_x_G, zeros]), fill='toself', fillcolor=c_fill_N, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
            fig_iso.add_trace(go.Scatter(x=x_iso, y=N_x_G, mode="lines", line=dict(color=c_line_N, width=2.5), name="Axial"), row=1, col=1)
            
            V_x_G_plot_static = -np.linspace(V2_G[idx_b], V5_G[idx_b], len(x_iso))
            fig_iso.add_trace(go.Scatter(x=x_poly, y=np.concatenate([V_x_G_plot_static, zeros]), fill='toself', fillcolor='rgba(0, 199, 46, 0.4)', mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)
            fig_iso.add_trace(go.Scatter(x=x_iso, y=V_x_G_plot_static, mode="lines", line=dict(color="#025F2D", width=2.5), name="Cortante"), row=2, col=1)
            
            fig_iso.add_trace(go.Scatter(x=x_poly, y=np.concatenate([-M_x_G, zeros]), fill='toself', fillcolor='rgba(39, 245, 221, 0.4)', mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=3, col=1)
            fig_iso.add_trace(go.Scatter(x=x_iso, y=-M_x_G, mode="lines", line=dict(color="#03B8B8", width=2.5), name="Momento"), row=3, col=1) 

        fig_iso.add_hline(y=0, line_dash="dash", line_color="#000000", line_width=3, row=1, col=1)
        fig_iso.add_hline(y=0, line_dash="dash", line_color="#000000", line_width=3, row=2, col=1)
        fig_iso.add_hline(y=0, line_dash="dash", line_color="#000000", line_width=3, row=3, col=1)

        fig_iso.update_traces(hovertemplate="Valor: %{y:.2f}<extra></extra>")
        
        fig_iso.update_xaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.2)", gridwidth=1, zeroline=True, zerolinecolor="rgba(0,0,0,0.5)", tickformat=".2f", row=1, col=1)
        fig_iso.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.2)", gridwidth=1, zeroline=True, zerolinecolor="rgba(0,0,0,0.5)", row=1, col=1)
        
        fig_iso.update_xaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.2)", gridwidth=1, zeroline=True, zerolinecolor="rgba(0,0,0,0.5)", tickformat=".2f", row=2, col=1)
        fig_iso.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.2)", gridwidth=1, zeroline=True, zerolinecolor="rgba(0,0,0,0.5)", row=2, col=1)
        
        fig_iso.update_xaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.2)", gridwidth=1, zeroline=True, zerolinecolor="rgba(0,0,0,0.5)", title_text=f"Distancia X ({u['L']})", tickformat=".2f", row=3, col=1)
        fig_iso.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.2)", gridwidth=1, zeroline=True, zerolinecolor="rgba(0,0,0,0.5)", row=3, col=1)

        fig_iso.update_layout(
            hovermode="x unified",
            height=750, template="plotly_white", showlegend=False, 
            margin=dict(l=20, r=20, t=50, b=20), plot_bgcolor="#AED1E8", paper_bgcolor="#005B64",
        )
        st.plotly_chart(fig_iso, use_container_width=True)

        # --- REPORTE DE RESULTADOS ---
        st.markdown("---")
        st.subheader("📚 Reporte de cálculos método matricial")
        st.write("Explora los resultados analíticos paso a paso.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nudos Totales", noNudos)
        c2.metric("Barras Totales", noBarras)
        c3.metric("Grados de Libertad", noNudos * 3)
        c4.metric("GDL Restringidos", len(gdlRestringidos))

        tabs_list = ["📏 Geometría y Propiedades", "🧩 Matrices por Barra", "🌐 Matriz Global Ensamblada"]
        if inc_sismo: tabs_list.append("🌋 Análisis Sísmico (NSR-10)")
        tabs = st.tabs(tabs_list)

        with tabs[0]:
            df_props = pd.DataFrame({
                "Barra": range(1, noBarras + 1),
                f"Longitud ({u['L']})": np.round(longBarras, 3),
                "Ángulo (°)": np.round(angulosBarras, 3),
                "Cos(θ)": np.round(np.cos(np.radians(angulosBarras)), 3),
                "Sen(θ)": np.round(np.sin(np.radians(angulosBarras)), 3),
                f"Área ({u['A']})": np.round(propSeccion[:, 1], 4),
                f"Inercia ({u['I']})": np.round(propSeccion[:, 2], 6),
            })
            st.dataframe(df_props, hide_index=True, use_container_width=True)

        with tabs[1]:
            barra_sel_t = st.selectbox("Seleccione la barra para inspeccionar sus matrices:", range(1, noBarras + 1))
            idx_t = barra_sel_t - 1
            col_mat1, col_mat2 = st.columns(2)
            with col_mat1:
                st.write(f"Matriz Local ($k$) - Barra {barra_sel_t}")
                st.dataframe(np.round(kLocal[:, :, idx_t], 3), hide_index=True, use_container_width=True)
            with col_mat2:
                st.write(f"Matriz Global ($K$) - Barra {barra_sel_t}")
                st.dataframe(np.round(kGlobal[idx_t], 3), hide_index=True, use_container_width=True)

        with tabs[2]:
            etiquetas_gdl = [f"GDL {i+1}" for i in range(noNudos * 3)]
            df_kEstructura = pd.DataFrame(np.round(kEstructura, 3), index=etiquetas_gdl, columns=etiquetas_gdl)
            st.dataframe(df_kEstructura, use_container_width=True)

        if inc_sismo:
            with tabs[3]:
                st.subheader(f"Resultados del Análisis Sísmico ({tipo_analisis_sismico})")
                
                c_s1, c_s2 = st.columns(2)
                with c_s1:
                    st.write("**1. Análisis de Cargas por Nivel (Takedown)**")
                    st.dataframe(df_reporte_pisos, hide_index=True, use_container_width=True)
                with c_s2:
                    if tipo_analisis_sismico == "Fuerza Horizontal Equivalente (FHE)":
                        st.write("**2. Verificación del Período Utilizado**")
                        participaciones_fhe = [(idx, row["Masa Partic. (%)"], row["Periodo (s)"]) for idx, row in df_sismo_modos.iterrows()]
                        participaciones_fhe.sort(key=lambda x: x[1], reverse=True)
                        T_1_val = participaciones_fhe[0][2] if participaciones_fhe else 0.1
                        masa_total_val = np.sum(df_reporte_pisos["Masa Activa Piso"])
                        df_resumen_periodo = pd.DataFrame([{
                            "Período Fundamental T1 (s)": round(T_1_val, 3),
                            "Masa Total Activa": round(masa_total_val, 3),
                            "Método": "Fuerza Horizontal Equivalente (FHE)"
                        }])
                        st.dataframe(df_resumen_periodo, hide_index=True, use_container_width=True)
                    else:
                        st.write("**2. Fuerzas Nodales Resultantes (SRSS / CQC)**")
                        st.dataframe(df_fuerzas_nodales_sismo, hide_index=True, use_container_width=True)

                if tipo_analisis_sismico == "Fuerza Horizontal Equivalente (FHE)":
                    st.markdown("---")
                    st.write("**3. Resumen Sísmico de Parámetros (NSR-10)**")
                    df_resumen_fhe_params = pd.DataFrame([{
                        "Zona Sísmica Aa": Aa, "Av": Av, "Fa": Fa, "Fv": Fv, "Importancia (I)": I_imp,
                        "Coef. Reducción (R)": R_factor, "Período T1 (s)": round(T_1, 3), "Sa (g)": round(Sa_d * R_factor, 3), f"Cortante Basal V ({u['F']})": round(V_s, 3)
                    }])
                    st.dataframe(df_resumen_fhe_params, hide_index=True, use_container_width=True)

                    st.markdown("---")
                    st.write(f"**4. Distribución de Fuerza Horizontal (FHE) - T1 = {T_1:.3f} s**")
                    if df_fhe is not None:
                        df_fhe_copia = df_fhe.copy()
                        df_fhe_copia["Cortante Acumulado"] = df_fhe_copia["Fuerza Sísmica (F)"].iloc[::-1].cumsum().iloc[::-1].round(3)
                        st.dataframe(df_fhe_copia, hide_index=True, use_container_width=True)

                    st.markdown("---")
                    st.write("**5. Verificación de Derivas y Desplazamientos por Piso**")
                    st.dataframe(df_derivas, hide_index=True, use_container_width=True)

                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        fig_etabs_drift = go.Figure()
                        fig_etabs_drift.add_trace(go.Scatter(
                            x=df_derivas["Deriva de Diseño (%)"], y=df_derivas["Nivel (Y)"],
                            mode="lines+markers", name="Deriva (%)",
                            line=dict(color="#FF5733", width=3), marker=dict(size=8, color="#FF5733")
                        ))
                        fig_etabs_drift.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Límite 1.0%")
                        fig_etabs_drift.update_layout(
                            title="Deriva de Piso por Nivel", xaxis_title="Deriva (%)", yaxis_title=f"Elevación Nivel ({u['L']})",
                            template="plotly_white", height=400, margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_etabs_drift, use_container_width=True)
                    with col_g2:
                        fig_etabs_disp = go.Figure()
                        fig_etabs_disp.add_trace(go.Scatter(
                            x=df_derivas["Desplazamiento Elástico"], y=df_derivas["Nivel (Y)"],
                            mode="lines+markers", name="Desplazamiento",
                            line=dict(color="#005B64", width=3), marker=dict(size=8, color="#005B64")
                        ))
                        fig_etabs_disp.update_layout(
                            title="Desplazamiento por Nivel", xaxis_title=f"Desplazamiento ({u['L']})", yaxis_title=f"Elevación Nivel ({u['L']})",
                            template="plotly_white", height=400, margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_etabs_disp, use_container_width=True)

                else:
                    st.markdown("---")
                    st.write("**3. Participación Modal y Factores de Participación (Γ)**")
                    if df_sismo_modos is not None:
                        df_modos_detalle = df_sismo_modos[["Modo", "Periodo (s)", "Masa Partic. (%)", "Masa Acum. (%)", "Γ (Participación)"]]
                        st.dataframe(df_modos_detalle, hide_index=True, use_container_width=True)

                    st.markdown("---")
                    st.write("**4. Verificación de Derivas de Diseño (Regla SRSS y R)**")
                    st.dataframe(df_derivas, hide_index=True, use_container_width=True)

                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        fig_etabs_drift = go.Figure()
                        fig_etabs_drift.add_trace(go.Scatter(
                            x=df_derivas["Deriva de Diseño (%)"], y=df_derivas["Nivel (Y)"],
                            mode="lines+markers", name="Deriva (%)",
                            line=dict(color="#FF5733", width=3), marker=dict(size=8, color="#FF5733")
                        ))
                        fig_etabs_drift.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Límite 1.0%")
                        fig_etabs_drift.update_layout(
                            title="Deriva de Piso por Nivel", xaxis_title="Deriva (%)", yaxis_title=f"Elevación Nivel ({u['L']})",
                            template="plotly_white", height=400, margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_etabs_drift, use_container_width=True)
                    with col_g2:
                        fig_etabs_disp = go.Figure()
                        fig_etabs_disp.add_trace(go.Scatter(
                            x=df_derivas["Desplazamiento Elástico"], y=df_derivas["Nivel (Y)"],
                            mode="lines+markers", name="Desplazamiento",
                            line=dict(color="#005B64", width=3), marker=dict(size=8, color="#005B64")
                        ))
                        fig_etabs_disp.update_layout(
                            title="Desplazamiento por Nivel", xaxis_title=f"Desplazamiento ({u['L']})", yaxis_title=f"Elevación Nivel ({u['L']})",
                            template="plotly_white", height=400, margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_etabs_disp, use_container_width=True)

                    st.markdown("---")
                    Tc_graf = 0.48 * Av * Fv / (Aa * Fa) if (Aa * Fa) > 0 else 0.01
                    T0_graf, Tl_graf = 0.1 * Tc_graf, 2.4 * Fv
                    T_max = max(max(df_sismo_modos["Periodo (s)"]) * 1.2, Tl_graf * 1.2)
                    T_vals = np.linspace(0, T_max, 200)
                    Sa_vals = []
                    for t in T_vals:
                        if t < T0_graf: Sa_vals.append((Aa * Fa * I_imp * (1.0 + 1.5 * t / T0_graf)) / R_factor)
                        elif t <= Tc_graf: Sa_vals.append((2.5 * Aa * Fa * I_imp) / R_factor)
                        elif t <= Tl_graf: Sa_vals.append((1.2 * Av * Fv * I_imp / t) / R_factor)
                        else: Sa_vals.append((1.2 * Av * Fv * Tl_graf * I_imp / (t**2)) / R_factor)

                    fig_sp = go.Figure()
                    fig_sp.add_trace(go.Scatter(x=T_vals, y=Sa_vals, mode="lines", name="Espectro de Diseño (Sa / R)", line=dict(color="#DE0000", width=2)))
                    fig_sp.add_trace(go.Scatter(
                        x=df_sismo_modos["Periodo (s)"], y=df_sismo_modos["Sa de Diseño (g)"], mode="markers+text", name="Modos",
                        text=df_sismo_modos["Modo"].apply(lambda x: f"M{x}"), textposition="top center",
                        marker=dict(size=10, color="#005B64", line=dict(width=2, color="white")),
                    ))
                    fig_sp.update_layout(
                        title="Espectro de Diseño Reducido (NSR-10) y Periodos Modales", xaxis_title="Periodo T (s)",
                        yaxis_title="Pseudo-Aceleración de Diseño Sa / R (g)", template="plotly_white", plot_bgcolor="#F0F2F6",
                    )
                    st.plotly_chart(fig_sp, use_container_width=True)

except Exception as e:
    st.error(f"❌ Revisa los datos de entrada para continuar con el cálculo. Detalle: {e}")
