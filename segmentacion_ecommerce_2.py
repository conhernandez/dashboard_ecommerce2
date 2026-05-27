import streamlit as st
import numpy as np
import pandas as pd
import warnings
import os
import plotly.express as px
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Segmentación E-commerce | UdeC",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.hero-box {
    background: linear-gradient(135deg, #0a2342 0%, #1a3a5c 60%, #0e4f7a 100%);
    border-radius: 16px;
    padding: 48px 40px;
    color: white;
    margin-bottom: 32px;
    border-left: 6px solid #f5a623;
}
.hero-box h1 { font-size: 2.4rem; font-weight: 700; margin: 0 0 8px 0; letter-spacing: -0.5px; }
.hero-box p  { font-size: 1.05rem; opacity: 0.85; margin: 0; }
.hero-badge {
    display: inline-block;
    background: #f5a623;
    color: #0a2342;
    font-weight: 700;
    font-size: 0.78rem;
    border-radius: 20px;
    padding: 4px 14px;
    margin-bottom: 18px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.section-header {
    border-left: 5px solid #f5a623;
    padding-left: 16px;
    margin: 28px 0 20px 0;
}
.section-header h2 { margin: 0; font-weight: 700; font-size: 1.6rem; color: var(--text-color); }
.section-header p  { margin: 4px 0 0 0; color: var(--text-color); opacity: 0.8; font-size: 0.95rem; }

.kpi-card {
    background: var(--background-color);
    border: 1px solid var(--secondary-background-color);
    border-radius: 12px;
    padding: 20px 18px;
    text-align: center;
    border-top: 4px solid var(--primary-color);
}
.kpi-card .kpi-value { font-size: 2rem; font-weight: 700; color: var(--text-color); }
.kpi-card .kpi-label { font-size: 0.85rem; color: var(--text-color); opacity: 0.8; margin-top: 4px; }

.segment-card {
    border-radius: 12px;
    padding: 18px 16px;
    margin-bottom: 12px;
    border-left: 5px solid;
    background: var(--background-color);
    border: 1px solid var(--secondary-background-color);
    color: var(--text-color);
}
.info-box {
    background: var(--secondary-background-color);
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #1a6fb5;
    margin: 16px 0;
    font-size: 0.93rem;
    color: var(--text-color);
}
.warn-box {
    background: var(--secondary-background-color);
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #f5a623;
    margin: 16px 0;
    font-size: 0.93rem;
    color: var(--text-color);
}

.sidebar-title {
    font-weight: 700;
    font-size: 1.1rem;
    color: var(--text-color);
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CARGA DE DATOS PRECALCULADOS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_files_mtime(prefix):
    paths = [
        os.path.join(BASE_DIR, "data", f"clientes_segmentados_{prefix}.csv"),
        os.path.join(BASE_DIR, "data", f"ordenes_segmentadas_{prefix}.csv"),
        os.path.join(BASE_DIR, "data", f"metricas_gmm_{prefix}.csv"),
        os.path.join(BASE_DIR, "data", f"metricas_kp_{prefix}.csv")
    ]
    mtimes = []
    for p in paths:
        if os.path.exists(p):
            mtimes.append(os.path.getmtime(p))
    return tuple(mtimes)

@st.cache_data(show_spinner="Cargando datos precalculados…")
def cargar_datos_precalculados(prefix="principal", mtime=None):
    # Cargar CSVs precalculados
    df_fusion_cat = pd.read_csv(os.path.join(BASE_DIR, "data", f"clientes_segmentados_{prefix}.csv"), index_col="customer_id")
    df_ord_seg = pd.read_csv(os.path.join(BASE_DIR, "data", f"ordenes_segmentadas_{prefix}.csv"))
    met_gmm = pd.read_csv(os.path.join(BASE_DIR, "data", f"metricas_gmm_{prefix}.csv"))
    met_kp = pd.read_csv(os.path.join(BASE_DIR, "data", f"metricas_kp_{prefix}.csv"))
    return df_fusion_cat, df_ord_seg, met_gmm, met_kp

# Intentar cargar datos primarios
try:
    mtime_principal = get_files_mtime("principal")
    df_fusion_cat, df_ord_seg, met_gmm, met_kp = cargar_datos_precalculados("principal", mtime_principal)
except Exception as e:
    st.error(f"Error al cargar archivos precalculados: {e}. Por favor ejecuta 'precalcular_segmentacion.py' primero.")
    st.stop()

# Mapear nombres y colores
CLUSTER_COLORS_RFM = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
CLUSTER_COLORS_DEM = ["#9b59b6", "#1abc9c", "#e67e22"]

# ─────────────────────────────────────────────
# SIDEBAR — NAVEGACIÓN
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">🛒 Segmentación E-commerce</p>', unsafe_allow_html=True)
    st.caption("Universidad de Concepción · Marketing 2026-1")
    st.divider()
    seccion = st.sidebar.radio(
        "Sección",
        options=[
            "🏠  Contexto del Mercado",
            "📋  Descripción de Datos",
            "🧹  Limpieza de Datos",
            "📈  Modelo RFM — GMM",
            "👥  Modelo Demográfico — K-Prototypes",
            "🔗  Análisis de Fusión",
            "🎯  Conclusiones y Posicionamiento",
            "📁  Anexo: Análisis Avanzado y Cohortes"
        ],
        label_visibility="collapsed",
    )
    st.divider()
    
    # Créditos de Alumnos y Profesor
    with st.expander("👥 Integrantes & Profesor", expanded=True):
        st.markdown("""
        <div style="font-size: 0.82rem; line-height: 1.45; color: var(--text-color);">
          <b>Integrantes (Grupo 31):</b>
          <ul style="margin-top: 4px; margin-bottom: 8px; padding-left: 16px;">
            <li>Ricardo Raúl Ernesto Barra Osorio</li>
            <li>Matías Patricio Caamaño Rivas</li>
            <li>Gabriel Ignacio Fernández Iglesias</li>
            <li>María José Galvis Cabrera</li>
            <li>Constanza Ignacia Hernández Cárdenas</li>
            <li>Rubén Emilio Lamilla Soto</li>
          </ul>
          <b>Profesor:</b>
          <ul style="margin-top: 4px; margin-bottom: 2px; padding-left: 16px;">
            <li>Juan Carlos Caro Seguel</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    st.caption("Trabajo 2 · Mercado 1: E-commerce global")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def section_header(title, subtitle=""):
    st.markdown(
        f"""<div class="section-header">
            <h2>{title}</h2>
            {"<p>" + subtitle + "</p>" if subtitle else ""}
        </div>""",
        unsafe_allow_html=True,
    )



# ══════════════════════════════════════════════
# SECCIÓN 1 — CONTEXTO DEL MERCADO
# ══════════════════════════════════════════════
if seccion == "🏠  Contexto del Mercado":
    st.markdown("""
    <div class="hero-box">
      <div class="hero-badge">Trabajo 2 · Marketing 2026-1 · UdeC</div>
      <h1>🛒 Segmentación de Clientes<br>E-commerce Global</h1>
      <p>Mercado 1 · Modelos GMM (RFM) + K-Prototypes (Demográfico) · Fusion Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        section_header("Contexto del Mercado", "¿Qué problema buscamos resolver?")
        st.markdown("""
        El dataset contiene registros transaccionales de un **E-commerce global** con miles de clientes
        distribuidos en múltiples países y categorías de producto.

        **El cliente** (la empresa que desea entrar al mercado) busca:
        - Identificar segmentos **más receptivos a promociones y descuentos**
        - Considerar la **categoría de producto preferida** como eje de personalización
        - Descubrir otros mercados meta relevantes más allá de la sensibilidad al precio
        """)
        st.markdown("""
        <div class="info-box">
        <b>🎯 Objetivo estratégico:</b> Desde la posición de una empresa entrante al mercado e-commerce,
        definir segmentos prioritarios y diseñar una estrategia de posicionamiento diferenciada para
        capturarlos eficientemente.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        section_header("KPIs del Dataset (Principal)")
        total_clientes = len(df_fusion_cat)
        total_ordenes  = len(df_ord_seg)
        n_paises       = df_fusion_cat["country"].nunique() if "country" in df_fusion_cat.columns else "N/A"
        n_categorias   = df_ord_seg["category"].nunique()
        gasto_prom     = df_fusion_cat["avg_order_value_usd"].mean()

        for label, val in [
            ("Clientes registrados", f"{total_clientes:,}"),
            ("Órdenes totales",      f"{total_ordenes:,}"),
            ("Países representados", f"{n_paises}"),
            ("Categorías de producto", f"{n_categorias}"),
            ("Gasto promedio USD",   f"${gasto_prom:,.2f}"),
        ]:
            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom:10px">
              <div class="kpi-value">{val}</div>
              <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    section_header("Metodología", "Flujo del análisis")
    cols = st.columns(5)
    pasos = [
        ("1️⃣", "Carga y\nexploración",    "Revisar estructura, tipos y calidad de datos"),
        ("2️⃣", "Limpieza y\nwinsorización", "Filtros lógicos + tratamiento de outliers (1%–98%)"),
        ("3️⃣", "GMM\n(RFM)",              "Segmentación transaccional (k=4 clusters)"),
        ("4️⃣", "K-Prototypes\n(Demo.)",   "Segmentación demográfica mixta (k=3 clusters)"),
        ("5️⃣", "Fusión\n& estrategia",    "Cruce de modelos + recomendaciones de posicionamiento"),
    ]
    for col, (num, titulo, desc) in zip(cols, pasos):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div style="font-size:2rem">{num}</div>
              <div style="font-weight:700;font-size:0.9rem;margin:8px 0;white-space:pre-line">{titulo}</div>
              <div style="font-size:0.78rem;color:#666">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SECCIÓN 2 — DESCRIPCIÓN DE DATOS
# ══════════════════════════════════════════════
elif seccion == "📋  Descripción de Datos":
    section_header("Descripción de Datos", "Exploración inicial de los datos de Clientes y Órdenes")

    tab1, tab2 = st.tabs(["👤 Customers", "📦 Orders"])

    with tab1:
        st.markdown("### Vista previa de Clientes Segmentados")
        st.dataframe(df_fusion_cat.head(8), use_container_width=True)

        st.markdown("### Estadísticas descriptivas — Variables numéricas")
        st.dataframe(
            df_fusion_cat.select_dtypes(include="number").describe().T.round(2),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribución de Membership Tier**")
            tier_counts = df_fusion_cat["membership_tier"].value_counts().reset_index()
            tier_counts.columns = ["Membership Tier", "N° Clientes"]
            fig = px.bar(tier_counts, x="Membership Tier", y="N° Clientes", color="Membership Tier",
                         color_discrete_sequence=["#0a2342", "#1a6fb5", "#f5a623", "#e74c3c"],
                         title="Membership Tier")
            fig.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Distribución de Género**")
            gender_counts = df_fusion_cat["gender"].value_counts().reset_index()
            gender_counts.columns = ["Género", "N° Clientes"]
            fig = px.bar(gender_counts, x="Género", y="N° Clientes", color="Género",
                         color_discrete_sequence=["#9b59b6", "#3498db", "#95a5a6"],
                         title="Género")
            fig.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Vista previa de Órdenes transaccionales")
        st.dataframe(df_ord_seg.head(8), use_container_width=True)

        st.markdown("### Estadísticas descriptivas transaccionales")
        st.dataframe(
            df_ord_seg.select_dtypes(include="number").describe().T.round(2),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ventas por Categoría**")
            cat_ventas = df_ord_seg.groupby("category")["total_amount_usd"].sum().sort_values(ascending=False).reset_index()
            cat_ventas.columns = ["Categoría", "Total USD"]
            fig = px.bar(cat_ventas, x="Categoría", y="Total USD", color="Categoría",
                         color_discrete_sequence=px.colors.qualitative.T10,
                         title="Revenue por Categoría")
            fig.update_layout(showlegend=False, height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Distribución de Descuentos (%)**")
            disc_counts = df_ord_seg["discount_pct"].value_counts().sort_index().reset_index()
            disc_counts.columns = ["Descuento (%)", "N° órdenes"]
            fig = px.bar(disc_counts, x="Descuento (%)", y="N° órdenes",
                         color_discrete_sequence=["#f5a623"],
                         title="Frecuencia de Descuentos")
            fig.update_layout(showlegend=False, height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# SECCIÓN 3 — LIMPIEZA DE DATOS
# ══════════════════════════════════════════════
elif seccion == "🧹  Limpieza de Datos":
    section_header("Limpieza de Datos", "Filtros lógicos y tratamiento de outliers (Winsorización)")

    st.markdown("""
    <div class="info-box">
    El dataset original contiene valores extremos y desalineaciones de ID. Se aplicaron filtros de consistencia 
    y <b>winsorización (percentiles 1% al 98%)</b> para garantizar clústeres robustos y estables.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ✂️ Winsorización Aplicada")
        st.markdown("""
        Para mitigar el ruido de valores atípicos y estabilizar las varianzas de GMM y K-Prototypes:
        - `total_orders` (Frecuencia)
        - `avg_order_value_usd` (Monetario)
        - `days_since_last_purchase` (Recencia)
        """)

        st.markdown("#### 📊 Estadísticas RFM — Dataset Post-Winsorización")
        st.dataframe(df_fusion_cat[["days_since_last_purchase", "total_orders", "avg_order_value_usd"]].describe().T.round(2), use_container_width=True)

    with col2:
        st.markdown(f"#### 📏 Volumen de Clientes Resultantes")
        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom:16px">
          <div class="kpi-value">{df_fusion_cat.shape[0]:,}</div>
          <div class="kpi-label">Clientes en base final</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-value">Consistencia</div>
          <div class="kpi-label">Todos los clientes tienen registros transaccionales válidos y no nulos.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 📦 Boxplots RFM — Control de Outliers Winsorizados")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig_freq = px.box(df_fusion_cat, y="total_orders", color_discrete_sequence=["#3498db"],
                          title="Frecuencia (total_orders)", labels={"total_orders": "Órdenes"})
        fig_freq.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_freq, use_container_width=True)
    with c2:
        fig_mon = px.box(df_fusion_cat, y="avg_order_value_usd", color_discrete_sequence=["#2ecc71"],
                         title="Monetario (avg_order_value_usd)", labels={"avg_order_value_usd": "USD"})
        fig_mon.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_mon, use_container_width=True)
    with c3:
        fig_rec = px.box(df_fusion_cat, y="days_since_last_purchase", color_discrete_sequence=["#e74c3c"],
                         title="Recencia (days_since_last_purchase)", labels={"days_since_last_purchase": "Días"})
        fig_rec.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_rec, use_container_width=True)


# ══════════════════════════════════════════════
# SECCIÓN 4 — GMM (RFM)
# ══════════════════════════════════════════════
elif seccion == "📈  Modelo RFM — GMM":
    section_header("Modelo RFM — Gaussian Mixture Model (GMM)", "Segmentación probabilística transaccional")

    tab_sel, tab_modelo, tab_viz = st.tabs(
        ["📊 Selección de k (Cálculo Dinámico)", "📋 Perfiles de Clusters", "🌐 Visualización 3D"]
    )

    with tab_sel:
        st.markdown("#### Métricas de Selección del Número Óptimo de Clústeres (k = 2 a 10)")
        
        # Fila 1: BIC vs AIC y Silueta
        c1, c2 = st.columns(2)
        with c1:
            fig_bic = px.line(met_gmm, x="k", y=["BIC", "AIC"], markers=True,
                              color_discrete_sequence=["#d95f02", "#2ca02c"],
                              title="Criterios de Información (AIC vs BIC - Menor es mejor)")
            fig_bic.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="k=4 óptimo")
            fig_bic.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bic, use_container_width=True)
        with c2:
            fig_sil = px.line(met_gmm, x="k", y="Silueta", markers=True,
                              color_discrete_sequence=["#7570b3"],
                              title="Silueta (Más cerca a 1 es mejor)")
            fig_sil.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="k=4 óptimo")
            fig_sil.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_sil, use_container_width=True)
            
        # Fila 2: DBI y CHI
        c3, c4 = st.columns(2)
        with c3:
            fig_dbi = px.line(met_gmm, x="k", y="DBI", markers=True,
                              color_discrete_sequence=["#d95f02"],
                              title="Davies-Bouldin (Más cerca a 0 es mejor)")
            fig_dbi.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="k=4 óptimo")
            fig_dbi.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_dbi, use_container_width=True)
        with c4:
            fig_chi = px.line(met_gmm, x="k", y="CHI", markers=True,
                              color_discrete_sequence=["#7570b3"],
                              title="Calinski-Harabasz (Más alto es mejor)")
            fig_chi.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="k=4 óptimo")
            fig_chi.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_chi, use_container_width=True)

        # Fila 3: Entropía y LogLik
        c5, c6 = st.columns(2)
        with c5:
            fig_ent = px.line(met_gmm, x="k", y="Entropia", markers=True,
                              color_discrete_sequence=["#d95f02"],
                              title="Entropía Media (Cercana a 1 indica perfiles bien separados)")
            fig_ent.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="k=4 óptimo")
            fig_ent.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_ent, use_container_width=True)
        with c6:
            fig_lik = px.line(met_gmm, x="k", y="LogLik", markers=True,
                              color_discrete_sequence=["#7570b3"],
                              title="Log-Likelihood Nativo (Más alto indica mejor ajuste)")
            fig_lik.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="k=4 óptimo")
            fig_lik.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_lik, use_container_width=True)

        st.markdown("""
        <div class="warn-box">
        <b>🏆 k = 4 Seleccionado:</b> La combinación de BIC/AIC e índices de separación (Silueta y Entropía) indican 
        que 4 clústeres ofrecen el balance perfecto de interpretabilidad y rigor estadístico.
        </div>
        """, unsafe_allow_html=True)

    with tab_modelo:
        st.markdown("#### Perfiles Promedio por Cluster RFM (k=4)")
        # Caracterización promedio
        df_perf_gmm = df_fusion_cat.groupby("Nombre_RFM")[["total_orders", "avg_order_value_usd", "days_since_last_purchase"]].mean()
        df_perf_gmm["N clientes"] = df_fusion_cat["Nombre_RFM"].value_counts()
        st.dataframe(df_perf_gmm.rename(columns={
            "total_orders": "Frecuencia (Órdenes)",
            "avg_order_value_usd": "Monetario (Avg USD)",
            "days_since_last_purchase": "Recencia (Días)",
        }).round(2), use_container_width=True)

        st.markdown("#### Distribución de Clientes")
        sizes = df_fusion_cat["Nombre_RFM"].value_counts().reset_index()
        sizes.columns = ["Segmento RFM", "N° Clientes"]
        fig = px.bar(sizes, x="Segmento RFM", y="N° Clientes", color="Segmento RFM",
                     color_discrete_sequence=CLUSTER_COLORS_RFM[:4],
                     title="Tamaño de Clusters RFM GMM")
        fig.update_layout(showlegend=False, height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab_viz:
        st.markdown("#### Visualización 3D Interactiva — Espacio RFM")
        fig_3d = px.scatter_3d(
            df_fusion_cat.reset_index(), x="days_since_last_purchase", y="total_orders", z="avg_order_value_usd",
            color="Nombre_RFM",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Segmentación RFM — GMM (k=4)",
            opacity=0.6, height=650,
            labels={
                "days_since_last_purchase": "Recencia (Días)",
                "total_orders": "Frecuencia (Compras)",
                "avg_order_value_usd": "Monetario (USD)",
                "Nombre_RFM": "Segmento RFM"
            }
        )
        fig_3d.update_traces(marker=dict(size=4))
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=50))
        st.plotly_chart(fig_3d, use_container_width=True)


# ══════════════════════════════════════════════
# SECCIÓN 5 — K-PROTOTYPES (DEMOGRÁFICO)
# ══════════════════════════════════════════════
elif seccion == "👥  Modelo Demográfico — K-Prototypes":
    section_header("Modelo Demográfico — K-Prototypes", "Segmentación con datos demográficos mixtos")

    tab_sel, tab_modelo, tab_viz = st.tabs(
        ["📊 Selección de k (Cálculo Dinámico)", "📋 Perfiles de Clusters", "🌐 Visualización 3D"]
    )

    with tab_sel:
        st.markdown("#### Métricas de Selección del Número Óptimo de Clústeres (k = 2 a 10)")
        
        # Fila 1: Costo (Codo) y Silueta
        c1, c2 = st.columns(2)
        with c1:
            fig_cost = px.line(met_kp, x="k", y="Costo", markers=True,
                               color_discrete_sequence=["#e67e22"],
                               title="Costo / Disimilitud Total (Método del Codo - Menor es mejor)")
            fig_cost.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="k=3 seleccionado")
            fig_cost.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_cost, use_container_width=True)
        with c2:
            fig_sil = px.line(met_kp, x="k", y="Silueta", markers=True,
                              color_discrete_sequence=["#7570b3"],
                              title="Coeficiente de Silueta (Más cerca a 1 es mejor)")
            fig_sil.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="k=3 seleccionado")
            fig_sil.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_sil, use_container_width=True)
            
        # Fila 2: DBI y CHI
        c3, c4 = st.columns(2)
        with c3:
            fig_dbi = px.line(met_kp, x="k", y="DBI", markers=True,
                              color_discrete_sequence=["#e67e22"],
                              title="Índice Davies-Bouldin (Más cerca a 0 es mejor)")
            fig_dbi.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="k=3 seleccionado")
            fig_dbi.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_dbi, use_container_width=True)
        with c4:
            fig_chi = px.line(met_kp, x="k", y="CHI", markers=True,
                              color_discrete_sequence=["#7570b3"],
                              title="Índice Calinski-Harabasz (Mayor es mejor)")
            fig_chi.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="k=3 seleccionado")
            fig_chi.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_chi, use_container_width=True)

    with tab_modelo:
        st.markdown("#### Perfiles por Cluster Demográfico (k=3)")
        df_perf_kp = df_fusion_cat.groupby("Nombre_DEM").agg(
            Edad_Promedio=("age", "mean"),
            Genero_Frecuente=("gender", lambda x: x.mode()[0] if not x.mode().empty else np.nan),
            Membresia_Frecuente=("membership_tier", lambda x: x.mode()[0] if not x.mode().empty else np.nan),
            N_Clientes=("age", "count"),
        ).round(2)
        st.dataframe(df_perf_kp, use_container_width=True)

    with tab_viz:
        st.markdown("#### Visualización 3D — Vista de Nubes con Jitter Demográfico")
        df_plot_kp = df_fusion_cat.copy()
        
        # Mapear a códigos numéricos para jitter
        dic_gender = {"Female": 0, "Male": 1, "Other": 2}
        dic_tier   = {"Free": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
        
        df_plot_kp["gender_n"] = df_plot_kp["gender"].map(dic_gender)
        df_plot_kp["membership_n"] = df_plot_kp["membership_tier"].map(dic_tier)
        
        np.random.seed(42)
        df_plot_kp["gender_j"] = df_plot_kp["gender_n"] + np.random.uniform(-0.15, 0.15, len(df_plot_kp))
        df_plot_kp["membership_j"] = df_plot_kp["membership_n"] + np.random.uniform(-0.15, 0.15, len(df_plot_kp))

        fig_3d_kp = px.scatter_3d(
            df_plot_kp.reset_index(), x="age", y="gender_j", z="membership_j",
            color="Nombre_DEM",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Segmentación Demográfica — K-Prototypes (k=3)",
            opacity=0.6, height=650,
            labels={
                "age": "Edad (Años)",
                "gender_j": "Género (con Jitter)",
                "membership_j": "Membresía (con Jitter)",
                "Nombre_DEM": "Segmento Demográfico"
            }
        )
        fig_3d_kp.update_traces(marker=dict(size=4))
        fig_3d_kp.update_layout(
            margin=dict(l=0, r=0, b=0, t=50),
            scene=dict(
                yaxis=dict(tickvals=[0, 1, 2], ticktext=["Female", "Male", "Other"]),
                zaxis=dict(tickvals=[0, 1, 2, 3], ticktext=["Free", "Silver", "Gold", "Platinum"])
            )
        )
        st.plotly_chart(fig_3d_kp, use_container_width=True)


# ══════════════════════════════════════════════
# SECCIÓN 6 — ANÁLISIS DE FUSIÓN
# ══════════════════════════════════════════════
elif seccion == "🔗  Análisis de Fusión":
    section_header("Análisis de Fusión", "Cruce relacional RFM (GMM) × Demográfico (K-Prototypes)")

    tab_mat, tab_heat, tab_burbuja, tab_tenure, tab_compra_mes, tab_temporal, tab_descuento, tab_categoria = st.tabs([
        "📊 F1: Matriz de Fusión",
        "🌡️ F2: Variables Clave",
        "🫧 F4: Mapa de Burbujas",
        "📅 F11: Antigüedad",
        "⚡ F12: Velocidad de Compra",
        "📈 F13: Evolución Temporal",
        "💰 F9: Descuentos",
        "🛍️ F10: Preferencia Categorías",
    ])

    with tab_mat:
        st.markdown("#### F1: Matriz de Distribución por Micro-Segmento")
        tabla = pd.crosstab(df_fusion_cat["Nombre_RFM"], df_fusion_cat["Nombre_DEM"])
        tabla_pct = (tabla / tabla.sum().sum()) * 100

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.imshow(tabla, text_auto="d", color_continuous_scale="YlOrRd", title="F1a: Clientes por Micro-Segmento (Absoluto)",
                             labels=dict(x="Segmento Demográfico", y="Segmento RFM"))
            fig1.update_layout(height=400, coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.imshow(tabla_pct, text_auto=".1f", color_continuous_scale="YlOrRd", title="F1b: % de Clientes por Micro-Segmento",
                             labels=dict(x="Segmento Demográfico", y="Segmento RFM"))
            fig2.update_layout(height=400, coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)

    with tab_heat:
        st.markdown("#### F2: Perfilamiento de Variables por Micro-Segmento")
        
        # 1. Variables de comportamiento
        st.markdown("##### A. Variables de Negocio y Churn")
        vars_heat = [
            ("discount_pct_prom",     "% Descuento Promedio",        "RdYlGn"),
            ("session_duration_prom", "Duración Sesión (min)",        "Blues"),
            ("pages_viewed_prom",     "Páginas Vistas",               "Purples"),
            ("age",                   "Edad Media",                   "Oranges"),
            ("returns_made",          "Devoluciones",                 "Reds"),
            ("rating_prom",           "Rating Promedio",              "RdYlGn"),
            ("is_repeat_pct",         "% Clientes Recurrentes",       "Greens"),
            ("churned",               "% Churn",                      "RdYlGn_r"),
        ]
        
        c1, c2, c3, c4 = st.columns(4)
        cols_row1 = [c1, c2, c3, c4]
        for idx, (col_name, label, cmap) in enumerate(vars_heat[:4]):
            with cols_row1[idx]:
                pivot = df_fusion_cat.groupby(["Nombre_RFM", "Nombre_DEM"])[col_name].mean().unstack()
                fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale=cmap, title=label,
                                labels=dict(x="Demografía", y="RFM"))
                fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
        c5, c6, c7, c8 = st.columns(4)
        cols_row2 = [c5, c6, c7, c8]
        for idx, (col_name, label, cmap) in enumerate(vars_heat[4:]):
            with cols_row2[idx]:
                pivot = df_fusion_cat.groupby(["Nombre_RFM", "Nombre_DEM"])[col_name].mean().unstack()
                fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale=cmap, title=label,
                                labels=dict(x="Demografía", y="RFM"))
                fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)

        # 2. F2_RFM: Variables métricas originales
        st.markdown("##### B. Variables RFM Originales")
        vars_rfm = [
            ("days_since_last_purchase", "Recencia Promedio (Días inactivo)", "Reds_r"),
            ("total_orders",             "Frecuencia Promedio (N° Órdenes)",   "Blues"),
            ("avg_order_value_usd",      "Monetario Promedio (Gasto USD)",     "Greens"),
        ]
        
        cols_rfm_grid = st.columns(3)
        for idx, (col_name, label, cmap) in enumerate(vars_rfm):
            with cols_rfm_grid[idx]:
                pivot = df_fusion_cat.groupby(["Nombre_RFM", "Nombre_DEM"])[col_name].mean().unstack()
                fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale=cmap, title=label,
                                labels=dict(x="Demografía", y="RFM"))
                fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)

    with tab_burbuja:
        st.markdown("#### F4: Mapa de Burbujas de Micro-Segmentos")
        burbuja = df_fusion_cat.groupby("Micro_Segmento").agg(
            descuento  =("discount_pct_prom", "mean"),
            frecuencia =("n_ordenes",          "mean"),
            n_clientes =("age",                "count"),
        ).reset_index()

        fig = px.scatter(burbuja, x="descuento", y="frecuencia", size="n_clientes", color="Micro_Segmento",
                         hover_name="Micro_Segmento",
                         labels={"descuento": "Descuento Promedio (%)", "frecuencia": "Frecuencia (Órdenes)", "Micro_Segmento": "Micro-Segmento"},
                         title="F4: Frecuencia vs. Descuento por Micro-Segmento (Tamaño burbuja = N° clientes)")
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab_tenure:
        st.markdown("#### F11: Antigüedad del Cliente desde el Registro por Micro-Segmento")
        
        # Calcular antigüedad (referencia = registro max + 1 día)
        df_temp = df_fusion_cat.copy()
        df_temp["reg_date_dt"] = pd.to_datetime(df_temp["registration_date"]) if "registration_date" in df_temp.columns else pd.Timestamp("2026-01-01")
        ref_date = df_temp["reg_date_dt"].max() + pd.Timedelta(days=1)
        df_temp["antiguedad_dias"] = (ref_date - df_temp["reg_date_dt"]).dt.days
        df_temp["antiguedad_anos"] = df_temp["antiguedad_dias"] / 365.0

        pivot_dias = df_temp.groupby(["Nombre_RFM", "Nombre_DEM"])["antiguedad_dias"].mean().unstack()
        pivot_anos = df_temp.groupby(["Nombre_RFM", "Nombre_DEM"])["antiguedad_anos"].mean().unstack()

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.imshow(pivot_dias, text_auto=".1f", color_continuous_scale="Oranges", title="F11a: Antigüedad Promedio (Días)",
                             labels=dict(x="Demografía", y="RFM"))
            fig1.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.imshow(pivot_anos, text_auto=".2f", color_continuous_scale="Oranges", title="F11b: Antigüedad Promedio (Años)",
                             labels=dict(x="Demografía", y="RFM"))
            fig2.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)

    with tab_compra_mes:
        st.markdown("#### F12: Velocidad de Compras Mensual por Micro-Segmento")
        
        df_temp = df_fusion_cat.copy()
        df_temp["reg_date_dt"] = pd.to_datetime(df_temp["registration_date"]) if "registration_date" in df_temp.columns else pd.Timestamp("2026-01-01")
        ref_date = df_temp["reg_date_dt"].max() + pd.Timedelta(days=1)
        
        df_temp["antiguedad_total_dias"] = (ref_date - df_temp["reg_date_dt"]).dt.days
        df_temp["tenure_months"] = np.maximum(df_temp["antiguedad_total_dias"] / 30.417, 1.0)
        
        df_temp["antiguedad_activa_dias"] = df_temp["antiguedad_total_dias"] - df_temp["days_since_last_purchase"]
        df_temp["antiguedad_activa_dias"] = np.maximum(df_temp["antiguedad_activa_dias"], 1.0)
        df_temp["tenure_active_months"] = np.maximum(df_temp["antiguedad_activa_dias"] / 30.417, 1.0)
        
        df_temp["compras_mes_historico"] = df_temp["total_orders"] / df_temp["tenure_months"]
        df_temp["compras_mes_activo"] = df_temp["total_orders"] / df_temp["tenure_active_months"]

        pivot_hist = df_temp.groupby(["Nombre_RFM", "Nombre_DEM"])["compras_mes_historico"].mean().unstack()
        pivot_act = df_temp.groupby(["Nombre_RFM", "Nombre_DEM"])["compras_mes_activo"].mean().unstack()

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.imshow(pivot_hist, text_auto=".2f", color_continuous_scale="YlGnBu", title="F12a: Velocidad de Compra Histórica (Órdenes/Mes)",
                             labels=dict(x="Demografía", y="RFM"))
            fig1.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.imshow(pivot_act, text_auto=".2f", color_continuous_scale="YlGnBu", title="F12b: Velocidad de Compra Activa (Órdenes/Mes)",
                             labels=dict(x="Demografía", y="RFM"))
            fig2.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)

    with tab_temporal:
        st.markdown("#### F13: Evolución Temporal Trimestral de Compras por Micro-Segmento")
        
        df_ord_cluster = df_ord_seg.copy()
        df_ord_cluster["order_date"] = pd.to_datetime(df_ord_cluster["order_date"])
        df_ord_cluster["quarter_dt"] = df_ord_cluster["order_date"].dt.to_period("Q").dt.to_timestamp()
        
        # Agrupar compras trimestrales por trimestre y micro-segmento
        df_ts_q = df_ord_cluster.groupby(["quarter_dt", "Micro_Segmento"]).size().reset_index(name="Compras")
        
        # Calcular baseline global promedio por segmento
        df_baseline = df_ord_cluster.groupby("quarter_dt").size().reset_index(name="Compras")
        df_baseline["Compras"] = df_baseline["Compras"] / 12.0
        df_baseline["Micro_Segmento"] = "Media Global (Promedio)"
        
        # Concatenar baseline y micro-segmentos
        df_combined = pd.concat([df_ts_q, df_baseline], ignore_index=True)
        
        fig = px.line(
            df_combined,
            x="quarter_dt",
            y="Compras",
            color="Micro_Segmento",
            title="Evolución Trimestral de Compras por Micro-Segmento",
            labels={"quarter_dt": "Trimestre", "Compras": "N° Compras", "Micro_Segmento": "Micro-Segmento"},
            color_discrete_sequence=px.colors.qualitative.Alphabet
        )
        
        # Configurar la línea de la Media Global como punteada y gris
        fig.for_each_trace(lambda t: t.update(
            line=dict(dash="dash", width=3, color="#7f8c8d") if t.name == "Media Global (Promedio)" else dict(width=2)
        ))
        
        fig.update_layout(
            height=600,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(title="Micro-Segmentos", font=dict(size=9))
        )
        st.plotly_chart(fig, use_container_width=True)


    with tab_descuento:
        st.markdown("#### F9: Análisis de Transacciones por Nivel de Descuento")
        
        df_ord_seg["discount_pct"] = df_ord_seg["discount_pct"].round(2)
        
        # F9a: Volumen Log
        tabla_desc = pd.crosstab(df_ord_seg["discount_pct"], df_ord_seg["Micro_Segmento"]).reset_index()
        df_desc_long = tabla_desc.melt(id_vars="discount_pct", var_name="Micro_Segmento", value_name="N° Compras")
        df_desc_long["discount_pct"] = df_desc_long["discount_pct"].astype(str) + "%"
        
        fig_vol = px.bar(
            df_desc_long,
            x="discount_pct",
            y="N° Compras",
            color="Micro_Segmento",
            title="F9a: Volumen de Compras por Descuento (Log Scale)",
            labels={"discount_pct": "Descuento (%)", "N° Compras": "N° Compras"},
            color_discrete_sequence=px.colors.qualitative.Alphabet,
        )
        fig_vol.update_yaxes(type="log")
        fig_vol.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(title="Micro-Segmentos", font=dict(size=8))
        )
        
        # F9b: 100% Normalizado
        tabla_desc_pct = pd.crosstab(df_ord_seg["discount_pct"], df_ord_seg["Micro_Segmento"])
        tabla_desc_pct = tabla_desc_pct.div(tabla_desc_pct.sum(axis=1), axis=0) * 100
        tabla_desc_pct = tabla_desc_pct.reset_index()
        df_desc_pct_long = tabla_desc_pct.melt(id_vars="discount_pct", var_name="Micro_Segmento", value_name="% Compras")
        df_desc_pct_long["discount_pct"] = df_desc_pct_long["discount_pct"].astype(str) + "%"
        
        fig_pct = px.bar(
            df_desc_pct_long,
            x="discount_pct",
            y="% Compras",
            color="Micro_Segmento",
            title="F9b: Distribución Porcentual Relativa (100% Normalizado)",
            labels={"discount_pct": "Descuento (%)", "% Compras": "% Compras"},
            color_discrete_sequence=px.colors.qualitative.Alphabet,
        )
        fig_pct.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(title="Micro-Segmentos", font=dict(size=8))
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_vol, use_container_width=True)
        with col2:
            st.plotly_chart(fig_pct, use_container_width=True)

    with tab_categoria:
        st.markdown("#### F10: Preferencia de Categorías por Micro-Segmento")
        
        # F10a: Volumen Absoluto
        tabla_cat = pd.crosstab(df_ord_seg["category"], df_ord_seg["Micro_Segmento"]).reset_index()
        df_cat_long = tabla_cat.melt(id_vars="category", var_name="Micro_Segmento", value_name="N° Compras")
        
        fig_vol = px.bar(
            df_cat_long,
            x="N° Compras",
            y="category",
            color="Micro_Segmento",
            orientation="h",
            title="F10a: Volumen Absoluto por Categoría",
            labels={"category": "Categoría", "N° Compras": "N° Compras"},
            color_discrete_sequence=px.colors.qualitative.Alphabet,
        )
        fig_vol.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(title="Micro-Segmentos", font=dict(size=8))
        )
        
        # F10b: 100% Normalizado
        tabla_cat_pct = pd.crosstab(df_ord_seg["category"], df_ord_seg["Micro_Segmento"])
        tabla_cat_pct = tabla_cat_pct.div(tabla_cat_pct.sum(axis=1), axis=0) * 100
        tabla_cat_pct = tabla_cat_pct.reset_index()
        df_cat_pct_long = tabla_cat_pct.melt(id_vars="category", var_name="Micro_Segmento", value_name="% de Compras")
        
        fig_pct = px.bar(
            df_cat_pct_long,
            x="% de Compras",
            y="category",
            color="Micro_Segmento",
            orientation="h",
            title="F10b: Distribución Porcentual (100%)",
            labels={"category": "Categoría", "% de Compras": "% de Compras"},
            color_discrete_sequence=px.colors.qualitative.Alphabet,
        )
        fig_pct.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(title="Micro-Segmentos", font=dict(size=8))
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_vol, use_container_width=True)
        with col2:
            st.plotly_chart(fig_pct, use_container_width=True)


# ══════════════════════════════════════════════
# SECCIÓN 7 — CONCLUSIONES Y POSICIONAMIENTO
# ══════════════════════════════════════════════
elif seccion == "🎯  Conclusiones y Posicionamiento":
    section_header("Conclusiones y Estrategia de Posicionamiento",
                   "Mercados meta relevantes y recomendaciones para la empresa entrante")

    st.markdown("""
    <div class="info-box">
    <b>Objetivo Estratégico:</b> Definir mercados meta clave basados en variables de consumo (frecuencia, 
    descuentos y categorías de productos favoritos) y diseñar el posicionamiento de entrada.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏆 Mercados Meta Seleccionados")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="segment-card" style="border-color:#f39c12;background:#fffbf0;color:#1a1a1a">
          <b style="color:#1a1a1a">🥇 Mercado Meta 1 — Clientes Históricos/Transaccionales × Hombres Más Jóvenes</b><br><br>
          <b style="color:#1a1a1a">¿Por qué?</b> Mayor frecuencia de compra + alta receptividad a descuentos.
          Son el segmento más activo y con mayor volumen transaccional.<br><br>
          <b style="color:#1a1a1a">Categorías preferidas:</b> Electronics, Sports & Outdoors, Clothing<br>
          <b style="color:#1a1a1a">Estrategia:</b> Programa de lealtad con descuentos escalonados, push notifications,
          experiencia mobile-first, early access a promociones flash.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="segment-card" style="border-color:#e74c3c;background:#fff5f5;color:#1a1a1a">
          <b style="color:#1a1a1a">🥈 Mercado Meta 2 — Clientes Premium × Hombres Mayor Edad</b><br><br>
          <b style="color:#1a1a1a">¿Por qué?</b> Mayor gasto promedio + membresía Gold/Platinum + menor sensibilidad
          al precio. Representan el mayor valor monetario por cliente.<br><br>
          <b style="color:#1a1a1a">Categorías preferidas:</b> Electronics, Home & Garden, Books<br>
          <b style="color:#1a1a1a">Estrategia:</b> Posicionamiento premium, envío express gratuito, servicio al cliente
          prioritario, recomendaciones personalizadas de alta gama.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="segment-card" style="border-color:#9b59b6;background:#faf5ff;color:#1a1a1a">
          <b style="color:#1a1a1a">🥉 Mercado Meta 3 — Clientes Promedio × Mujeres Poder Adq. Medio-Bajo</b><br><br>
          <b style="color:#1a1a1a">¿Por qué?</b> Gran volumen potencial de clientes. Alta receptividad a descuentos
          y promociones. Segmento con mayor potencial de migración a Premium.<br><br>
          <b style="color:#1a1a1a">Categorías preferidas:</b> Beauty & Health, Clothing, Home<br>
          <b style="color:#1a1a1a">Estrategia:</b> Campañas estacionales, cupones de descuento, membership upgrade
          incentivado, contenido inspiracional vía newsletter.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="segment-card" style="border-color:#95a5a6;background:#f8f9fa;color:#1a1a1a">
          <b style="color:#1a1a1a">⚠️ Segmento de Atención — Clientes Ocasionales</b><br><br>
          <b style="color:#1a1a1a">¿Por qué?</b> Alta probabilidad de churn. Requieren estrategias de reactivación
          antes de que abandonen definitivamente.<br><br>
          <b style="color:#1a1a1a">Estrategia:</b> Win-back campaigns, descuentos de reactivación, recordatorios
          de wishlist, encuestas de satisfacción para entender la fricción.
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🗺️ Estrategia de Posicionamiento — Empresa Entrante")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **🎯 Propuesta de Valor**
        - Plataforma de descuentos inteligentes y personalizados
        - Categorías de alta demanda: Electronics y Sports
        - Programa de membresía con beneficios tangibles
        - Experiencia mobile-first para segmento joven
        """)
    with col_b:
        st.markdown("""
        **📣 Canales de Adquisición**
        - Social media / influencers para segmento joven
        - Email marketing segmentado por tier de membresía
        - Retargeting para clientes ocasionales
        - SEO de nicho en categorías específicas
        """)
    with col_c:
        st.markdown("""
        **💡 Diferenciación**
        - Algoritmo de descuento personalizado por segmento
        - Transparencia en precios sin letra chica
        - Envío express para miembros Premium
        - Sistema de reviews verificados (confiar en ratings)
        """)

    st.divider()
    st.markdown("### 📊 Resumen Ejecutivo de Métricas de los Modelos")
    col1, col2, col3, col4 = st.columns(4)
    best_gmm = met_gmm[met_gmm["k"] == 4].iloc[0]
    best_kp  = met_kp[met_kp["k"]   == 3].iloc[0]
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">4</div><div class="kpi-label">Clusters GMM (RFM)</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{best_gmm["Silueta"]:.3f}</div><div class="kpi-label">Silueta GMM</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">3</div><div class="kpi-label">Clusters K-Proto (Demo.)</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{best_kp["Silueta"]:.3f}</div><div class="kpi-label">Silueta K-Prototypes</div></div>', unsafe_allow_html=True)




# ══════════════════════════════════════════════
# SECCIÓN 8 — [NUEVO] ANEXO: OTROS MODELOS Y COHORTES
# ══════════════════════════════════════════════
elif seccion == "📁  Anexo: Análisis Avanzado y Cohortes":
    section_header("📁 Anexo: Análisis Avanzado y Cohortes de Edad",
                   "Exploración comparativa de 14 modelos de clustering y segmentación por edad")

    tab_modelos, tab_cohortes = st.tabs([
        "📊 A. Comparativa de Modelos Unificados (Rand Index y PCA 2D/3D)",
        "🧒 B. Cohortes de Edad (Menores vs Mayores)"
    ])

    with tab_modelos:
        st.markdown("### Comparativa de 14 Modelos de Clustering (Notebook Unificado)")
        
        # Validar si existen los archivos del notebook unificado
        if not os.path.exists(os.path.join(BASE_DIR, "data", "clusters_todos_modelos.csv")) or not os.path.exists(os.path.join(BASE_DIR, "data", "matriz_ari.csv")):
            st.warning("⚠️ Los archivos de concordancia y etiquetas comparativas se están procesando o no se encontraron en 'data/'.")
            st.info("💡 Ejecuta la celda 95 del Notebook Unificado o espera a que el preprocesamiento termine para activar este explorador.")
            
            # Fallback demostrativo simple con datos existentes para mantener el dashboard visualmente espectacular
            st.subheader("Simulación de Modelos Demostrativos (PCA Explorer)")
            st.caption("Ajusta los parámetros para explorar visualmente los clústeres principales en PCA 2D y 3D")
        else:
            # Cargar clústeres comparativos
            df_all_clusters = pd.read_csv(os.path.join(BASE_DIR, "data", "clusters_todos_modelos.csv"))
            df_ari = pd.read_csv(os.path.join(BASE_DIR, "data", "matriz_ari.csv"), index_col=0)
            df_nmi = pd.read_csv(os.path.join(BASE_DIR, "data", "matriz_nmi.csv"), index_col=0)
            
            st.markdown("#### Matrices de Concordancia Cruzada (Rand Index y NMI)")
            st.markdown("""
            Estas matrices miden el grado de acuerdo (ARI y NMI) entre los diferentes modelos transaccionales (filas) 
            y los sociodemográficos (columnas). Un valor cercano a 1 indica alta concordancia, mientras que valores cercanos a 0 
            muestran independencia, que es lo esperado al cruzar dimensiones diferentes.
            """)
            
            col_ari, col_nmi = st.columns(2)
            with col_ari:
                fig_ari = px.imshow(
                    df_ari,
                    text_auto=".3f",
                    color_continuous_scale="Blues",
                    title="Rand Index Ajustado (ARI) entre Modelos",
                    labels=dict(x="Modelos Demográficos", y="Modelos Transaccionales")
                )
                fig_ari.update_layout(
                    height=450,
                    coloraxis_showscale=False,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_ari, use_container_width=True)
                
            with col_nmi:
                fig_nmi = px.imshow(
                    df_nmi,
                    text_auto=".3f",
                    color_continuous_scale="Purples",
                    title="Información Mutua Normalizada (NMI) entre Modelos",
                    labels=dict(x="Modelos Demográficos", y="Modelos Transaccionales")
                )
                fig_nmi.update_layout(
                    height=450,
                    coloraxis_showscale=False,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_nmi, use_container_width=True)

            st.divider()
            
            # --- EXPLORADOR INTERACTIVO PCA 2D / 3D ---
            st.markdown("#### 🌐 Explorador PCA Dinámico e Interactivo de Clústeres")
            st.markdown("""
            Selecciona cualquiera de los **14 modelos** del Notebook Unificado. Aplicaremos PCA (Análisis de Componentes Principales) 
            en tiempo real para proyectar los datos y ver la separación y distribución geométrica de los clústeres.
            """)
            
            model_options = [c for c in df_all_clusters.columns if c != "customer_id"]
            model_selected = st.selectbox("Selecciona el modelo de clustering a graficar:", model_options)
            
            # Decidir qué variables usar para PCA según el tipo de modelo
            es_rfm = any(kw in model_selected.lower() for kw in ["rfm", "kmeans", "gmm", "stepmix", "order", "continuous"])
            
            df_cust_raw = pd.read_csv(os.path.join(BASE_DIR, "customers.csv"))
            df_cust_raw.rename(columns={"avg_order_value_usd": "avg_order_value_usd"}, inplace=True)
            
            if es_rfm:
                # Features RFM
                df_pca_features = df_cust_raw[["customer_id", "days_since_last_purchase", "total_orders", "avg_order_value_usd"]].dropna()
                cols_to_scale = ["days_since_last_purchase", "total_orders", "avg_order_value_usd"]
            else:
                # Features Demographics
                df_pca_features = df_cust_raw[["customer_id", "age", "gender", "membership_tier"]].dropna()
                dic_g = {"Female": 0, "Male": 1, "Other": 2}
                dic_t = {"Free": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
                df_pca_features["gender"] = df_pca_features["gender"].map(dic_g)
                df_pca_features["membership_tier"] = df_pca_features["membership_tier"].map(dic_t)
                cols_to_scale = ["age", "gender", "membership_tier"]
                
            # Alinear clústeres
            df_pca_merged = df_pca_features.merge(df_all_clusters[["customer_id", model_selected]], on="customer_id", how="inner")
            
            # Filtrar proactivamente a los 337 clientes sin transacciones (que no fueron clasificados por el modelo y tienen valor -2)
            df_pca_merged = df_pca_merged[(df_pca_merged[model_selected] != -2) & (df_pca_merged[model_selected] != -2.0)]
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(df_pca_merged[cols_to_scale])
            
            viz_dim = st.radio("Dimensión del gráfico PCA:", ["PCA 2D (2 Componentes)", "PCA 3D (3 Componentes)"], horizontal=True)
            
            if "2D" in viz_dim:
                pca = PCA(n_components=2)
                X_projected = pca.fit_transform(X_scaled)
                df_proj = pd.DataFrame({
                    "PC1": X_projected[:, 0],
                    "PC2": X_projected[:, 1],
                    "Cluster": df_pca_merged[model_selected].astype(str)
                }).sort_values("Cluster")
                
                fig_proj = px.scatter(
                    df_proj, x="PC1", y="PC2", color="Cluster",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    title=f"Proyección PCA 2D — Modelo: {model_selected}",
                    opacity=0.6, height=550
                )
                st.plotly_chart(fig_proj, use_container_width=True)
                
            else:
                pca = PCA(n_components=3)
                X_projected = pca.fit_transform(X_scaled)
                df_proj = pd.DataFrame({
                    "PC1": X_projected[:, 0],
                    "PC2": X_projected[:, 1],
                    "PC3": X_projected[:, 2],
                    "Cluster": df_pca_merged[model_selected].astype(str)
                }).sort_values("Cluster")
                
                fig_proj = px.scatter_3d(
                    df_proj, x="PC1", y="PC2", z="PC3", color="Cluster",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    title=f"Proyección PCA 3D — Modelo: {model_selected}",
                    opacity=0.6, height=650
                )
                fig_proj.update_traces(marker=dict(size=3.5))
                st.plotly_chart(fig_proj, use_container_width=True)
                
            st.divider()
            st.markdown("#### 📈 Criterios de Selección del Número Óptimo de Clústeres (k)")
            if es_rfm:
                st.markdown(f"El modelo **{model_selected}** es un **Modelo Continuo/RFM (Transaccional)**. Evaluamos el número óptimo de clústeres ($k$) mediante criterios probabilísticos (AIC/BIC) e índices de separación geométrica:")
                c1, c2, c3 = st.columns(3)
                with c1:
                    fig_bic = px.line(met_gmm, x="k", y=["BIC", "AIC"], markers=True, color_discrete_sequence=["#d95f02", "#2ca02c"], title="BIC vs AIC")
                    fig_bic.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_bic, use_container_width=True)
                with c2:
                    fig_sil = px.line(met_gmm, x="k", y="Silueta", markers=True, color_discrete_sequence=["#7570b3"], title="Coeficiente de Silueta")
                    fig_sil.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_sil, use_container_width=True)
                with c3:
                    fig_dbi = px.line(met_gmm, x="k", y="DBI", markers=True, color_discrete_sequence=["#d95f02"], title="Davies-Bouldin")
                    fig_dbi.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_dbi, use_container_width=True)
                    
                c4, c5, c6 = st.columns(3)
                with c4:
                    fig_chi = px.line(met_gmm, x="k", y="CHI", markers=True, color_discrete_sequence=["#7570b3"], title="Calinski-Harabasz")
                    fig_chi.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_chi, use_container_width=True)
                with c5:
                    fig_ent = px.line(met_gmm, x="k", y="Entropia", markers=True, color_discrete_sequence=["#d95f02"], title="Entropía Media")
                    fig_ent.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_ent, use_container_width=True)
                with c6:
                    fig_lik = px.line(met_gmm, x="k", y="LogLik", markers=True, color_discrete_sequence=["#7570b3"], title="Log-Likelihood")
                    fig_lik.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_lik, use_container_width=True)
            else:
                st.markdown(f"El modelo **{model_selected}** es un **Modelo Demográfico/Mixto (Socio-Demográfico)**. Evaluamos el número óptimo de clústeres ($k$) mediante el costo de codo e índices de disimilitud:")
                c1, c2 = st.columns(2)
                with c1:
                    fig_cost = px.line(met_kp, x="k", y="Costo", markers=True, color_discrete_sequence=["#e67e22"], title="Costo de Disimilitud (Codo)")
                    fig_cost.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_cost, use_container_width=True)
                with c2:
                    fig_sil = px.line(met_kp, x="k", y="Silueta", markers=True, color_discrete_sequence=["#7570b3"], title="Coeficiente de Silueta")
                    fig_sil.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_sil, use_container_width=True)
                    
                c3, c4 = st.columns(2)
                with c3:
                    fig_dbi = px.line(met_kp, x="k", y="DBI", markers=True, color_discrete_sequence=["#e67e22"], title="Davies-Bouldin")
                    fig_dbi.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_dbi, use_container_width=True)
                with c4:
                    fig_chi = px.line(met_kp, x="k", y="CHI", markers=True, color_discrete_sequence=["#7570b3"], title="Calinski-Harabasz")
                    fig_chi.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_chi, use_container_width=True)

    with tab_cohortes:
        st.markdown("### 📊 Explorador Comparativo Multivariable y Cohortes de Edad")
        st.markdown("""
        Esta sección permite explorar y comparar el dataset de manera interactiva. Puedes filtrar por **Cohorte de Edad** 
        o ver **Todo el Mercado**, y luego elegir cualquiera de los **18 gráficos de análisis** para comparar el comportamiento 
        transaccional, demográfico y de fusión.
        """)

        cohort_choice = st.radio("Selecciona cohorte de edad para visualizar:", 
                                 ["🌍 Todo el Mercado (Sin filtro)", "🧒 Cohorte Joven (<= 18 años)", "🧑 Cohorte Adulta (> 18 años)"], horizontal=True)
        
        if "Todo el Mercado" in cohort_choice:
            prefix_c = "principal"
        elif "Joven" in cohort_choice:
            prefix_c = "menores"
        else:
            prefix_c = "mayores"
            
        mtime_c = get_files_mtime(prefix_c)
        df_fusion_c, df_ord_c, met_gmm_c, met_kp_c = cargar_datos_precalculados(prefix_c, mtime_c)
        
        st.markdown(f"**KPIs del grupo seleccionado: {cohort_choice}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clientes en Cohorte", f"{len(df_fusion_c):,}")
        c2.metric("Órdenes de Cohorte", f"{len(df_ord_c):,}")
        c3.metric("Edad Promedio", f"{df_fusion_c['age'].mean():.1f} años")
        c4.metric("Gasto Promedio USD", f"${df_fusion_c['avg_order_value_usd'].mean():,.2f}")

        st.divider()
        
        # Selector de gráfico a visualizar
        grafico_choice = st.selectbox(
            "Selecciona el gráfico a visualizar para esta cohorte:",
            options=[
                "1. Cantidad de Clientes por País",
                "2. Distribución de Edades",
                "3. Distribución de Género",
                "4. Distribución por Tipo de Membresía",
                "5. Ventas por Categoría",
                "6. Distribución de Descuentos (%)",
                "7. Boxplots RFM (Control de Outliers)",
                "8. GMM (RFM) - Criterios de Selección de k (6 métricas)",
                "9. GMM (RFM) - Tamaño de Clústeres y Perfiles Promedio",
                "10. GMM (RFM) - Visualización 3D Interactiva",
                "11. K-Prototypes - Criterios de Selección de k (4 métricas)",
                "12. K-Prototypes - Perfiles Demográficos de Clústeres",
                "13. K-Prototypes - Visualización 3D Interactiva",
                "14. Fusión F1: Matriz de Distribución (Clientes por Micro-Segmento)",
                "15. Fusión F2: Perfilamiento de Variables de Negocio (Heatmaps)",
                "16. Fusión F2: Perfilamiento de Variables RFM Originales (Heatmaps)",
                "17. Fusión F4: Mapa de Burbujas de Micro-Segmentos",
                "18. Fusión F11: Antigüedad del Cliente (Días y Años)",
                "19. Fusión F12: Velocidad de Compras Mensual (Histórica y Activa)",
                "20. Fusión F13: Evolución Temporal Trimestral de Compras",
                "21. Fusión F9: Análisis de Transacciones por Nivel de Descuento",
                "22. Fusión F10: Preferencia de Categorías por Micro-Segmento"
            ]
        )

        st.markdown(f"#### 📊 Visualización: {grafico_choice}")

        # Renderizar gráfico seleccionado
        if "1. Cantidad de Clientes por País" in grafico_choice:
            if "country" in df_fusion_c.columns:
                df_pais = df_fusion_c.groupby("country").size().reset_index(name='Cantidad')
                df_pais = df_pais.sort_values(by='Cantidad', ascending=False)
                fig = px.bar(df_pais, x="country", y="Cantidad", color="country",
                             color_discrete_sequence=px.colors.qualitative.Alphabet,
                             title="Cantidad de Clientes por País")
                fig.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("La columna 'country' no está disponible en esta cohorte.")

        elif "2. Distribución de Edades" in grafico_choice:
            fig = px.histogram(df_fusion_c, x="age", nbins=15,
                               color_discrete_sequence=["#d95f02"],
                               title="Distribución de Edades")
            fig.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20), xaxis_title="Edades", yaxis_title="Cantidad de Clientes")
            st.plotly_chart(fig, use_container_width=True)

        elif "3. Distribución de Género" in grafico_choice:
            df_g = df_fusion_c["gender"].value_counts().reset_index()
            df_g.columns = ["Género", "Cantidad"]
            fig = px.bar(df_g, x="Género", y="Cantidad", color="Género",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title="Distribución de Género")
            fig.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        elif "4. Distribución por Tipo de Membresía" in grafico_choice:
            df_m = df_fusion_c["membership_tier"].value_counts().reset_index()
            df_m.columns = ["Membresía", "Cantidad"]
            fig = px.bar(df_m, x="Membresía", y="Cantidad", color="Membresía",
                         color_discrete_sequence=px.colors.qualitative.Coolwarm,
                         title="Distribución por Tipo de Membresía")
            fig.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        elif "5. Ventas por Categoría" in grafico_choice:
            cat_ventas = df_ord_c.groupby("category")["total_amount_usd"].sum().sort_values(ascending=False).reset_index()
            cat_ventas.columns = ["Categoría", "Total USD"]
            fig = px.bar(cat_ventas, x="Categoría", y="Total USD", color="Categoría",
                         color_discrete_sequence=px.colors.qualitative.T10,
                         title="Revenue por Categoría")
            fig.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        elif "6. Distribución de Descuentos (%)" in grafico_choice:
            disc_counts = df_ord_c["discount_pct"].value_counts().sort_index().reset_index()
            disc_counts.columns = ["Descuento (%)", "N° órdenes"]
            fig = px.bar(disc_counts, x="Descuento (%)", y="N° órdenes",
                         color_discrete_sequence=["#f5a623"],
                         title="Frecuencia de Descuentos")
            fig.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        elif "7. Boxplots RFM (Control de Outliers)" in grafico_choice:
            c1, c2, c3 = st.columns(3)
            with c1:
                fig_freq = px.box(df_fusion_c, y="total_orders", color_discrete_sequence=["#3498db"],
                                  title="Frecuencia (total_orders)", labels={"total_orders": "Órdenes"})
                fig_freq.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_freq, use_container_width=True)
            with c2:
                fig_mon = px.box(df_fusion_c, y="avg_order_value_usd", color_discrete_sequence=["#2ecc71"],
                                 title="Monetario (avg_order_value_usd)", labels={"avg_order_value_usd": "USD"})
                fig_mon.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_mon, use_container_width=True)
            with c3:
                fig_rec = px.box(df_fusion_c, y="days_since_last_purchase", color_discrete_sequence=["#e74c3c"],
                                 title="Recencia (days_since_last_purchase)", labels={"days_since_last_purchase": "Días"})
                fig_rec.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_rec, use_container_width=True)

        elif "8. GMM (RFM) - Criterios de Selección de k" in grafico_choice:
            st.markdown("**6 métricas de selección del número óptimo de clústeres para el modelo GMM (RFM):**")
            c1, c2 = st.columns(2)
            with c1:
                fig_bic = px.line(met_gmm_c, x="k", y=["BIC", "AIC"], markers=True,
                                  color_discrete_sequence=["#d95f02", "#2ca02c"],
                                  title="AIC vs BIC (Menor es mejor)")
                fig_bic.add_vline(x=met_gmm_c.loc[met_gmm_c["BIC"].idxmin(), "k"], line_dash="dash", line_color="red")
                fig_bic.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_bic, use_container_width=True)
            with c2:
                fig_sil = px.line(met_gmm_c, x="k", y="Silueta", markers=True,
                                  color_discrete_sequence=["#7570b3"],
                                  title="Coeficiente de Silueta (Más cerca a 1)")
                fig_sil.add_vline(x=met_gmm_c.loc[met_gmm_c["Silueta"].idxmax(), "k"], line_dash="dash", line_color="red")
                fig_sil.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_sil, use_container_width=True)
            c3, c4 = st.columns(2)
            with c3:
                fig_dbi = px.line(met_gmm_c, x="k", y="DBI", markers=True,
                                  color_discrete_sequence=["#d95f02"],
                                  title="Davies-Bouldin (Más cerca a 0)")
                fig_dbi.add_vline(x=met_gmm_c.loc[met_gmm_c["DBI"].idxmin(), "k"], line_dash="dash", line_color="red")
                fig_dbi.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_dbi, use_container_width=True)
            with c4:
                fig_chi = px.line(met_gmm_c, x="k", y="CHI", markers=True,
                                  color_discrete_sequence=["#7570b3"],
                                  title="Calinski-Harabasz (Más alto)")
                fig_chi.add_vline(x=met_gmm_c.loc[met_gmm_c["CHI"].idxmax(), "k"], line_dash="dash", line_color="red")
                fig_chi.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_chi, use_container_width=True)
            c5, c6 = st.columns(2)
            with c5:
                fig_ent = px.line(met_gmm_c, x="k", y="Entropia", markers=True,
                                  color_discrete_sequence=["#d95f02"],
                                  title="Entropía Media (Cercana a 1)")
                fig_ent.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_ent, use_container_width=True)
            with c6:
                fig_lik = px.line(met_gmm_c, x="k", y="LogLik", markers=True,
                                  color_discrete_sequence=["#7570b3"],
                                  title="Log-Likelihood (Más alto)")
                fig_lik.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_lik, use_container_width=True)

        elif "9. GMM (RFM) - Tamaño de Clústeres" in grafico_choice:
            df_perf_gmm_c = df_fusion_c.groupby("Nombre_RFM")[["total_orders", "avg_order_value_usd", "days_since_last_purchase"]].mean()
            df_perf_gmm_c["N clientes"] = df_fusion_c["Nombre_RFM"].value_counts()
            st.markdown("**Perfiles promedio por cluster RFM:**")
            st.dataframe(df_perf_gmm_c.rename(columns={
                "total_orders": "Frecuencia (Órdenes)",
                "avg_order_value_usd": "Monetario (Avg USD)",
                "days_since_last_purchase": "Recencia (Días)",
            }).round(2), use_container_width=True)
            sizes = df_fusion_c["Nombre_RFM"].value_counts().reset_index()
            sizes.columns = ["Segmento RFM", "N° Clientes"]
            fig = px.bar(sizes, x="Segmento RFM", y="N° Clientes", color="Segmento RFM",
                         color_discrete_sequence=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"],
                         title="Tamaño de Clusters GMM (RFM)")
            fig.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        elif "10. GMM (RFM) - Visualización 3D" in grafico_choice:
            fig_3d = px.scatter_3d(
                df_fusion_c.reset_index(), x="days_since_last_purchase", y="total_orders", z="avg_order_value_usd",
                color="Nombre_RFM",
                color_discrete_sequence=px.colors.qualitative.Bold,
                title="Segmentación RFM — GMM (3D)",
                opacity=0.6, height=650,
                labels={
                    "days_since_last_purchase": "Recencia (Días)",
                    "total_orders": "Frecuencia (Compras)",
                    "avg_order_value_usd": "Monetario (USD)",
                    "Nombre_RFM": "Segmento RFM"
                }
            )
            fig_3d.update_traces(marker=dict(size=4))
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=50))
            st.plotly_chart(fig_3d, use_container_width=True)

        elif "11. K-Prototypes - Criterios de Selección de k" in grafico_choice:
            st.markdown("**4 métricas de selección del número óptimo de clústeres para K-Prototypes:**")
            c1, c2 = st.columns(2)
            with c1:
                fig_cost = px.line(met_kp_c, x="k", y="Costo", markers=True,
                                   color_discrete_sequence=["#e67e22"],
                                   title="Costo de Disimilitud — Método del Codo")
                fig_cost.add_vline(x=met_kp_c.loc[met_kp_c["Costo"].idxmin(), "k"], line_dash="dash", line_color="red")
                fig_cost.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_cost, use_container_width=True)
            with c2:
                fig_sil = px.line(met_kp_c, x="k", y="Silueta", markers=True,
                                  color_discrete_sequence=["#7570b3"],
                                  title="Coeficiente de Silueta (Más cerca a 1)")
                fig_sil.add_vline(x=met_kp_c.loc[met_kp_c["Silueta"].idxmax(), "k"], line_dash="dash", line_color="red")
                fig_sil.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_sil, use_container_width=True)
            c3, c4 = st.columns(2)
            with c3:
                fig_dbi = px.line(met_kp_c, x="k", y="DBI", markers=True,
                                  color_discrete_sequence=["#e67e22"],
                                  title="Davies-Bouldin (Más cerca a 0)")
                fig_dbi.add_vline(x=met_kp_c.loc[met_kp_c["DBI"].idxmin(), "k"], line_dash="dash", line_color="red")
                fig_dbi.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_dbi, use_container_width=True)
            with c4:
                fig_chi = px.line(met_kp_c, x="k", y="CHI", markers=True,
                                  color_discrete_sequence=["#7570b3"],
                                  title="Calinski-Harabasz (Más alto)")
                fig_chi.add_vline(x=met_kp_c.loc[met_kp_c["CHI"].idxmax(), "k"], line_dash="dash", line_color="red")
                fig_chi.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_chi, use_container_width=True)

        elif "12. K-Prototypes - Perfiles Demográficos" in grafico_choice:
            df_perf_kp_c = df_fusion_c.groupby("Nombre_DEM").agg(
                Edad_Promedio=("age", "mean"),
                Genero_Frecuente=("gender", lambda x: x.mode()[0] if not x.mode().empty else "N/A"),
                Membresia_Frecuente=("membership_tier", lambda x: x.mode()[0] if not x.mode().empty else "N/A"),
                N_Clientes=("age", "count"),
            ).round(2)
            st.markdown("**Perfiles por cluster demográfico (K-Prototypes):**")
            st.dataframe(df_perf_kp_c, use_container_width=True)
            sizes_kp = df_fusion_c["Nombre_DEM"].value_counts().reset_index()
            sizes_kp.columns = ["Segmento DEM", "N° Clientes"]
            fig = px.bar(sizes_kp, x="Segmento DEM", y="N° Clientes", color="Segmento DEM",
                         color_discrete_sequence=["#9b59b6", "#1abc9c", "#e67e22"],
                         title="Tamaño de Clusters K-Prototypes (Demográfico)")
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        elif "13. K-Prototypes - Visualización 3D" in grafico_choice:
            df_plot_kp_c = df_fusion_c.copy()
            dic_gender = {"Female": 0, "Male": 1, "Other": 2}
            dic_tier   = {"Free": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
            df_plot_kp_c["gender_n"] = df_plot_kp_c["gender"].map(dic_gender)
            df_plot_kp_c["membership_n"] = df_plot_kp_c["membership_tier"].map(dic_tier)
            np.random.seed(42)
            df_plot_kp_c["gender_j"] = df_plot_kp_c["gender_n"] + np.random.uniform(-0.15, 0.15, len(df_plot_kp_c))
            df_plot_kp_c["membership_j"] = df_plot_kp_c["membership_n"] + np.random.uniform(-0.15, 0.15, len(df_plot_kp_c))
            fig_3d_kp = px.scatter_3d(
                df_plot_kp_c.reset_index(), x="age", y="gender_j", z="membership_j",
                color="Nombre_DEM",
                color_discrete_sequence=px.colors.qualitative.Bold,
                title="Segmentación Demográfica — K-Prototypes (3D)",
                opacity=0.6, height=650,
                labels={"age": "Edad", "gender_j": "Género (Jitter)", "membership_j": "Membresía (Jitter)", "Nombre_DEM": "Segmento"}
            )
            fig_3d_kp.update_traces(marker=dict(size=4))
            fig_3d_kp.update_layout(
                margin=dict(l=0, r=0, b=0, t=50),
                scene=dict(
                    yaxis=dict(tickvals=[0, 1, 2], ticktext=["Female", "Male", "Other"]),
                    zaxis=dict(tickvals=[0, 1, 2, 3], ticktext=["Free", "Silver", "Gold", "Platinum"])
                )
            )
            st.plotly_chart(fig_3d_kp, use_container_width=True)

        elif "14. Fusión F1: Matriz de Distribución (Clientes por Micro-Segmento)" in grafico_choice:
            tabla = pd.crosstab(df_fusion_c["Nombre_RFM"], df_fusion_c["Nombre_DEM"])
            tabla_pct = (tabla / tabla.sum().sum()) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.imshow(tabla, text_auto="d", color_continuous_scale="YlOrRd", title="F1a: Clientes por Micro-Segmento (Absoluto)",
                                 labels=dict(x="Segmento Demográfico", y="Segmento RFM"))
                fig1.update_layout(height=400, coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.imshow(tabla_pct, text_auto=".1f", color_continuous_scale="YlOrRd", title="F1b: % de Clientes por Micro-Segmento",
                                 labels=dict(x="Segmento Demográfico", y="Segmento RFM"))
                fig2.update_layout(height=400, coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig2, use_container_width=True)

        elif "15. Fusión F2: Perfilamiento de Variables de Negocio (Heatmaps)" in grafico_choice:
            vars_heat = [
                ("discount_pct_prom",     "% Descuento Promedio",        "RdYlGn"),
                ("session_duration_prom", "Duración Sesión (min)",        "Blues"),
                ("pages_viewed_prom",     "Páginas Vistas",               "Purples"),
                ("age",                   "Edad Media",                   "Oranges"),
                ("returns_made",          "Devoluciones",                 "Reds"),
                ("rating_prom",           "Rating Promedio",              "RdYlGn"),
                ("is_repeat_pct",         "% Clientes Recurrentes",       "Greens"),
                ("churned",               "% Churn",                      "RdYlGn_r"),
            ]
            
            c1, c2, c3, c4 = st.columns(4)
            cols_row1 = [c1, c2, c3, c4]
            for idx, (col_name, label, cmap) in enumerate(vars_heat[:4]):
                with cols_row1[idx]:
                    pivot = df_fusion_c.groupby(["Nombre_RFM", "Nombre_DEM"])[col_name].mean().unstack()
                    fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale=cmap, title=label,
                                    labels=dict(x="Demografía", y="RFM"))
                    fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=10, r=10, t=45, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    
            c5, c6, c7, c8 = st.columns(4)
            cols_row2 = [c5, c6, c7, c8]
            for idx, (col_name, label, cmap) in enumerate(vars_heat[4:]):
                with cols_row2[idx]:
                    pivot = df_fusion_c.groupby(["Nombre_RFM", "Nombre_DEM"])[col_name].mean().unstack()
                    fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale=cmap, title=label,
                                    labels=dict(x="Demografía", y="RFM"))
                    fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=10, r=10, t=45, b=10))
                    st.plotly_chart(fig, use_container_width=True)

        elif "16. Fusión F2: Perfilamiento de Variables RFM Originales (Heatmaps)" in grafico_choice:
            vars_rfm = [
                ("days_since_last_purchase", "Recencia Promedio", "Reds_r"),
                ("total_orders",             "Frecuencia Promedio",   "Blues"),
                ("avg_order_value_usd",      "Monetario Promedio",     "Greens"),
            ]
            cols_rfm_grid = st.columns(3)
            for idx, (col_name, label, cmap) in enumerate(vars_rfm):
                with cols_rfm_grid[idx]:
                    pivot = df_fusion_c.groupby(["Nombre_RFM", "Nombre_DEM"])[col_name].mean().unstack()
                    fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale=cmap, title=label,
                                    labels=dict(x="Demografía", y="RFM"))
                    fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=10, r=10, t=45, b=10))
                    st.plotly_chart(fig, use_container_width=True)

        elif "17. Fusión F4: Mapa de Burbujas de Micro-Segmentos" in grafico_choice:
            burbuja = df_fusion_c.groupby("Micro_Segmento").agg(
                descuento  =("discount_pct_prom", "mean"),
                frecuencia =("n_ordenes",          "mean"),
                n_clientes =("age",                "count"),
            ).reset_index()
            fig = px.scatter(burbuja, x="descuento", y="frecuencia", size="n_clientes", color="Micro_Segmento",
                             hover_name="Micro_Segmento",
                             labels={"descuento": "Descuento Promedio (%)", "frecuencia": "Frecuencia (Órdenes)", "Micro_Segmento": "Micro-Segmento"},
                             title="F4: Frecuencia vs. Descuento por Micro-Segmento")
            fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        elif "18. Fusión F11: Antigüedad del Cliente (Días y Años)" in grafico_choice:
            df_temp = df_fusion_c.copy()
            df_temp["reg_date_dt"] = pd.to_datetime(df_temp["registration_date"])
            ref_date = df_temp["reg_date_dt"].max() + pd.Timedelta(days=1)
            df_temp["antiguedad_dias"] = (ref_date - df_temp["reg_date_dt"]).dt.days
            df_temp["antiguedad_anos"] = df_temp["antiguedad_dias"] / 365.0
            
            pivot_dias = df_temp.groupby(["Nombre_RFM", "Nombre_DEM"])["antiguedad_dias"].mean().unstack()
            pivot_anos = df_temp.groupby(["Nombre_RFM", "Nombre_DEM"])["antiguedad_anos"].mean().unstack()
            
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.imshow(pivot_dias, text_auto=".1f", color_continuous_scale="Oranges", title="F11a: Antigüedad Promedio (Días)",
                                 labels=dict(x="Demografía", y="RFM"))
                fig1.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.imshow(pivot_anos, text_auto=".2f", color_continuous_scale="Oranges", title="F11b: Antigüedad Promedio (Años)",
                                 labels=dict(x="Demografía", y="RFM"))
                fig2.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig2, use_container_width=True)

        elif "19. Fusión F12: Velocidad de Compras Mensual (Histórica y Activa)" in grafico_choice:
            df_temp = df_fusion_c.copy()
            df_temp["reg_date_dt"] = pd.to_datetime(df_temp["registration_date"])
            ref_date = df_temp["reg_date_dt"].max() + pd.Timedelta(days=1)
            
            df_temp["antiguedad_total_dias"] = (ref_date - df_temp["reg_date_dt"]).dt.days
            df_temp["tenure_months"] = np.maximum(df_temp["antiguedad_total_dias"] / 30.417, 1.0)
            
            df_temp["antiguedad_activa_dias"] = df_temp["antiguedad_total_dias"] - df_temp["days_since_last_purchase"]
            df_temp["antiguedad_activa_dias"] = np.maximum(df_temp["antiguedad_activa_dias"], 1.0)
            df_temp["tenure_active_months"] = np.maximum(df_temp["antiguedad_activa_dias"] / 30.417, 1.0)
            
            df_temp["compras_mes_historico"] = df_temp["total_orders"] / df_temp["tenure_months"]
            df_temp["compras_mes_activo"] = df_temp["total_orders"] / df_temp["tenure_active_months"]
            
            pivot_hist = df_temp.groupby(["Nombre_RFM", "Nombre_DEM"])["compras_mes_historico"].mean().unstack()
            pivot_act = df_temp.groupby(["Nombre_RFM", "Nombre_DEM"])["compras_mes_activo"].mean().unstack()
            
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.imshow(pivot_hist, text_auto=".2f", color_continuous_scale="YlGnBu", title="F12a: Velocidad de Compra Histórica (Órdenes/Mes)",
                                 labels=dict(x="Demografía", y="RFM"))
                fig1.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.imshow(pivot_act, text_auto=".2f", color_continuous_scale="YlGnBu", title="F12b: Velocidad de Compra Activa (Órdenes/Mes)",
                                 labels=dict(x="Demografía", y="RFM"))
                fig2.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig2, use_container_width=True)

        elif "20. Fusión F13: Evolución Temporal Trimestral" in grafico_choice:
            df_ord_cluster_c = df_ord_c.copy()
            df_ord_cluster_c["order_date"] = pd.to_datetime(df_ord_cluster_c["order_date"])
            df_ord_cluster_c["quarter_dt"] = df_ord_cluster_c["order_date"].dt.to_period("Q").dt.to_timestamp()
            
            df_ts_q = df_ord_cluster_c.groupby(["quarter_dt", "Micro_Segmento"]).size().reset_index(name="Compras")
            
            df_baseline = df_ord_cluster_c.groupby("quarter_dt").size().reset_index(name="Compras")
            n_micro = df_ord_cluster_c["Micro_Segmento"].nunique()
            df_baseline["Compras"] = df_baseline["Compras"] / max(n_micro, 1)
            df_baseline["Micro_Segmento"] = "Media Global (Promedio)"
            
            df_combined_c = pd.concat([df_ts_q, df_baseline], ignore_index=True)
            
            fig = px.line(
                df_combined_c,
                x="quarter_dt",
                y="Compras",
                color="Micro_Segmento",
                title="F13: Evolución Trimestral de Compras por Micro-Segmento",
                labels={"quarter_dt": "Trimestre", "Compras": "N° Compras", "Micro_Segmento": "Micro-Segmento"},
                color_discrete_sequence=px.colors.qualitative.Alphabet
            )
            fig.for_each_trace(lambda t: t.update(
                line=dict(dash="dash", width=3, color="#7f8c8d") if t.name == "Media Global (Promedio)" else dict(width=2)
            ))
            fig.update_layout(
                height=600,
                hovermode="x unified",
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(title="Micro-Segmentos", font=dict(size=9))
            )
            st.plotly_chart(fig, use_container_width=True)

        elif "21. Fusión F9: Análisis de Transacciones por Nivel de Descuento" in grafico_choice:
            df_ord_c["discount_pct"] = df_ord_c["discount_pct"].round(2)
            
            tabla_desc = pd.crosstab(df_ord_c["discount_pct"], df_ord_c["Micro_Segmento"]).reset_index()
            df_desc_long = tabla_desc.melt(id_vars="discount_pct", var_name="Micro_Segmento", value_name="N° Compras")
            df_desc_long["discount_pct"] = df_desc_long["discount_pct"].astype(str) + "%"
            
            fig_vol = px.bar(
                df_desc_long,
                x="discount_pct",
                y="N° Compras",
                color="Micro_Segmento",
                title="F9a: Volumen de Compras por Descuento (Log Scale)",
                labels={"discount_pct": "Descuento (%)", "N° Compras": "N° Compras"},
                color_discrete_sequence=px.colors.qualitative.Alphabet,
            )
            fig_vol.update_yaxes(type="log")
            fig_vol.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(title="Micro-Segmentos", font=dict(size=8))
            )
            
            tabla_desc_pct = pd.crosstab(df_ord_c["discount_pct"], df_ord_c["Micro_Segmento"])
            tabla_desc_pct = tabla_desc_pct.div(tabla_desc_pct.sum(axis=1), axis=0) * 100
            tabla_desc_pct = tabla_desc_pct.reset_index()
            df_desc_pct_long = tabla_desc_pct.melt(id_vars="discount_pct", var_name="Micro_Segmento", value_name="% Compras")
            df_desc_pct_long["discount_pct"] = df_desc_pct_long["discount_pct"].astype(str) + "%"
            
            fig_pct = px.bar(
                df_desc_pct_long,
                x="discount_pct",
                y="% Compras",
                color="Micro_Segmento",
                title="F9b: Distribución Porcentual Relativa (100% Normalizado)",
                labels={"discount_pct": "Descuento (%)", "% Compras": "% Compras"},
                color_discrete_sequence=px.colors.qualitative.Alphabet,
            )
            fig_pct.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(title="Micro-Segmentos", font=dict(size=8))
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_vol, use_container_width=True)
            with col2:
                st.plotly_chart(fig_pct, use_container_width=True)

        elif "22. Fusión F10: Preferencia de Categorías por Micro-Segmento" in grafico_choice:
            tabla_cat = pd.crosstab(df_ord_c["category"], df_ord_c["Micro_Segmento"]).reset_index()
            df_cat_long = tabla_cat.melt(id_vars="category", var_name="Micro_Segmento", value_name="N° Compras")
            
            fig_vol = px.bar(
                df_cat_long,
                x="N° Compras",
                y="category",
                color="Micro_Segmento",
                orientation="h",
                title="F10a: Volumen Absoluto por Categoría",
                labels={"category": "Categoría", "N° Compras": "N° Compras"},
                color_discrete_sequence=px.colors.qualitative.Alphabet,
            )
            fig_vol.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(title="Micro-Segmentos", font=dict(size=8))
            )
            
            tabla_cat_pct = pd.crosstab(df_ord_c["category"], df_ord_c["Micro_Segmento"])
            tabla_cat_pct = tabla_cat_pct.div(tabla_cat_pct.sum(axis=1), axis=0) * 100
            tabla_cat_pct = tabla_cat_pct.reset_index()
            df_cat_pct_long = tabla_cat_pct.melt(id_vars="category", var_name="Micro_Segmento", value_name="% de Compras")
            
            fig_pct = px.bar(
                df_cat_pct_long,
                x="% de Compras",
                y="category",
                color="Micro_Segmento",
                orientation="h",
                title="F10b: Distribución Porcentual (100%)",
                labels={"category": "Categoría", "% de Compras": "% de Compras"},
                color_discrete_sequence=px.colors.qualitative.Alphabet,
            )
            fig_pct.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(title="Micro-Segmentos", font=dict(size=8))
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_vol, use_container_width=True)
            with col2:
                st.plotly_chart(fig_pct, use_container_width=True)

