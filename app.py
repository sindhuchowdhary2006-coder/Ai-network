"""
NetGuard AI — Network Attack Forecasting
Complete rewrite with proper Streamlit Cloud compatible design
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="NetGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.data_processor import (
    generate_mock_traffic_data, parse_uploaded_csv, extract_features,
    compute_flow_statistics, generate_time_series_data,
    generate_k_step_forecast, get_recent_alerts, TOP_FEATURES, ATTACK_LABELS,
)
from utils.model_utils import (
    generate_training_curves, get_model_performance_metrics, get_confusion_matrix,
    FEATURE_IMPORTANCE_DATA, get_shap_values, get_attention_weights,
    get_benchmark_dataframe, get_roc_curve_data, BENCHMARK_METRICS,
    MODEL_ARCHITECTURES, get_model_status, BENCHMARK_MODELS,
)
from utils.mitre_attack import (
    MITRE_STAGES, get_kill_chain_progression, generate_stage_probability_vector,
    get_technique_details_for_stage, get_mitre_heatmap_data, map_label_to_stage,
)
from utils.visualizations import (
    attack_probability_gauge, live_timeline_chart, flow_rate_chart,
    k_step_forecast_chart, training_curves_chart, feature_importance_chart,
    shap_beeswarm_chart, attention_heatmap, confusion_matrix_chart,
    benchmark_bar_chart, roc_curve_chart, label_distribution_pie,
    mitre_stage_radar, network_traffic_3d, benchmark_radar_chart, COLORS,
)

# ══════════════════════════════════════════════════════════════
#  MASTER CSS — Dark Cyber Theme with Deep Space Background
# ══════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Orbitron:wght@400;700;900&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── DEEP SPACE ANIMATED BACKGROUND ── */
    .stApp {
        background: #020818 !important;
        background-image:
            radial-gradient(ellipse at 20% 50%, rgba(79,139,249,0.15) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(124,77,255,0.12) 0%, transparent 40%),
            radial-gradient(ellipse at 50% 80%, rgba(0,212,255,0.08) 0%, transparent 40%),
            radial-gradient(ellipse at 90% 90%, rgba(229,57,53,0.08) 0%, transparent 30%) !important;
        background-attachment: fixed !important;
        min-height: 100vh;
    }

    /* ── STARS LAYER ── */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            radial-gradient(1px 1px at 10% 15%, rgba(255,255,255,0.6) 0%, transparent 100%),
            radial-gradient(1px 1px at 25% 35%, rgba(255,255,255,0.4) 0%, transparent 100%),
            radial-gradient(1.5px 1.5px at 40% 10%, rgba(79,139,249,0.8) 0%, transparent 100%),
            radial-gradient(1px 1px at 55% 55%, rgba(255,255,255,0.5) 0%, transparent 100%),
            radial-gradient(1px 1px at 70% 25%, rgba(255,255,255,0.3) 0%, transparent 100%),
            radial-gradient(1.5px 1.5px at 85% 70%, rgba(124,77,255,0.7) 0%, transparent 100%),
            radial-gradient(1px 1px at 15% 75%, rgba(255,255,255,0.4) 0%, transparent 100%),
            radial-gradient(1px 1px at 35% 85%, rgba(255,255,255,0.3) 0%, transparent 100%),
            radial-gradient(1px 1px at 60% 45%, rgba(255,255,255,0.5) 0%, transparent 100%),
            radial-gradient(1px 1px at 75% 90%, rgba(255,255,255,0.4) 0%, transparent 100%),
            radial-gradient(1px 1px at 90% 40%, rgba(255,255,255,0.6) 0%, transparent 100%),
            radial-gradient(1px 1px at 5% 90%, rgba(255,255,255,0.3) 0%, transparent 100%),
            radial-gradient(1px 1px at 45% 60%, rgba(255,255,255,0.4) 0%, transparent 100%),
            radial-gradient(1.5px 1.5px at 80% 5%, rgba(0,212,255,0.8) 0%, transparent 100%),
            radial-gradient(1px 1px at 20% 50%, rgba(255,255,255,0.5) 0%, transparent 100%),
            radial-gradient(1px 1px at 65% 80%, rgba(255,255,255,0.3) 0%, transparent 100%),
            radial-gradient(1px 1px at 30% 20%, rgba(255,255,255,0.6) 0%, transparent 100%),
            radial-gradient(1px 1px at 50% 30%, rgba(255,255,255,0.4) 0%, transparent 100%),
            radial-gradient(1px 1px at 95% 60%, rgba(255,255,255,0.5) 0%, transparent 100%),
            radial-gradient(1px 1px at 8% 40%, rgba(255,255,255,0.3) 0%, transparent 100%);
        pointer-events: none;
        z-index: 0;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e27 0%, #0d1235 40%, #0a0e27 100%) !important;
        border-right: 1px solid rgba(79,139,249,0.3) !important;
        box-shadow: 4px 0 30px rgba(79,139,249,0.15) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }
    [data-testid="stSidebar"] * {
        color: #c8d8ff !important;
    }

    /* ── SIDEBAR RADIO NAV ── */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 2px !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(79,139,249,0.1) !important;
        border-radius: 10px !important;
        padding: 10px 16px !important;
        margin: 2px 0 !important;
        cursor: pointer !important;
        transition: all 0.25s ease !important;
        display: flex !important;
        align-items: center !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #8ba4d4 !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(79,139,249,0.15) !important;
        border-color: rgba(79,139,249,0.5) !important;
        color: #ffffff !important;
        transform: translateX(4px) !important;
    }
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked),
    [data-testid="stSidebar"] .stRadio label[aria-checked="true"] {
        background: linear-gradient(135deg, rgba(79,139,249,0.3), rgba(124,77,255,0.2)) !important;
        border-color: rgba(79,139,249,0.7) !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(79,139,249,0.3), inset 0 0 20px rgba(79,139,249,0.05) !important;
    }

    /* ── MAIN CONTENT AREA ── */
    .main .block-container {
        padding: 1.5rem 2rem 3rem 2rem !important;
        max-width: 100% !important;
        position: relative;
        z-index: 1;
    }

    /* ── CARDS ── */
    .ng-card {
        background: linear-gradient(135deg, rgba(10,14,39,0.95) 0%, rgba(13,18,53,0.9) 100%) !important;
        border: 1px solid rgba(79,139,249,0.25) !important;
        border-radius: 16px !important;
        padding: 20px 22px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(79,139,249,0.1), inset 0 1px 0 rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(10px) !important;
        position: relative;
        overflow: hidden;
    }
    .ng-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(79,139,249,0.6), transparent);
    }
    .ng-card:hover {
        border-color: rgba(79,139,249,0.5) !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 20px rgba(79,139,249,0.15) !important;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }

    /* ── PAGE HEADER ── */
    .page-header {
        background: linear-gradient(135deg, rgba(79,139,249,0.2) 0%, rgba(124,77,255,0.15) 50%, rgba(0,212,255,0.1) 100%) !important;
        border: 1px solid rgba(79,139,249,0.4) !important;
        border-radius: 20px !important;
        padding: 28px 32px !important;
        margin-bottom: 24px !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 8px 40px rgba(79,139,249,0.2), inset 0 0 60px rgba(79,139,249,0.05) !important;
    }
    .page-header::after {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(124,77,255,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .page-header h1 {
        font-family: 'Orbitron', monospace !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin: 0 !important;
        text-shadow: 0 0 20px rgba(79,139,249,0.8) !important;
        letter-spacing: 1px !important;
    }
    .page-header p {
        font-size: 0.88rem !important;
        color: rgba(200,216,255,0.8) !important;
        margin: 8px 0 0 0 !important;
    }

    /* ── METRIC CARDS ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(10,14,39,0.98) 0%, rgba(13,18,53,0.95) 100%) !important;
        border: 1px solid rgba(79,139,249,0.2) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        text-align: center !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
        position: relative !important;
        overflow: hidden !important;
        transition: all 0.3s ease !important;
        height: 120px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: var(--card-color, linear-gradient(90deg, #4F8BF9, #7C4DFF));
    }
    .metric-card:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 20px var(--glow-color, rgba(79,139,249,0.3)) !important;
        border-color: rgba(79,139,249,0.5) !important;
    }
    .metric-value {
        font-size: 2rem !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
        font-family: 'Orbitron', monospace !important;
    }
    .metric-label {
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        color: rgba(180,196,255,0.7) !important;
        margin-top: 4px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    .metric-delta {
        font-size: 0.72rem !important;
        margin-top: 4px !important;
        font-weight: 600 !important;
    }

    /* ── SECTION HEADER ── */
    .section-header {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #c8d8ff !important;
        border-left: 3px solid #4F8BF9 !important;
        padding-left: 12px !important;
        margin: 16px 0 12px 0 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    /* ── ALERT BADGES ── */
    .badge-critical {
        background: linear-gradient(135deg, rgba(229,57,53,0.3), rgba(229,57,53,0.1));
        color: #ff6b6b; border: 1px solid rgba(229,57,53,0.5);
        border-radius: 20px; padding: 3px 12px; font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.5px;
        box-shadow: 0 0 10px rgba(229,57,53,0.3);
    }
    .badge-high {
        background: linear-gradient(135deg, rgba(255,143,0,0.3), rgba(255,143,0,0.1));
        color: #ffb347; border: 1px solid rgba(255,143,0,0.5);
        border-radius: 20px; padding: 3px 12px; font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .badge-medium {
        background: linear-gradient(135deg, rgba(255,235,0,0.2), rgba(255,235,0,0.05));
        color: #ffd700; border: 1px solid rgba(255,235,0,0.4);
        border-radius: 20px; padding: 3px 12px; font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .badge-low {
        background: linear-gradient(135deg, rgba(67,160,71,0.3), rgba(67,160,71,0.1));
        color: #69f0ae; border: 1px solid rgba(67,160,71,0.5);
        border-radius: 20px; padding: 3px 12px; font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.5px;
    }

    /* ── KILL CHAIN STAGES ── */
    .kc-stage {
        border-radius: 12px; padding: 12px 16px; margin-bottom: 8px;
        border: 1px solid rgba(79,139,249,0.15);
        background: rgba(255,255,255,0.02);
        transition: all 0.2s ease;
    }
    .kc-completed { border-left: 3px solid #43A047; opacity: 0.75; }
    .kc-active {
        border-left: 3px solid #E53935;
        box-shadow: 0 0 20px rgba(229,57,53,0.3), inset 0 0 20px rgba(229,57,53,0.05);
        background: rgba(229,57,53,0.08);
        border-color: rgba(229,57,53,0.4);
    }
    .kc-pending { border-left: 3px solid rgba(79,139,249,0.2); opacity: 0.4; }

    /* ── BUTTONS ── */
    .stButton > button {
        background: linear-gradient(135deg, #4F8BF9 0%, #7C4DFF 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(79,139,249,0.4) !important;
        text-transform: uppercase !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(79,139,249,0.6) !important;
    }

    /* ── INPUTS ── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(10,14,39,0.8) !important;
        border: 1px solid rgba(79,139,249,0.3) !important;
        border-radius: 8px !important;
        color: #c8d8ff !important;
    }
    .stSelectbox > div > div:hover,
    .stTextInput > div > div > input:focus {
        border-color: rgba(79,139,249,0.7) !important;
        box-shadow: 0 0 10px rgba(79,139,249,0.2) !important;
    }

    /* ── SLIDERS ── */
    .stSlider .st-be { background: rgba(79,139,249,0.3) !important; }
    .stSlider .st-bf { background: linear-gradient(90deg, #4F8BF9, #7C4DFF) !important; }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(10,14,39,0.8) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        border: 1px solid rgba(79,139,249,0.2) !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 500 !important;
        color: rgba(200,216,255,0.6) !important;
        font-size: 0.85rem !important;
        padding: 8px 16px !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4F8BF9, #7C4DFF) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(79,139,249,0.4) !important;
    }

    /* ── DATAFRAMES ── */
    .stDataFrame { border-radius: 12px !important; overflow: hidden !important; }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: rgba(10,14,39,0.9) !important;
        border: 1px solid rgba(79,139,249,0.2) !important;
        border-radius: 12px !important;
    }

    /* ── PROGRESS BAR ── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4F8BF9, #7C4DFF) !important;
        box-shadow: 0 0 10px rgba(79,139,249,0.5) !important;
    }

    /* ── EXPANDER ── */
    .streamlit-expanderHeader {
        background: rgba(79,139,249,0.1) !important;
        border: 1px solid rgba(79,139,249,0.2) !important;
        border-radius: 8px !important;
        color: #c8d8ff !important;
    }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(10,14,39,0.5); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #4F8BF9, #7C4DFF);
        border-radius: 10px;
    }

    /* ── HIDE STREAMLIT DEFAULTS ── */
    #MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] {
        visibility: hidden !important;
        display: none !important;
    }

    /* ── TEXT COLORS ── */
    h1, h2, h3, h4, h5, h6 { color: #e0eaff !important; }
    p, span, div, label { color: #b0c4ee !important; }
    .stMarkdown p { color: #b0c4ee !important; }

    /* ── METRICS ── */
    [data-testid="stMetricValue"] {
        color: #7EB8FF !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: rgba(176,196,238,0.7) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    [data-testid="stMetricDelta"] { font-weight: 600 !important; }

    /* ── GLOWING LINE DIVIDER ── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(79,139,249,0.5), transparent) !important;
        margin: 16px 0 !important;
    }

    /* ── INFO / WARNING / SUCCESS ── */
    .stAlert {
        background: rgba(79,139,249,0.1) !important;
        border: 1px solid rgba(79,139,249,0.3) !important;
        border-radius: 10px !important;
        color: #c8d8ff !important;
    }
    [data-baseweb="notification"] {
        background: rgba(79,139,249,0.1) !important;
        border-left: 3px solid #4F8BF9 !important;
    }

    /* Checkbox */
    .stCheckbox label span { color: #c8d8ff !important; }

    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PLOTLY DARK THEME DEFAULTS
# ══════════════════════════════════════════════════════════════
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,14,39,0.6)",
    font=dict(family="Inter, sans-serif", color="#b0c4ee"),
    margin=dict(l=40, r=40, t=50, b=40),
    hoverlabel=dict(bgcolor="#0d1235", font_size=13, font_family="Inter", font_color="#e0eaff",
                    bordercolor="rgba(79,139,249,0.5)"),
    legend=dict(bgcolor="rgba(10,14,39,0.8)", bordercolor="rgba(79,139,249,0.3)",
                borderwidth=1, font=dict(color="#b0c4ee")),
)
GRID_STYLE = dict(
    showgrid=True, gridcolor="rgba(79,139,249,0.1)", zeroline=False,
    linecolor="rgba(79,139,249,0.2)", tickfont=dict(color="#7090c0"),
)


def dark_fig(fig, title="", height=380):
    fig.update_layout(**DARK_LAYOUT, height=height)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15, color="#c8d8ff"), x=0.01))
    fig.update_xaxes(**GRID_STYLE)
    fig.update_yaxes(**GRID_STYLE)
    return fig


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:24px 0 20px 0;">
            <div style="font-size:3rem; filter:drop-shadow(0 0 20px rgba(79,139,249,0.8));">🛡️</div>
            <div style="font-family:'Orbitron',monospace; font-size:1.1rem; font-weight:900;
                        color:#ffffff; letter-spacing:2px; margin-top:8px;
                        text-shadow: 0 0 20px rgba(79,139,249,0.8);">NETGUARD AI</div>
            <div style="font-size:0.7rem; color:rgba(176,196,255,0.6); margin-top:4px;
                        letter-spacing:1px; text-transform:uppercase;">Network Attack Forecasting</div>
        </div>
        <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(79,139,249,0.5),transparent); margin:0 0 16px 0;"></div>
        """, unsafe_allow_html=True)

        pages = [
            "📊  Dashboard",
            "📂  Data Ingestion",
            "🧠  Model Training",
            "🔮  Attack Prediction",
            "🗺️  MITRE ATT&CK",
            "💡  Explainability",
            "📈  Benchmark",
            "⚙️  Settings",
        ]
        selected = st.radio("nav", pages, label_visibility="collapsed")

        # Model status
        ms = get_model_status()
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(79,139,249,0.08); border:1px solid rgba(79,139,249,0.2);
                    border-radius:12px; padding:14px 16px; margin-top:8px;">
            <div style="font-size:0.65rem; color:rgba(176,196,255,0.6); text-transform:uppercase;
                        letter-spacing:1px; margin-bottom:8px;">⚡ MODEL STATUS</div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <div style="width:8px; height:8px; border-radius:50%; background:#43A047;
                             box-shadow:0 0 8px #43A047;"></div>
                <span style="font-size:0.82rem; font-weight:700; color:#69f0ae;">OPERATIONAL</span>
            </div>
            <div style="font-size:0.75rem; color:rgba(176,196,255,0.7); line-height:1.6;">
                {ms['model_name']}<br>
                <span style="color:#4F8BF9;">{ms['version']}</span> · {ms['dataset']}
            </div>
        </div>
        <div style="text-align:center; margin-top:16px; font-size:0.65rem;
                    color:rgba(176,196,255,0.3); letter-spacing:1px;">
            SIH 2024 · NTRO CHALLENGE
        </div>
        """, unsafe_allow_html=True)

    return selected.split("  ", 1)[1].strip()


# ══════════════════════════════════════════════════════════════
#  HELPER COMPONENTS
# ══════════════════════════════════════════════════════════════
def page_header(icon, title, subtitle):
    st.markdown(f"""
    <div class="page-header">
        <h1>{icon} {title}</h1>
        <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)


def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def card_open():
    st.markdown('<div class="ng-card">', unsafe_allow_html=True)


def card_close():
    st.markdown('</div>', unsafe_allow_html=True)


def metric_html(label, value, delta="", color="#4F8BF9", glow="rgba(79,139,249,0.4)"):
    delta_html = ""
    if delta:
        up = delta.startswith("+")
        dc = "#69f0ae" if up else "#ff6b6b"
        delta_html = f'<div class="metric-delta" style="color:{dc};">{delta}</div>'
    return f"""
    <div class="metric-card" style="--card-color:{color}; --glow-color:{glow};">
        <div style="position:absolute;top:0;left:0;right:0;height:2px;
                    background:{color};box-shadow:0 0 8px {glow};"></div>
        <div class="metric-value" style="color:{color};
             text-shadow:0 0 20px {glow};">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>"""


def alert_badge(severity):
    cls = {"Critical": "badge-critical", "High": "badge-high",
           "Medium": "badge-medium", "Low": "badge-low"}.get(severity, "badge-low")
    return f'<span class="{cls}">{severity}</span>'


# ══════════════════════════════════════════════════════════════
#  DARK CHART OVERRIDES
# ══════════════════════════════════════════════════════════════
def make_gauge(prob, title="Attack Probability"):
    pct = round(prob * 100, 1)
    if pct < 25:   color, level = "#69f0ae", "LOW"
    elif pct < 50: color, level = "#ffd700", "MODERATE"
    elif pct < 75: color, level = "#ff9800", "HIGH"
    else:          color, level = "#ff4444", "CRITICAL"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        title={"text": f"<span style='font-size:0.85em;color:#c8d8ff'>{title}</span><br>"
                       f"<span style='font-size:0.75em;color:{color};font-weight:700'>{level}</span>",
               "font": {"size": 14}},
        number={"suffix": "%", "font": {"size": 42, "color": color, "family": "Orbitron"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a6fa5", "tickfont": {"size": 10, "color": "#7090c0"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(10,14,39,0.8)",
            "borderwidth": 2,
            "bordercolor": "rgba(79,139,249,0.3)",
            "steps": [
                {"range": [0, 25],   "color": "rgba(105,240,174,0.06)"},
                {"range": [25, 50],  "color": "rgba(255,215,0,0.06)"},
                {"range": [50, 75],  "color": "rgba(255,152,0,0.06)"},
                {"range": [75, 100], "color": "rgba(255,68,68,0.06)"},
            ],
            "threshold": {"line": {"color": "#ff4444", "width": 3}, "thickness": 0.8, "value": 70},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=30, r=30, t=60, b=20),
        font=dict(family="Inter, sans-serif", color="#b0c4ee"),
    )
    return fig


def make_timeline(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Timestamp"], y=df["Attack Probability"],
        mode="lines", name="Attack Prob",
        line=dict(color="#ff4444", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,68,68,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=df["Timestamp"], y=df["Anomaly Score"],
        mode="lines", name="Anomaly Score",
        line=dict(color="#7C4DFF", width=1.8, dash="dot"),
    ))
    fig.add_hline(y=0.7, line_dash="dash", line_color="rgba(255,68,68,0.7)",
                  annotation_text="Threshold 0.70", annotation_font_color="#ff6b6b",
                  annotation_position="top right")
    dark_fig(fig, "Live Attack Probability Timeline", 320)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[0, 1], title="Probability", **GRID_STYLE),
        xaxis=dict(title="Time", **GRID_STYLE),
    )
    return fig


def make_flow_chart(df):
    fig = go.Figure(go.Bar(
        x=df["Timestamp"], y=df["Flow Rate (flows/min)"],
        marker=dict(
            color=df["Flow Rate (flows/min)"],
            colorscale=[[0, "#1a2a6c"], [0.5, "#4F8BF9"], [1.0, "#7C4DFF"]],
            line_width=0,
        ),
        opacity=0.9,
    ))
    dark_fig(fig, "Network Flow Rate (flows/min)", 300)
    fig.update_layout(yaxis_title="Flows/min", xaxis_title="Time",
                      yaxis=dict(**GRID_STYLE), xaxis=dict(**GRID_STYLE))
    return fig


def make_kstep(df):
    fig = go.Figure()
    stage_colors = {
        "Reconnaissance": "#4FC3F7", "Initial Access": "#81C784",
        "Execution": "#FFB74D", "Lateral Movement": "#FF8A65",
        "Command & Control": "#CE93D8", "Exfiltration": "#EF5350",
    }
    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(df["Step"]) + list(df["Step"])[::-1],
        y=list(df["Upper Bound"]) + list(df["Lower Bound"])[::-1],
        fill="toself", fillcolor="rgba(79,139,249,0.1)",
        line=dict(color="rgba(255,255,255,0)"), name="95% Confidence",
    ))
    fig.add_trace(go.Scatter(
        x=df["Step"], y=df["Attack Probability"],
        mode="lines+markers", name="Attack Prob",
        line=dict(color="#4F8BF9", width=3),
        marker=dict(size=9, color="#7C4DFF", line=dict(color="#c8d8ff", width=2),
                    symbol="circle"),
    ))
    fig.add_hline(y=0.7, line_dash="dash", line_color="rgba(255,68,68,0.6)",
                  annotation_text="Alert Threshold", annotation_font_color="#ff6b6b")
    # Stage labels
    prev = None
    for _, row in df.iterrows():
        stage = row["Predicted Stage"]
        if stage != prev:
            fig.add_annotation(
                x=row["Step"], y=min(row["Attack Probability"] + 0.08, 0.98),
                text=stage, showarrow=False,
                font=dict(size=8, color=stage_colors.get(stage, "#c8d8ff")),
                bgcolor="rgba(10,14,39,0.85)",
                bordercolor=stage_colors.get(stage, "#4F8BF9"),
                borderwidth=1, borderpad=3,
            )
            prev = stage
    dark_fig(fig, "K-Step Forward Attack Probability Forecast", 400)
    fig.update_layout(
        xaxis=dict(title="Forecast Step (t+k)", tickmode="linear", **GRID_STYLE),
        yaxis=dict(title="Attack Probability", range=[0, 1.05], **GRID_STYLE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_training_curves(df):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Loss Curves", "Accuracy & F1 Curves"))
    fig.add_trace(go.Scatter(x=df["Epoch"], y=df["Train Loss"], name="Train Loss",
                              line=dict(color="#4F8BF9", width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Epoch"], y=df["Val Loss"], name="Val Loss",
                              line=dict(color="#ff4444", width=2.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Epoch"], y=df["Train Accuracy"], name="Train Acc",
                              line=dict(color="#69f0ae", width=2.2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["Epoch"], y=df["Val Accuracy"], name="Val Acc",
                              line=dict(color="#00d4ff", width=2.2, dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["Epoch"], y=df["Train F1"], name="Train F1",
                              line=dict(color="#ffd700", width=2.0, dash="dash")), row=1, col=2)
    fig.update_layout(**DARK_LAYOUT, height=360,
                      legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
    fig.update_xaxes(**GRID_STYLE)
    fig.update_yaxes(**GRID_STYLE)
    for i in fig.layout.annotations:
        i.font.color = "#c8d8ff"
    return fig


def make_feature_importance(data, top_n=16):
    items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    feats = [x[0] for x in items]
    vals = [x[1] for x in items]
    colors = [f"rgba(79,{int(139+(116*(v/max(vals)))//1)},249,0.9)" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals[::-1], y=feats[::-1], orientation="h",
        marker=dict(color=vals[::-1],
                    colorscale=[[0, "#1a2a6c"], [0.5, "#4F8BF9"], [1, "#7C4DFF"]],
                    line_width=0),
        text=[f"{v:.3f}" for v in vals[::-1]], textposition="outside",
        textfont=dict(color="#c8d8ff", size=11),
    ))
    dark_fig(fig, "Global Feature Importance (SHAP-based)", 440)
    fig.update_layout(xaxis_title="Importance", showlegend=False,
                      xaxis=dict(**GRID_STYLE), yaxis=dict(**GRID_STYLE, tickfont=dict(color="#b0c4ee", size=11)))
    return fig


def make_attention_heatmap(attn):
    seq_len = attn.shape[0]
    labels = [f"t-{seq_len-1-i}" if i < seq_len-1 else "t" for i in range(seq_len)]
    fig = go.Figure(go.Heatmap(
        z=attn, x=labels, y=labels,
        colorscale=[[0, "#0a0e27"], [0.3, "#1a3a8a"], [0.7, "#4F8BF9"], [1.0, "#7C4DFF"]],
        showscale=True,
        colorbar=dict(title="Weight", thickness=12, tickfont=dict(color="#b0c4ee")),
        hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>",
    ))
    dark_fig(fig, "Transformer Attention Weights", 420)
    fig.update_layout(xaxis_title="Key", yaxis_title="Query",
                      xaxis=dict(**GRID_STYLE), yaxis=dict(**GRID_STYLE))
    return fig


def make_confusion_matrix(cm, model_name):
    labels = ["Benign", "Attack"]
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ann = []
    for i in range(2):
        for j in range(2):
            ann.append(dict(
                text=f"<b>{cm[i][j]:,}</b><br>({norm[i][j]:.1%})",
                x=labels[j], y=labels[i], xref="x", yref="y", showarrow=False,
                font=dict(size=14, color="white" if norm[i][j] > 0.4 else "#c8d8ff"),
            ))
    fig = go.Figure(go.Heatmap(
        z=norm, x=labels, y=labels,
        colorscale=[[0, "#0a0e27"], [0.5, "#1a3a8a"], [1.0, "#4F8BF9"]],
        showscale=True,
        colorbar=dict(title="Rate", thickness=12, tickfont=dict(color="#b0c4ee")),
    ))
    dark_fig(fig, f"Confusion Matrix — {model_name}", 340)
    fig.update_layout(annotations=ann, xaxis_title="Predicted", yaxis_title="Actual",
                      xaxis=dict(**GRID_STYLE), yaxis=dict(**GRID_STYLE))
    return fig


def make_roc_curves(curves):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                              line=dict(color="rgba(150,150,150,0.4)", dash="dash", width=1.5)))
    palette = ["#4F8BF9", "#7C4DFF", "#ff4444", "#69f0ae", "#ffd700", "#00d4ff"]
    for i, (model, data) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"], mode="lines",
            name=f"{model} (AUC={data['auc']:.4f})",
            line=dict(color=palette[i % len(palette)],
                      width=3 if "World Model" in model else 1.8,
                      dash="solid" if "World Model" in model else "dot"),
        ))
    dark_fig(fig, "ROC Curves — All Models", 440)
    fig.update_layout(
        xaxis=dict(title="FPR", range=[0, 1], **GRID_STYLE),
        yaxis=dict(title="TPR", range=[0, 1.02], **GRID_STYLE),
        legend=dict(x=0.55, y=0.1, bgcolor="rgba(10,14,39,0.9)",
                    bordercolor="rgba(79,139,249,0.3)", borderwidth=1),
    )
    return fig


def make_benchmark_bar(df, metric):
    sd = df.sort_values(metric, ascending=True)
    colors = ["#4F8BF9" if "World Model" in m else
              "#00d4ff" if m in ["Random Forest", "XGBoost"] else
              "rgba(120,130,180,0.6)" for m in sd["Model"]]
    fig = go.Figure(go.Bar(
        y=sd["Model"], x=sd[metric], orientation="h",
        marker=dict(color=colors, line_width=0),
        text=[f"{v:.4f}" for v in sd[metric]], textposition="outside",
        textfont=dict(color="#c8d8ff", size=12),
    ))
    dark_fig(fig, f"Benchmark — {metric}", 340)
    fig.update_layout(xaxis=dict(title=metric, range=[0, 1.05], **GRID_STYLE),
                      showlegend=False, yaxis=dict(**GRID_STYLE, tickfont=dict(color="#b0c4ee")))
    return fig


def make_pie(counts):
    labels = list(counts.keys())
    values = list(counts.values())
    palette = ["#4F8BF9", "#ff4444", "#ffd700", "#69f0ae", "#CE93D8",
               "#FF8A65", "#4FC3F7", "#F06292", "#00d4ff", "#7C4DFF"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.5,
        marker=dict(colors=palette[:len(labels)], line=dict(color="#0a0e27", width=2)),
        textinfo="percent+label", textfont=dict(size=10, color="#c8d8ff"),
        hovertemplate="<b>%{label}</b><br>%{value} flows<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**DARK_LAYOUT, height=360,
                      title=dict(text="Traffic Label Distribution", font=dict(color="#c8d8ff"), x=0.01),
                      legend=dict(font=dict(color="#b0c4ee", size=10)))
    return fig


def make_mitre_radar(stage_probs):
    cats = list(stage_probs.keys())
    vals = list(stage_probs.values())
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(79,139,249,0.15)",
        line=dict(color="#4F8BF9", width=2.5),
        marker=dict(size=7, color="#7C4DFF"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(10,14,39,0.8)",
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#7090c0", size=9),
                            gridcolor="rgba(79,139,249,0.15)"),
            angularaxis=dict(tickfont=dict(color="#c8d8ff", size=11),
                             gridcolor="rgba(79,139,249,0.15)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Stage Probability Distribution", font=dict(color="#c8d8ff", size=14), x=0.01),
        font=dict(family="Inter"), height=400,
        margin=dict(l=60, r=60, t=60, b=30),
    )
    return fig


def make_3d_scatter(df, n=300):
    pdf = df.sample(min(n, len(df)), random_state=42)
    cx = "Flow Duration" if "Flow Duration" in df.columns else df.columns[0]
    cy = "Flow Byts/s" if "Flow Byts/s" in df.columns else df.columns[1]
    cz = "Tot Fwd Pkts" if "Tot Fwd Pkts" in df.columns else df.columns[2]
    if "Label" in df.columns:
        colors = ["#ff4444" if l != "Benign" else "#4F8BF9" for l in pdf["Label"]]
        text = pdf["Label"].tolist()
    else:
        colors = "#4F8BF9"
        text = ["Unknown"] * len(pdf)
    fig = go.Figure(go.Scatter3d(
        x=pdf[cx].clip(0, pdf[cx].quantile(0.99)),
        y=pdf[cy].clip(0, pdf[cy].quantile(0.99)),
        z=pdf[cz].clip(0, pdf[cz].quantile(0.99)),
        mode="markers",
        marker=dict(size=4, color=colors, opacity=0.8,
                    line=dict(color="rgba(255,255,255,0.1)", width=0.3)),
        text=text,
        hovertemplate=f"<b>%{{text}}</b><br>{cx}: %{{x:.1f}}<br>{cy}: %{{y:.1f}}<br>{cz}: %{{z:.1f}}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(10,14,39,0.8)",
            xaxis=dict(title=cx[:18], backgroundcolor="rgba(10,14,39,0.6)",
                       gridcolor="rgba(79,139,249,0.15)", tickfont=dict(color="#7090c0")),
            yaxis=dict(title=cy[:18], backgroundcolor="rgba(10,14,39,0.6)",
                       gridcolor="rgba(79,139,249,0.15)", tickfont=dict(color="#7090c0")),
            zaxis=dict(title=cz[:18], backgroundcolor="rgba(10,14,39,0.6)",
                       gridcolor="rgba(79,139,249,0.15)", tickfont=dict(color="#7090c0")),
        ),
        title=dict(text="3D Network Traffic Scatter", font=dict(color="#c8d8ff"), x=0.01),
        height=500, font=dict(family="Inter", color="#b0c4ee"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def make_benchmark_radar(metrics, models):
    cats = ["F1 Score", "Precision", "Recall", "Accuracy", "AUC-ROC"]
    palette = ["#4F8BF9", "#7C4DFF", "#ff4444", "#69f0ae", "#ffd700", "#00d4ff"]
    fig = go.Figure()
    for i, model in enumerate(models):
        m = metrics[model]
        vals = [m["F1 Score"], m["Precision"], m["Recall"], m["Accuracy"], m["AUC-ROC"]]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself",
            fillcolor=f"rgba({int(palette[i%len(palette)][1:3],16)},"
                      f"{int(palette[i%len(palette)][3:5],16)},"
                      f"{int(palette[i%len(palette)][5:7],16)},0.08)",
            line=dict(color=palette[i % len(palette)],
                      width=3 if "World Model" in model else 1.8),
            name=model,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(10,14,39,0.8)",
            radialaxis=dict(visible=True, range=[0.8, 1.0], tickfont=dict(color="#7090c0", size=9),
                            gridcolor="rgba(79,139,249,0.15)"),
            angularaxis=dict(tickfont=dict(color="#c8d8ff", size=12),
                             gridcolor="rgba(79,139,249,0.15)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Multi-Model Performance Radar", font=dict(color="#c8d8ff", size=14), x=0.01),
        font=dict(family="Inter"), height=440,
        legend=dict(x=1.05, bgcolor="rgba(10,14,39,0.9)",
                    bordercolor="rgba(79,139,249,0.3)", borderwidth=1,
                    font=dict(color="#b0c4ee")),
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════

def page_dashboard():
    page_header("📊", "Dashboard", "Real-time overview of network security posture and attack forecasting")

    ts = generate_time_series_data(60)
    df = generate_mock_traffic_data(300)
    stats = compute_flow_statistics(df)
    prob = float(ts["Attack Probability"].iloc[-1])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(metric_html("Total Flows", f"{stats['total_flows']:,}", "+12.4%", "#4F8BF9"), unsafe_allow_html=True)
    with c2: st.markdown(metric_html("Attack Flows", f"{stats['attack_flows']:,}", f"+{stats['attack_rate']}%", "#ff4444", "rgba(255,68,68,0.4)"), unsafe_allow_html=True)
    with c3: st.markdown(metric_html("Benign Flows", f"{stats['benign_flows']:,}", "-5.2%", "#69f0ae", "rgba(105,240,174,0.4)"), unsafe_allow_html=True)
    with c4: st.markdown(metric_html("Flow Rate", f"{stats['avg_flow_bytes_s']/1000:.1f}K/s", "+8.1%", "#ffd700", "rgba(255,215,0,0.4)"), unsafe_allow_html=True)
    with c5: st.markdown(metric_html("Avg Duration", f"{stats['avg_duration_ms']:.0f}ms", "-2.3%", "#00d4ff", "rgba(0,212,255,0.4)"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        card_open()
        section_header("Current Threat Level")
        st.plotly_chart(make_gauge(prob), use_container_width=True, key="d_gauge")
        ms = get_model_status()
        st.markdown(f"""
        <div style="display:flex;justify-content:space-around;margin-top:8px;padding-top:12px;
                    border-top:1px solid rgba(79,139,249,0.15);">
            <div style="text-align:center;">
                <div style="font-family:'Orbitron',monospace;font-size:0.95rem;font-weight:700;
                            color:#4F8BF9;text-shadow:0 0 10px rgba(79,139,249,0.5);">{ms['version']}</div>
                <div style="font-size:0.65rem;color:rgba(176,196,255,0.6);text-transform:uppercase;letter-spacing:0.5px;">Version</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'Orbitron',monospace;font-size:0.95rem;font-weight:700;
                            color:#69f0ae;text-shadow:0 0 10px rgba(105,240,174,0.5);">97.1%</div>
                <div style="font-size:0.65rem;color:rgba(176,196,255,0.6);text-transform:uppercase;letter-spacing:0.5px;">Accuracy</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'Orbitron',monospace;font-size:0.95rem;font-weight:700;
                            color:#ffd700;text-shadow:0 0 10px rgba(255,215,0,0.5);">12ms</div>
                <div style="font-size:0.65rem;color:rgba(176,196,255,0.6);text-transform:uppercase;letter-spacing:0.5px;">Latency</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        card_close()

    with col2:
        card_open()
        section_header("Live Attack Probability Timeline")
        st.plotly_chart(make_timeline(ts), use_container_width=True, key="d_tl")
        card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    col3, col4 = st.columns([2, 1])
    with col3:
        card_open()
        section_header("Network Flow Rate")
        st.plotly_chart(make_flow_chart(ts), use_container_width=True, key="d_flow")
        card_close()
    with col4:
        card_open()
        section_header("Attack Distribution")
        st.plotly_chart(make_pie(stats["label_distribution"]), use_container_width=True, key="d_pie")
        card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    card_open()
    section_header("Recent Security Alerts")
    alerts = get_recent_alerts(8)
    hcols = st.columns([1.2, 2.5, 1.8, 1.2, 1.2])
    for hc, h in zip(hcols, ["Time", "Alert Type", "Source IP", "Severity", "Confidence"]):
        hc.markdown(f"<div style='font-size:0.68rem;font-weight:700;color:rgba(176,196,255,0.5);text-transform:uppercase;letter-spacing:1px;padding:4px 0;'>{h}</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(79,139,249,0.3),transparent);margin:4px 0 8px 0;'></div>", unsafe_allow_html=True)
    for a in alerts:
        ac = st.columns([1.2, 2.5, 1.8, 1.2, 1.2])
        ac[0].markdown(f"<div style='font-size:0.8rem;color:rgba(176,196,255,0.5);font-family:monospace;'>{a['time']}</div>", unsafe_allow_html=True)
        ac[1].markdown(f"<div style='font-size:0.85rem;font-weight:500;color:#c8d8ff;'>{a['type']}</div>", unsafe_allow_html=True)
        ac[2].markdown(f"<div style='font-size:0.8rem;color:#4F8BF9;font-family:monospace;'>{a['source_ip']}</div>", unsafe_allow_html=True)
        ac[3].markdown(alert_badge(a["severity"]), unsafe_allow_html=True)
        cc = "#69f0ae" if a["confidence"] > 0.85 else "#ffd700"
        ac[4].markdown(f"<div style='font-size:0.85rem;font-weight:700;color:{cc};'>{a['confidence']:.0%}</div>", unsafe_allow_html=True)
    card_close()


def page_data_ingestion():
    page_header("📂", "Data Ingestion", "Upload PCAP or CSV files, extract CIC-IDS2018 features and explore traffic")

    col1, col2 = st.columns([1, 1])
    with col1:
        card_open()
        section_header("Upload Network Data")
        st.markdown("""
        <div style="background:rgba(79,139,249,0.05);border:2px dashed rgba(79,139,249,0.3);
                    border-radius:12px;padding:24px;text-align:center;margin-bottom:16px;">
            <div style="font-size:2.5rem;margin-bottom:8px;filter:drop-shadow(0 0 15px rgba(79,139,249,0.6));">📁</div>
            <div style="font-size:0.9rem;font-weight:600;color:#4F8BF9;">Drop your file here</div>
            <div style="font-size:0.75rem;color:rgba(176,196,255,0.5);margin-top:4px;">CSV · PCAP · PCAPNG</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("Select", type=["csv","pcap","pcapng"], label_visibility="collapsed")
        if st.button("🔄 Load Demo Dataset (CIC-IDS2018)", use_container_width=True):
            st.session_state["ingested_df"] = generate_mock_traffic_data(500)
            st.success("✅ CIC-IDS2018 demo dataset loaded — 500 flows")
        if uploaded:
            if uploaded.name.endswith(".csv"):
                try:
                    st.session_state["ingested_df"] = parse_uploaded_csv(uploaded)
                    st.success(f"✅ Loaded {len(st.session_state['ingested_df']):,} rows")
                except Exception as e:
                    st.error(f"Parse error: {e}")
            else:
                st.session_state["ingested_df"] = generate_mock_traffic_data(500)
                st.info("PCAP parsing requires Scapy — loaded synthetic demo instead.")
        card_close()

    with col2:
        card_open()
        section_header("Feature Extraction Pipeline")
        features_status = [
            ("✅", "Flow-level statistics (bytes, packets, duration)", True),
            ("✅", "Packet-level features (TTL, window, payload)", True),
            ("✅", "TCP flag counters (SYN, ACK, FIN, RST, PSH, URG)", True),
            ("✅", "IAT statistics (mean, std, max, min)", True),
            ("✅", "Bidirectional flow ratios", True),
            ("✅", "Window size & subflow features", True),
            ("✅", "Active / Idle time features", True),
            ("✅", "Label encoding (13 attack categories)", True),
            ("⏳", "MinMax normalization", False),
            ("⏳", "Sequence windowing (len=20)", False),
        ]
        for icon, feat, done in features_status:
            color = "#69f0ae" if done else "#ffd700"
            bg = "rgba(105,240,174,0.05)" if done else "rgba(255,215,0,0.05)"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;padding:7px 10px;"
                f"border-radius:8px;margin-bottom:3px;background:{bg};"
                f"border:1px solid {'rgba(105,240,174,0.15)' if done else 'rgba(255,215,0,0.1)'};"
                f"'><span style='font-size:0.85rem;'>{icon}</span>"
                f"<span style='font-size:0.82rem;color:#c8d8ff;flex:1;'>{feat}</span>"
                f"<span style='font-size:0.7rem;font-weight:700;color:{color};text-transform:uppercase;'>"
                f"{'Done' if done else 'Pending'}</span></div>",
                unsafe_allow_html=True
            )
        card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    df_show = st.session_state.get("ingested_df", generate_mock_traffic_data(100))

    card_open()
    section_header("Data Explorer")
    t1, t2, t3 = st.tabs(["📋  Raw Data", "📊  Statistics", "🌐  3D Scatter"])
    with t1:
        n = st.slider("Rows", 10, 100, 30, key="di_rows")
        st.dataframe(df_show.head(n), use_container_width=True, height=320)
        c1,c2,c3 = st.columns(3)
        c1.metric("Rows", f"{len(df_show):,}")
        c2.metric("Columns", len(df_show.columns))
        c3.metric("Memory", f"{df_show.memory_usage(deep=True).sum()/1024:.1f} KB")
    with t2:
        num = df_show.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(df_show[num].describe().round(3), use_container_width=True, height=320)
    with t3:
        st.plotly_chart(make_3d_scatter(df_show), use_container_width=True, key="di_3d")
    card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    card_open()
    section_header("Extracted Feature Matrix (Top 16 Features)")
    st.dataframe(extract_features(df_show).head(20), use_container_width=True, height=280)
    card_close()


def page_model_training():
    page_header("🧠", "World Model Training", "Configure, train, and monitor LSTM / Transformer world model")

    col1, col2 = st.columns([1, 1])
    with col1:
        card_open()
        section_header("Training Configuration")
        arch = st.selectbox("Architecture", list(MODEL_ARCHITECTURES.keys()), key="t_arch")
        epochs = st.slider("Epochs", 10, 100, 50, key="t_ep")
        batch = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
        lr = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3,
                              format_func=lambda x: f"{x:.0e}")
        seq = st.slider("Sequence Length", 5, 50, 20)
        k = st.slider("K-step Horizon", 1, 20, 10)
        drop = st.slider("Dropout", 0.0, 0.6, 0.3, 0.05)
        es = st.checkbox("Early Stopping (patience=5)", value=True)

        if st.button("🚀 Start Training", use_container_width=True):
            pb = st.progress(0)
            st_txt = st.empty()
            for i in range(epochs):
                pb.progress((i+1)/epochs)
                st_txt.markdown(
                    f"<div style='font-size:0.82rem;color:#4F8BF9;font-family:monospace;'>"
                    f"Epoch {i+1:03d}/{epochs} &nbsp;│&nbsp; "
                    f"loss: <span style='color:#ffd700;'>{0.72*np.exp(-0.065*(i+1))+0.08:.4f}</span> &nbsp;│&nbsp; "
                    f"val_loss: <span style='color:#ff4444;'>{0.75*np.exp(-0.058*(i+1))+0.10:.4f}</span>"
                    f"</div>", unsafe_allow_html=True)
                time.sleep(0.03)
            st.session_state["trained_model"] = arch
            st_txt.markdown("<div style='color:#69f0ae;font-weight:700;font-size:0.9rem;'>✅ Training complete!</div>", unsafe_allow_html=True)
        card_close()

    with col2:
        card_open()
        section_header("Model Architecture")
        arch_data = MODEL_ARCHITECTURES.get(arch, list(MODEL_ARCHITECTURES.values())[0])
        for i, layer in enumerate(arch_data["layers"]):
            is_lstm = "LSTM" in layer["name"] or "Attention" in layer["name"]
            bg = "rgba(79,139,249,0.1)" if is_lstm else "rgba(255,255,255,0.02)"
            border = "rgba(79,139,249,0.4)" if is_lstm else "rgba(79,139,249,0.1)"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:8px 12px;border-radius:8px;margin-bottom:4px;"
                f"background:{bg};border:1px solid {border};'>"
                f"<span style='font-size:0.82rem;color:#c8d8ff;font-weight:500;'>{i+1}. {layer['name']}</span>"
                f"<span style='font-size:0.75rem;color:#4F8BF9;'>{layer['units']}</span>"
                f"<span style='font-size:0.72rem;color:rgba(176,196,255,0.5);'>{layer['activation']}</span>"
                f"</div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        mc1,mc2,mc3 = st.columns(3)
        mc1.metric("Parameters", arch_data["params"])
        mc2.metric("Optimizer", arch_data["optimizer"].split(" ")[0])
        mc3.metric("Loss", "Binary CE")
        card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    card_open()
    section_header("Training Curves")
    arch_key = "LSTM" if "LSTM" in arch else "Transformer"
    curves_df = generate_training_curves(epochs=epochs, model_type=arch_key)
    st.plotly_chart(make_training_curves(curves_df), use_container_width=True, key="tc")
    perf = get_model_performance_metrics(arch)
    cols = st.columns(6)
    for col, (lbl, key) in zip(cols, [("Accuracy","accuracy"),("Precision","precision"),
                                       ("Recall","recall"),("F1","f1_score"),("FPR","fpr"),("AUC","auc_roc")]):
        col.metric(lbl, f"{perf[key]:.4f}")
    card_close()


def page_attack_prediction():
    page_header("🔮", "Attack Prediction", "K-step forward simulation — predict infiltration before it completes")

    col1, col2 = st.columns([1, 1])
    with col1:
        card_open()
        section_header("Forecast Configuration")
        k = st.slider("Forecast Horizon (K steps)", 3, 20, 10, key="p_k")
        base = st.slider("Current Base Probability", 0.1, 0.9, 0.45, 0.05, key="p_base")
        st.selectbox("Attack Type Filter", ["All"] + ATTACK_LABELS[1:], key="p_at")

        if st.button("▶ Run Forecast", use_container_width=True):
            with st.spinner("Running K-step world model simulation..."):
                time.sleep(0.6)
            st.success("✅ Forecast complete")

        fdf = generate_k_step_forecast(k=k, base_prob=base)

        section_header("Step-by-Step Predictions")
        disp = fdf[["Step","Attack Probability","Predicted Stage"]].copy()
        disp["Attack Probability"] = disp["Attack Probability"].apply(lambda x: f"{x:.4f}")

        stage_c = {"Reconnaissance":"#4FC3F7","Initial Access":"#81C784","Execution":"#FFB74D",
                   "Lateral Movement":"#FF8A65","Command & Control":"#CE93D8","Exfiltration":"#ff4444"}
        for _, row in disp.iterrows():
            sc = stage_c.get(row["Predicted Stage"], "#c8d8ff")
            prob_val = float(row["Attack Probability"])
            pc = "#ff4444" if prob_val > 0.7 else "#ffd700" if prob_val > 0.5 else "#69f0ae"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;padding:6px 10px;"
                f"border-radius:8px;margin-bottom:3px;background:rgba(79,139,249,0.04);"
                f"border:1px solid rgba(79,139,249,0.1);'>"
                f"<span style='font-size:0.8rem;color:rgba(176,196,255,0.5);font-family:monospace;"
                f"min-width:50px;'>t+{row['Step']}</span>"
                f"<div style='width:8px;height:8px;border-radius:50%;background:{sc};"
                f"box-shadow:0 0 6px {sc};'></div>"
                f"<span style='font-size:0.8rem;color:{sc};flex:1;'>{row['Predicted Stage']}</span>"
                f"<span style='font-size:0.82rem;font-weight:700;color:{pc};"
                f"font-family:monospace;'>{row['Attack Probability']}</span>"
                f"</div>", unsafe_allow_html=True)
        card_close()

    with col2:
        card_open()
        section_header(f"Step t+{k} Probability")
        fdf2 = generate_k_step_forecast(k=k, base_prob=base)
        st.plotly_chart(make_gauge(float(fdf2["Attack Probability"].iloc[-1]), f"Step t+{k}"),
                        use_container_width=True, key="p_gauge")
        section_header("Stage Frequency")
        for stage, cnt in fdf2["Predicted Stage"].value_counts().items():
            sc = stage_c.get(stage, "#4F8BF9") if 'stage_c' in dir() else "#4F8BF9"
            stage_colors2 = {"Reconnaissance":"#4FC3F7","Initial Access":"#81C784","Execution":"#FFB74D",
                             "Lateral Movement":"#FF8A65","Command & Control":"#CE93D8","Exfiltration":"#ff4444"}
            cc = stage_colors2.get(stage, "#4F8BF9")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;padding:6px 0;"
                f"border-bottom:1px solid rgba(79,139,249,0.08);'>"
                f"<div style='width:10px;height:10px;border-radius:50%;background:{cc};"
                f"box-shadow:0 0 6px {cc};'></div>"
                f"<span style='font-size:0.84rem;color:#c8d8ff;flex:1;'>{stage}</span>"
                f"<span style='font-weight:700;color:{cc};font-size:0.9rem;'>{cnt} steps</span>"
                f"</div>", unsafe_allow_html=True)
        card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    card_open()
    section_header("K-Step Forecast Chart")
    st.plotly_chart(make_kstep(fdf2), use_container_width=True, key="p_kstep")
    card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    card_open()
    section_header("Infiltration Summary")
    ic1,ic2,ic3,ic4 = st.columns(4)
    ic1.metric("Min Probability", f"{fdf2['Attack Probability'].min():.4f}")
    ic2.metric("Max Probability", f"{fdf2['Attack Probability'].max():.4f}")
    ic3.metric("Mean Probability", f"{fdf2['Attack Probability'].mean():.4f}")
    ic4.metric("Steps Above 0.7", int((fdf2["Attack Probability"]>0.7).sum()))
    card_close()


def page_mitre_attack():
    page_header("🗺️", "MITRE ATT&CK Mapping", "Kill chain progression, technique mapping and attack stage classification")

    col1, col2 = st.columns([1, 1])
    with col1:
        card_open()
        section_header("Kill Chain Progression")
        stage_sel = st.selectbox("Predicted Stage", [s["name"] for s in MITRE_STAGES], index=3)
        chain = get_kill_chain_progression(stage_sel)
        for s in chain:
            icon = "✅" if s["status"]=="completed" else ("🔴" if s["status"]=="active" else "⬜")
            status_c = "#69f0ae" if s["status"]=="completed" else ("#ff4444" if s["status"]=="active" else "rgba(176,196,255,0.3)")
            st.markdown(
                f"<div class='kc-stage kc-{s['status']}'>"
                f"<div style='display:flex;align-items:center;gap:12px;'>"
                f"<span style='font-size:1.2rem;filter:drop-shadow(0 0 8px {status_c});'>{s['icon']}</span>"
                f"<div style='flex:1;'>"
                f"<div style='font-size:0.88rem;font-weight:600;color:#c8d8ff;'>{icon} {s['name']}</div>"
                f"<div style='font-size:0.72rem;color:rgba(176,196,255,0.55);margin-top:2px;'>{s['description'][:60]}…</div>"
                f"</div>"
                f"<span style='font-size:0.7rem;font-weight:700;color:{status_c};"
                f"text-transform:uppercase;letter-spacing:0.5px;'>{s['status']}</span>"
                f"</div></div>", unsafe_allow_html=True)
        card_close()

    with col2:
        card_open()
        section_header("Stage Probability Radar")
        sp = generate_stage_probability_vector(stage_sel)
        st.plotly_chart(make_mitre_radar(sp), use_container_width=True, key="m_radar")
        card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns([1, 1])
    with col3:
        card_open()
        section_header(f"ATT&CK Techniques — {stage_sel}")
        tdf = get_technique_details_for_stage(stage_sel)
        st.dataframe(tdf, use_container_width=True, hide_index=True)
        card_close()

    with col4:
        card_open()
        section_header("Network Indicators of Compromise")
        stage_data = next(s for s in MITRE_STAGES if s["name"]==stage_sel)
        for ind in stage_data["network_indicators"]:
            st.markdown(
                f"<div style='padding:7px 0;border-bottom:1px solid rgba(79,139,249,0.08);"
                f"display:flex;align-items:center;gap:10px;'>"
                f"<span style='color:#ff4444;font-size:0.9rem;'>⚠</span>"
                f"<span style='font-size:0.84rem;color:#c8d8ff;'>{ind}</span></div>",
                unsafe_allow_html=True)
        section_header("Detection Features")
        for feat in stage_data["detection_features"]:
            st.markdown(
                f"<span style='display:inline-block;background:rgba(79,139,249,0.15);"
                f"border:1px solid rgba(79,139,249,0.4);border-radius:20px;"
                f"padding:3px 14px;margin:3px;font-size:0.78rem;color:#4F8BF9;"
                f"font-weight:600;box-shadow:0 0 8px rgba(79,139,249,0.2);'>{feat}</span>",
                unsafe_allow_html=True)
        card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    card_open()
    section_header("MITRE ATT&CK Detection Heatmap")
    hdf = get_mitre_heatmap_data()
    pivot = hdf.pivot(index="Stage", columns="Technique", values="Detection Count").fillna(0)
    fig_h = px.imshow(pivot,
                      color_continuous_scale=[[0,"#0a0e27"],[0.4,"#1a3a8a"],[0.7,"#4F8BF9"],[1,"#7C4DFF"]],
                      aspect="auto")
    fig_h.update_layout(**DARK_LAYOUT, height=300,
                        margin=dict(l=10,r=10,t=20,b=60),
                        coloraxis_colorbar=dict(tickfont=dict(color="#b0c4ee")))
    fig_h.update_xaxes(tickfont=dict(size=9,color="#7090c0"), **GRID_STYLE)
    fig_h.update_yaxes(tickfont=dict(color="#b0c4ee"), **GRID_STYLE)
    st.plotly_chart(fig_h, use_container_width=True, key="m_heat")
    card_close()


def page_explainability():
    page_header("💡", "Explainability", "SHAP values, attention weights and feature attribution analysis")

    t1, t2, t3 = st.tabs(["🎯  SHAP Analysis", "📊  Feature Importance", "🔍  Attention Weights"])

    with t1:
        card_open()
        section_header("SHAP Value Distribution")
        n = st.slider("Samples", 10, 50, 20, key="x_n")
        shap_df = get_shap_values(n)
        # Beeswarm
        top_f = shap_df.abs().mean().sort_values(ascending=False).head(12).index.tolist()
        fig_bee = go.Figure()
        for i, feat in enumerate(top_f):
            vals = shap_df[feat].values
            y_j = np.random.normal(i, 0.15, len(vals))
            colors = ["#ff4444" if v > 0 else "#4F8BF9" for v in vals]
            fig_bee.add_trace(go.Scatter(x=vals, y=y_j, mode="markers", name=feat,
                                          marker=dict(color=colors, size=6, opacity=0.75),
                                          showlegend=False))
        dark_fig(fig_bee, "SHAP Beeswarm Plot", 440)
        fig_bee.update_layout(
            xaxis=dict(title="SHAP Value (impact on output)", **GRID_STYLE),
            yaxis=dict(tickvals=list(range(len(top_f))), ticktext=top_f, **GRID_STYLE),
        )
        st.plotly_chart(fig_bee, use_container_width=True, key="x_bee")

        section_header("Mean |SHAP| Values")
        ms = shap_df.abs().mean().sort_values(ascending=False).head(10)
        fig_ms = go.Figure(go.Bar(
            x=ms.values, y=ms.index, orientation="h",
            marker=dict(color=ms.values, colorscale=[[0,"#1a3a8a"],[1,"#7C4DFF"]], line_width=0),
            text=[f"{v:.3f}" for v in ms.values], textposition="outside",
            textfont=dict(color="#c8d8ff"),
        ))
        dark_fig(fig_ms, "", 340)
        fig_ms.update_layout(showlegend=False, coloraxis_showscale=False,
                              xaxis=dict(title="Mean |SHAP|", **GRID_STYLE),
                              yaxis=dict(**GRID_STYLE, tickfont=dict(color="#b0c4ee")))
        st.plotly_chart(fig_ms, use_container_width=True, key="x_ms")
        card_close()

    with t2:
        card_open()
        section_header("Global Feature Importance")
        top_n = st.slider("Top N Features", 5, 16, 16, key="x_fn")
        st.plotly_chart(make_feature_importance(FEATURE_IMPORTANCE_DATA, top_n), use_container_width=True, key="x_fi")
        fi_df = pd.DataFrame(list(FEATURE_IMPORTANCE_DATA.items()), columns=["Feature", "Importance"]) \
                  .sort_values("Importance", ascending=False).reset_index(drop=True)
        fi_df.index += 1
        fi_df["Importance"] = fi_df["Importance"].apply(lambda x: f"{x:.3f}")
        st.dataframe(fi_df, use_container_width=True, height=360)
        card_close()

    with t3:
        card_open()
        section_header("Transformer Self-Attention Heatmap")
        sl = st.slider("Sequence Length", 5, 20, 15, key="x_sl")
        attn = get_attention_weights(sl)
        st.plotly_chart(make_attention_heatmap(attn), use_container_width=True, key="x_attn")
        st.info("Bright cells = high attention. The model attends more to recent timesteps when predicting attack progression. Diagonal patterns indicate temporal locality in learned features.")
        card_close()


def page_benchmark():
    page_header("📈", "Benchmark", "World Model vs Logistic Regression vs ensemble baselines — full comparison")

    bdf = get_benchmark_dataframe()
    card_open()
    section_header("Model Performance Table")
    def highlight_best(s):
        if s.name in ["F1 Score","Precision","Recall","Accuracy","AUC-ROC"]:
            return ["background:rgba(79,139,249,0.2);color:#7EB8FF;font-weight:700;" if v==s.max() else "" for v in s]
        elif s.name == "FPR":
            return ["background:rgba(105,240,174,0.15);color:#69f0ae;font-weight:700;" if v==s.min() else "" for v in s]
        return ["" for _ in s]
    styled = bdf.style.apply(highlight_best, axis=0).format({
        "F1 Score":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}",
        "Accuracy":"{:.4f}","FPR":"{:.4f}","AUC-ROC":"{:.4f}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True, height=260)
    card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    bc1, bc2 = st.columns(2)
    with bc1:
        card_open()
        m1 = st.selectbox("Metric", ["F1 Score","Precision","Recall","Accuracy","AUC-ROC"], key="bm1")
        st.plotly_chart(make_benchmark_bar(bdf, m1), use_container_width=True, key="b_b1")
        card_close()
    with bc2:
        card_open()
        m2 = st.selectbox("Metric", ["AUC-ROC","FPR","Accuracy","F1 Score"], key="bm2")
        st.plotly_chart(make_benchmark_bar(bdf, m2), use_container_width=True, key="b_b2")
        card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    col_roc, col_cm = st.columns([3,2])
    with col_roc:
        card_open()
        section_header("ROC Curves")
        st.plotly_chart(make_roc_curves(get_roc_curve_data()), use_container_width=True, key="b_roc")
        card_close()
    with col_cm:
        card_open()
        section_header("Confusion Matrix")
        cm_sel = st.selectbox("Model", ["World Model (LSTM)","World Model (Transformer)","Logistic Regression"], key="b_cm")
        cm = get_confusion_matrix(cm_sel)
        st.plotly_chart(make_confusion_matrix(cm, cm_sel), use_container_width=True, key="b_cmf")
        p = get_model_performance_metrics(cm_sel)
        cc1,cc2 = st.columns(2)
        cc1.metric("Precision", f"{p['precision']:.4f}")
        cc2.metric("Recall", f"{p['recall']:.4f}")
        cc1.metric("F1 Score", f"{p['f1_score']:.4f}")
        cc2.metric("FPR", f"{p['fpr']:.4f}")
        card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    card_open()
    section_header("Multi-Model Radar Comparison")
    sel_models = st.multiselect("Select models", BENCHMARK_MODELS,
                                default=["World Model (LSTM)","World Model (Transformer)","Logistic Regression","XGBoost"])
    if len(sel_models) >= 2:
        st.plotly_chart(make_benchmark_radar(BENCHMARK_METRICS, sel_models), use_container_width=True, key="b_rad")
    else:
        st.info("Select at least 2 models for radar comparison.")
    card_close()


def page_settings():
    page_header("⚙️", "Settings", "Model configuration, detection thresholds, and system settings")

    col1, col2 = st.columns(2)
    with col1:
        card_open()
        section_header("Model Configuration")
        st.selectbox("Active Model", list(MODEL_ARCHITECTURES.keys()))
        st.number_input("Sequence Window Length", 5, 100, 20)
        st.number_input("K-step Forecast Horizon", 1, 30, 10)
        st.number_input("LSTM Hidden Units", 64, 512, 256, step=64)
        st.number_input("LSTM Layers", 1, 6, 2)
        st.select_slider("Dropout Rate", [0.0,0.1,0.2,0.3,0.4,0.5], value=0.3)
        st.selectbox("Optimizer", ["Adam","AdamW","SGD","RMSprop"])
        if st.button("💾 Save Model Config", use_container_width=True):
            st.success("✅ Model configuration saved")
        card_close()

    with col2:
        card_open()
        section_header("Detection Thresholds")
        st.slider("Alert Threshold", 0.3, 0.95, 0.70, 0.05)
        st.slider("Critical Threshold", 0.5, 0.99, 0.85, 0.05)
        st.slider("Anomaly Score Threshold", 0.2, 0.95, 0.60, 0.05)
        st.slider("Min Prediction Confidence", 0.5, 0.99, 0.65, 0.05)
        st.markdown("<hr>", unsafe_allow_html=True)
        section_header("Alert Settings")
        st.checkbox("Enable Email Alerts")
        st.checkbox("Enable Webhook Notifications")
        st.checkbox("Auto-block Suspicious IPs")
        st.selectbox("Aggregation Window", ["1 min","5 min","15 min","1 hour"], index=1)
        if st.button("💾 Save Threshold Config", use_container_width=True):
            st.success("✅ Threshold configuration saved")
        card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        card_open()
        section_header("Data Pipeline")
        st.selectbox("Input Source", ["File Upload","Live PCAP","Network Tap","SIEM Integration"])
        st.number_input("Max Flow Buffer", 1000, 100000, 10000, step=1000)
        st.selectbox("Normalization", ["MinMax","Standard","Robust","None"])
        st.checkbox("Apply PCA Reduction")
        st.number_input("Train/Val/Test Split (%)", 60, 80, 70)
        if st.button("💾 Save Data Config", use_container_width=True):
            st.success("✅ Data configuration saved")
        card_close()

    with col4:
        card_open()
        section_header("System Information")
        ms = get_model_status()
        info = [("App Version","1.0.0"),("Model",ms["model_name"]),("Version",ms["version"]),
                ("Dataset",ms["dataset"]),("Parameters",ms["total_params"]),
                ("Last Trained",ms["last_trained"]),("Epochs",str(ms["epochs_trained"])),("Device",ms["device"])]
        for k, v in info:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:8px 0;"
                f"border-bottom:1px solid rgba(79,139,249,0.08);'>"
                f"<span style='font-size:0.82rem;color:rgba(176,196,255,0.55);'>{k}</span>"
                f"<span style='font-size:0.82rem;font-weight:600;color:#c8d8ff;'>{v}</span>"
                f"</div>", unsafe_allow_html=True)
        card_close()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    inject_css()

    if "ingested_df" not in st.session_state:
        st.session_state["ingested_df"] = generate_mock_traffic_data(300)

    page = render_sidebar()

    dispatch = {
        "Dashboard": page_dashboard,
        "Data Ingestion": page_data_ingestion,
        "Model Training": page_model_training,
        "Attack Prediction": page_attack_prediction,
        "MITRE ATT&CK": page_mitre_attack,
        "Explainability": page_explainability,
        "Benchmark": page_benchmark,
        "Settings": page_settings,
    }
    dispatch.get(page, page_dashboard)()


if __name__ == "__main__":
    main()
