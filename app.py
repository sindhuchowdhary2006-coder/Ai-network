"""
CyberSentinel AI — Network Attack Forecasting System
Complete fixed version — all errors resolved
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="CyberSentinel AI",
    page_icon="🔐",
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
    get_technique_details_for_stage, get_mitre_heatmap_data,
)


# ══════════════════════════════════════════════════════════════
#  CHART HELPERS  (no **DARK_LAYOUT + other kwargs together)
# ══════════════════════════════════════════════════════════════
_PAPER  = "rgba(0,0,0,0)"
_PLOT   = "rgba(8,12,35,0.7)"
_FONT   = dict(family="Inter, sans-serif", color="#c8d8ff")
_HOVER  = dict(bgcolor="#0d1235", font_size=13,
               font_family="Inter", font_color="#e0eaff",
               bordercolor="rgba(79,139,249,0.5)")
_LEGEND = dict(bgcolor="rgba(8,12,35,0.9)",
               bordercolor="rgba(79,139,249,0.3)", borderwidth=1,
               font=dict(color="#b0c4ee"))
_GRID   = dict(showgrid=True, gridcolor="rgba(79,139,249,0.1)",
               zeroline=False, linecolor="rgba(79,139,249,0.2)",
               tickfont=dict(color="#6a8cc0"))


def _base_layout(height=380, title=""):
    layout = dict(
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
        font=_FONT,
        margin=dict(l=40, r=40, t=50, b=40),
        hoverlabel=_HOVER,
        legend=_LEGEND,
        height=height,
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=15, color="#c8d8ff"), x=0.01)
    return layout


def _apply(fig, height=380, title=""):
    fig.update_layout(**_base_layout(height, title))
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Orbitron:wght@400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family:'Inter',sans-serif !important; }

    /* ── BACKGROUND ── */
    .stApp {
        background:
            radial-gradient(ellipse at 15% 40%, rgba(79,139,249,0.18) 0%, transparent 55%),
            radial-gradient(ellipse at 85% 15%, rgba(124,77,255,0.14) 0%, transparent 45%),
            radial-gradient(ellipse at 50% 85%, rgba(0,212,255,0.10) 0%, transparent 45%),
            radial-gradient(ellipse at 90% 80%, rgba(229,57,53,0.08) 0%, transparent 35%),
            #030712 !important;
        background-attachment: fixed !important;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#060b1f 0%,#090f28 50%,#060b1f 100%) !important;
        border-right: 1px solid rgba(79,139,249,0.25) !important;
        box-shadow: 4px 0 40px rgba(0,0,0,0.6) !important;
    }
    [data-testid="stSidebar"] > div { background:transparent !important; }
    [data-testid="stSidebar"] * { color:#c8d8ff !important; }

    /* hide the "nav" label */
    [data-testid="stSidebar"] .stRadio > label { display:none !important; }

    /* ── NAV ITEMS ── */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        display:flex; flex-direction:column; gap:2px;
    }
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
        background:rgba(255,255,255,0.02) !important;
        border:1px solid rgba(79,139,249,0.08) !important;
        border-radius:10px !important;
        padding:10px 14px !important;
        cursor:pointer !important;
        transition:all .2s ease !important;
        display:flex !important; align-items:center !important;
        font-size:.87rem !important; font-weight:500 !important;
        color:rgba(180,200,255,.65) !important;
        margin:1px 0 !important; width:100% !important;
    }
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
        background:rgba(79,139,249,0.12) !important;
        border-color:rgba(79,139,249,0.35) !important;
        color:#fff !important;
        transform:translateX(3px) !important;
    }
    [data-testid="stSidebar"] .stRadio label[aria-checked="true"] {
        background:linear-gradient(135deg,rgba(79,139,249,.28),rgba(124,77,255,.18)) !important;
        border-color:rgba(79,139,249,.6) !important;
        color:#fff !important;
        box-shadow:0 0 18px rgba(79,139,249,.3),inset 0 0 18px rgba(79,139,249,.04) !important;
    }
    /* hide radio circle */
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] > div:first-child { display:none !important; }

    /* ── MAIN CONTENT ── */
    .main .block-container {
        padding:1.5rem 2rem 3rem !important;
        max-width:100% !important;
    }

    /* ── PAGE HEADER ── */
    .page-header {
        background:linear-gradient(135deg,rgba(79,139,249,.18) 0%,rgba(124,77,255,.12) 60%,rgba(0,212,255,.08) 100%);
        border:1px solid rgba(79,139,249,.35);
        border-radius:18px; padding:26px 30px; margin-bottom:22px;
        position:relative; overflow:hidden;
        box-shadow:0 8px 40px rgba(79,139,249,.18),inset 0 0 60px rgba(79,139,249,.04);
    }
    .page-header::after {
        content:''; position:absolute; top:-60%; right:-15%;
        width:360px; height:360px;
        background:radial-gradient(circle,rgba(124,77,255,.12) 0%,transparent 70%);
        pointer-events:none;
    }
    .page-header h1 {
        font-family:'Orbitron',monospace !important;
        font-size:1.45rem !important; font-weight:800 !important;
        color:#ffffff !important; margin:0 !important;
        text-shadow:0 0 24px rgba(79,139,249,.9) !important;
        letter-spacing:.5px !important;
    }
    .page-header p {
        font-size:.85rem !important; color:rgba(200,216,255,.75) !important;
        margin:7px 0 0 !important;
    }

    /* ── CARDS ── */
    .ng-card {
        background:linear-gradient(135deg,rgba(8,12,35,.97),rgba(11,16,46,.94));
        border:1px solid rgba(79,139,249,.18);
        border-radius:16px; padding:20px 22px; margin-bottom:14px;
        box-shadow:0 6px 30px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.04);
        position:relative; overflow:hidden;
        transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    }
    .ng-card::before {
        content:''; position:absolute; top:0; left:0; right:0; height:1px;
        background:linear-gradient(90deg,transparent,rgba(79,139,249,.5),transparent);
    }
    .ng-card:hover {
        transform:translateY(-2px);
        border-color:rgba(79,139,249,.4) !important;
        box-shadow:0 12px 40px rgba(0,0,0,.5),0 0 20px rgba(79,139,249,.12) !important;
    }

    /* ── METRIC CARDS ── */
    .metric-card {
        background:linear-gradient(135deg,rgba(8,12,35,.98),rgba(11,16,46,.95));
        border:1px solid rgba(79,139,249,.15);
        border-radius:14px; padding:18px 14px;
        text-align:center;
        box-shadow:0 4px 24px rgba(0,0,0,.35);
        position:relative; overflow:hidden;
        transition:transform .2s ease, box-shadow .2s ease;
        height:118px; display:flex; flex-direction:column;
        justify-content:center; align-items:center;
    }
    .metric-card:hover {
        transform:translateY(-3px);
        box-shadow:0 10px 36px rgba(0,0,0,.5),0 0 18px var(--gc,rgba(79,139,249,.3));
    }
    .metric-top-bar {
        position:absolute; top:0; left:0; right:0; height:2px;
        box-shadow:0 0 8px var(--gc,rgba(79,139,249,.5));
    }
    .metric-value {
        font-family:'Orbitron',monospace !important;
        font-size:1.75rem !important; font-weight:800 !important;
        line-height:1.1 !important;
    }
    .metric-label {
        font-size:.66rem !important; font-weight:600 !important;
        color:rgba(176,196,255,.55) !important; margin-top:4px !important;
        text-transform:uppercase !important; letter-spacing:1px !important;
    }
    .metric-delta { font-size:.7rem !important; margin-top:3px !important; font-weight:600 !important; }

    /* ── SECTION HEADER ── */
    .sh {
        font-size:.88rem; font-weight:700; color:#d0e0ff;
        border-left:3px solid #4F8BF9; padding-left:11px;
        margin:16px 0 12px; text-transform:uppercase; letter-spacing:.5px;
    }

    /* ── SEVERITY BADGES ── */
    .bc { background:rgba(229,57,53,.25);color:#ff6b6b;border:1px solid rgba(229,57,53,.5);
          border-radius:20px;padding:2px 11px;font-size:.7rem;font-weight:700;
          text-transform:uppercase;letter-spacing:.5px;box-shadow:0 0 8px rgba(229,57,53,.25); }
    .bh { background:rgba(255,143,0,.2);color:#ffb347;border:1px solid rgba(255,143,0,.4);
          border-radius:20px;padding:2px 11px;font-size:.7rem;font-weight:700;text-transform:uppercase; }
    .bm { background:rgba(255,215,0,.15);color:#ffd700;border:1px solid rgba(255,215,0,.35);
          border-radius:20px;padding:2px 11px;font-size:.7rem;font-weight:700;text-transform:uppercase; }
    .bl { background:rgba(67,160,71,.2);color:#69f0ae;border:1px solid rgba(67,160,71,.4);
          border-radius:20px;padding:2px 11px;font-size:.7rem;font-weight:700;text-transform:uppercase; }

    /* ── KILL CHAIN ── */
    .kc { border-radius:10px;padding:11px 14px;margin-bottom:6px;
          border:1px solid rgba(79,139,249,.1);background:rgba(255,255,255,.02);transition:all .2s; }
    .kc-completed { border-left:3px solid #43A047; opacity:.78; }
    .kc-active    { border-left:3px solid #E53935;background:rgba(229,57,53,.07);
                    border-color:rgba(229,57,53,.35);box-shadow:0 0 18px rgba(229,57,53,.25); }
    .kc-pending   { border-left:3px solid rgba(79,139,249,.2); opacity:.4; }

    /* ── BUTTONS ── */
    .stButton > button {
        background:linear-gradient(135deg,#4F8BF9,#7C4DFF) !important;
        color:#fff !important; border:none !important;
        border-radius:9px !important; padding:9px 22px !important;
        font-weight:700 !important; font-size:.85rem !important;
        letter-spacing:.5px !important; text-transform:uppercase !important;
        transition:all .25s ease !important;
        box-shadow:0 4px 18px rgba(79,139,249,.4) !important;
    }
    .stButton > button:hover {
        transform:translateY(-2px) !important;
        box-shadow:0 8px 28px rgba(79,139,249,.6) !important;
    }

    /* ── INPUTS / SELECT ── */
    .stSelectbox>div>div,
    .stNumberInput>div>div>input,
    .stTextInput>div>div>input {
        background:rgba(8,12,35,.85) !important;
        border:1px solid rgba(79,139,249,.25) !important;
        border-radius:8px !important; color:#c8d8ff !important;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        background:rgba(8,12,35,.8) !important;
        border-radius:11px !important; padding:4px !important;
        border:1px solid rgba(79,139,249,.18) !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius:8px !important; color:rgba(200,216,255,.55) !important;
        font-weight:500 !important; font-size:.84rem !important; padding:8px 16px !important;
    }
    .stTabs [aria-selected="true"] {
        background:linear-gradient(135deg,#4F8BF9,#7C4DFF) !important;
        color:#fff !important; box-shadow:0 4px 14px rgba(79,139,249,.4) !important;
    }

    /* ── PROGRESS ── */
    .stProgress>div>div>div {
        background:linear-gradient(90deg,#4F8BF9,#7C4DFF) !important;
        box-shadow:0 0 10px rgba(79,139,249,.5) !important;
    }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width:5px; height:5px; }
    ::-webkit-scrollbar-track { background:rgba(8,12,35,.5); }
    ::-webkit-scrollbar-thumb { background:linear-gradient(#4F8BF9,#7C4DFF); border-radius:10px; }

    /* ── HIDE STREAMLIT CLUTTER ── */
    #MainMenu,footer,.stDeployButton,[data-testid="stToolbar"] { display:none !important; }

    /* ── TEXT ── */
    h1,h2,h3,h4,h5,h6 { color:#e0eaff !important; }
    p,label { color:#b0c4ee !important; }
    .stMarkdown p { color:#b0c4ee !important; }
    [data-testid="stMetricValue"] {
        color:#7EB8FF !important;
        font-family:'Orbitron',monospace !important; font-weight:700 !important;
    }
    [data-testid="stMetricLabel"] {
        color:rgba(176,196,255,.65) !important; font-size:.72rem !important;
        text-transform:uppercase !important; letter-spacing:.5px !important;
    }
    hr { border:none !important; height:1px !important;
         background:linear-gradient(90deg,transparent,rgba(79,139,249,.4),transparent) !important;
         margin:14px 0 !important; }
    .stAlert,[data-baseweb="notification"] {
        background:rgba(79,139,249,.08) !important;
        border-left:3px solid #4F8BF9 !important;
        border-radius:8px !important; color:#c8d8ff !important;
    }
    .stCheckbox label span { color:#c8d8ff !important; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:24px 0 18px;">
            <div style="font-size:2.6rem;filter:drop-shadow(0 0 22px rgba(79,139,249,.9));">🔐</div>
            <div style="font-family:'Orbitron',monospace;font-size:1.05rem;font-weight:900;
                        color:#fff;letter-spacing:2px;margin-top:8px;
                        text-shadow:0 0 22px rgba(79,139,249,.9);">CYBERSENTINEL</div>
            <div style="font-family:'Orbitron',monospace;font-size:.58rem;font-weight:400;
                        color:rgba(176,196,255,.55);letter-spacing:3px;margin-top:3px;">AI · NETWORK DEFENSE</div>
        </div>
        <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(79,139,249,.45),transparent);
                    margin:0 0 14px;"></div>
        <div style="font-size:.62rem;color:rgba(176,196,255,.35);text-transform:uppercase;
                    letter-spacing:2px;padding:0 4px 8px;">Navigation</div>
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
        sel = st.radio("_", pages, label_visibility="collapsed")

        ms = get_model_status()
        st.markdown(f"""
        <div style="background:rgba(79,139,249,.07);border:1px solid rgba(79,139,249,.18);
                    border-radius:12px;padding:13px 15px;margin-top:18px;">
            <div style="font-size:.6rem;color:rgba(176,196,255,.45);text-transform:uppercase;
                        letter-spacing:1.5px;margin-bottom:7px;">⚡ Model Status</div>
            <div style="display:flex;align-items:center;gap:7px;margin-bottom:5px;">
                <div style="width:7px;height:7px;border-radius:50%;background:#43A047;
                             box-shadow:0 0 8px #43A047;flex-shrink:0;"></div>
                <span style="font-size:.8rem;font-weight:700;color:#69f0ae;">OPERATIONAL</span>
            </div>
            <div style="font-size:.72rem;color:rgba(176,196,255,.65);line-height:1.6;">
                {ms['model_name']}<br>
                <span style="color:#4F8BF9;font-weight:600;">{ms['version']}</span>
                &nbsp;·&nbsp;{ms['dataset']}
            </div>
        </div>
        <div style="text-align:center;margin-top:14px;font-size:.6rem;
                    color:rgba(176,196,255,.25);letter-spacing:1px;">
            SIH 2024 · NTRO CHALLENGE
        </div>
        """, unsafe_allow_html=True)

    return sel.split("  ", 1)[1].strip()


# ══════════════════════════════════════════════════════════════
#  REUSABLE UI BITS
# ══════════════════════════════════════════════════════════════
def ph(icon, title, sub):
    st.markdown(f"""
    <div class="page-header">
        <h1>{icon} {title}</h1><p>{sub}</p>
    </div>""", unsafe_allow_html=True)


def sh(txt):
    st.markdown(f'<div class="sh">{txt}</div>', unsafe_allow_html=True)


def co():  st.markdown('<div class="ng-card">', unsafe_allow_html=True)
def cc():  st.markdown('</div>', unsafe_allow_html=True)


def metric_card(label, value, delta="", color="#4F8BF9", glow="rgba(79,139,249,.4)"):
    dh = ""
    if delta:
        dc = "#69f0ae" if delta.startswith("+") else "#ff6b6b"
        dh = f'<div class="metric-delta" style="color:{dc};">{delta}</div>'
    return f"""
    <div class="metric-card" style="--gc:{glow};">
        <div class="metric-top-bar" style="background:{color};"></div>
        <div class="metric-value" style="color:{color};text-shadow:0 0 18px {glow};">{value}</div>
        <div class="metric-label">{label}</div>{dh}
    </div>"""


def badge(sev):
    cls = {"Critical":"bc","High":"bh","Medium":"bm","Low":"bl"}.get(sev,"bl")
    return f'<span class="{cls}">{sev}</span>'


# ══════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════
SC = {"Reconnaissance":"#4FC3F7","Initial Access":"#81C784","Execution":"#FFB74D",
      "Lateral Movement":"#FF8A65","Command & Control":"#CE93D8","Exfiltration":"#ff4444"}


def gauge(prob, title="Attack Probability"):
    pct = round(prob * 100, 1)
    if pct < 25:   c, lv = "#69f0ae", "LOW"
    elif pct < 50: c, lv = "#ffd700", "MODERATE"
    elif pct < 75: c, lv = "#ff9800", "HIGH"
    else:          c, lv = "#ff4444", "CRITICAL"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        title={"text": f"<span style='font-size:.85em;color:#c8d8ff'>{title}</span><br>"
                       f"<span style='font-size:.75em;color:{c};font-weight:700'>{lv}</span>",
               "font": {"size": 14}},
        number={"suffix": "%", "font": {"size": 40, "color": c, "family": "Orbitron"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a6fa5",
                     "tickfont": {"size": 10, "color": "#7090c0"}},
            "bar":  {"color": c, "thickness": 0.25},
            "bgcolor": "rgba(8,12,35,.8)",
            "borderwidth": 2, "bordercolor": "rgba(79,139,249,.25)",
            "steps": [
                {"range": [0, 25],   "color": "rgba(105,240,174,.05)"},
                {"range": [25, 50],  "color": "rgba(255,215,0,.05)"},
                {"range": [50, 75],  "color": "rgba(255,152,0,.05)"},
                {"range": [75, 100], "color": "rgba(255,68,68,.05)"},
            ],
            "threshold": {"line": {"color": "#ff4444", "width": 3},
                          "thickness": .8, "value": 70},
        },
    ))
    fig.update_layout(paper_bgcolor=_PAPER, height=300,
                      margin=dict(l=30,r=30,t=60,b=20),
                      font=dict(family="Inter",color="#b0c4ee"))
    return fig


def timeline_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Timestamp"], y=df["Attack Probability"], mode="lines",
        name="Attack Prob", line=dict(color="#ff4444", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,68,68,.08)"))
    fig.add_trace(go.Scatter(
        x=df["Timestamp"], y=df["Anomaly Score"], mode="lines",
        name="Anomaly Score", line=dict(color="#7C4DFF", width=1.8, dash="dot")))
    fig.add_hline(y=.7, line_dash="dash", line_color="rgba(255,68,68,.6)",
                  annotation_text="Threshold 0.70", annotation_font_color="#ff6b6b",
                  annotation_position="top right")
    fig.update_layout(**_base_layout(320, "Live Attack Probability Timeline"),
                      legend=dict(**_LEGEND, orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"))
    fig.update_xaxes(**_GRID, title_text="Time")
    fig.update_yaxes(**_GRID, range=[0,1], title_text="Probability")
    return fig


def flow_chart(df):
    fig = go.Figure(go.Bar(
        x=df["Timestamp"], y=df["Flow Rate (flows/min)"],
        marker=dict(color=df["Flow Rate (flows/min)"],
                    colorscale=[[0,"#122060"],[.5,"#4F8BF9"],[1,"#7C4DFF"]],
                    line_width=0),
        opacity=.9))
    fig.update_layout(**_base_layout(300, "Network Flow Rate (flows/min)"))
    fig.update_xaxes(**_GRID, title_text="Time")
    fig.update_yaxes(**_GRID, title_text="Flows/min")
    return fig


def kstep_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(df["Step"])+list(df["Step"])[::-1],
        y=list(df["Upper Bound"])+list(df["Lower Bound"])[::-1],
        fill="toself", fillcolor="rgba(79,139,249,.08)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
    fig.add_trace(go.Scatter(
        x=df["Step"], y=df["Attack Probability"],
        mode="lines+markers", name="Attack Prob",
        line=dict(color="#4F8BF9", width=3),
        marker=dict(size=9, color="#7C4DFF",
                    line=dict(color="#c8d8ff", width=2))))
    fig.add_hline(y=.7, line_dash="dash", line_color="rgba(255,68,68,.6)",
                  annotation_text="Alert Threshold", annotation_font_color="#ff6b6b")
    prev = None
    for _, row in df.iterrows():
        s = row["Predicted Stage"]
        if s != prev:
            fig.add_annotation(
                x=row["Step"], y=min(row["Attack Probability"]+.09,.97),
                text=s, showarrow=False,
                font=dict(size=8, color=SC.get(s,"#c8d8ff")),
                bgcolor="rgba(8,12,35,.85)",
                bordercolor=SC.get(s,"#4F8BF9"), borderwidth=1, borderpad=3)
            prev = s
    fig.update_layout(**_base_layout(400, "K-Step Forward Attack Probability Forecast"),
                      legend=dict(**_LEGEND, orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"))
    fig.update_xaxes(**_GRID, title_text="Forecast Step (t+k)", tickmode="linear")
    fig.update_yaxes(**_GRID, range=[0,1.05], title_text="Attack Probability")
    return fig


def training_chart(df):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Loss Curves", "Accuracy & F1"])
    fig.add_trace(go.Scatter(x=df["Epoch"],y=df["Train Loss"],  name="Train Loss",  line=dict(color="#4F8BF9",width=2.2)), row=1,col=1)
    fig.add_trace(go.Scatter(x=df["Epoch"],y=df["Val Loss"],    name="Val Loss",    line=dict(color="#ff4444",width=2.2,dash="dot")), row=1,col=1)
    fig.add_trace(go.Scatter(x=df["Epoch"],y=df["Train Accuracy"],name="Train Acc", line=dict(color="#69f0ae",width=2.2)), row=1,col=2)
    fig.add_trace(go.Scatter(x=df["Epoch"],y=df["Val Accuracy"],  name="Val Acc",   line=dict(color="#00d4ff",width=2.2,dash="dot")), row=1,col=2)
    fig.add_trace(go.Scatter(x=df["Epoch"],y=df["Train F1"],      name="Train F1",  line=dict(color="#ffd700",width=2,dash="dash")), row=1,col=2)
    fig.update_layout(
        paper_bgcolor=_PAPER, plot_bgcolor=_PLOT,
        font=_FONT, height=360,
        legend=dict(**_LEGEND, orientation="h", y=-.2, x=.5, xanchor="center"),
        margin=dict(l=40,r=40,t=50,b=40),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#c8d8ff"
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def feat_imp_chart(data, top_n=16):
    items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    feats = [x[0] for x in items]
    vals  = [x[1] for x in items]
    fig = go.Figure(go.Bar(
        x=vals[::-1], y=feats[::-1], orientation="h",
        marker=dict(color=vals[::-1],
                    colorscale=[[0,"#122060"],[.5,"#4F8BF9"],[1,"#7C4DFF"]],
                    line_width=0),
        text=[f"{v:.3f}" for v in vals[::-1]], textposition="outside",
        textfont=dict(color="#c8d8ff",size=11)))
    fig.update_layout(**_base_layout(440, "Global Feature Importance (SHAP-based)"))
    fig.update_xaxes(**_GRID, title_text="Importance Score")
    fig.update_yaxes(**_GRID, tickfont=dict(color="#b0c4ee",size=11))
    return fig


def attn_heatmap(attn):
    n = attn.shape[0]
    labels = [f"t-{n-1-i}" if i < n-1 else "t" for i in range(n)]
    fig = go.Figure(go.Heatmap(
        z=attn, x=labels, y=labels,
        colorscale=[[0,"#080c23"],[.35,"#1a3a8a"],[.7,"#4F8BF9"],[1,"#7C4DFF"]],
        showscale=True,
        colorbar=dict(title="Weight", thickness=12, tickfont=dict(color="#b0c4ee")),
        hovertemplate="Query:%{y}<br>Key:%{x}<br>Wt:%{z:.4f}<extra></extra>"))
    fig.update_layout(**_base_layout(420, "Transformer Attention Weights (Self-Attention)"))
    fig.update_xaxes(**_GRID, title_text="Key")
    fig.update_yaxes(**_GRID, title_text="Query")
    return fig


def conf_matrix(cm, name):
    labels = ["Benign", "Attack"]
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ann = [dict(text=f"<b>{cm[i][j]:,}</b><br>({norm[i][j]:.1%})",
                x=labels[j], y=labels[i], xref="x", yref="y",
                showarrow=False,
                font=dict(size=14, color="white" if norm[i][j]>.4 else "#c8d8ff"))
           for i in range(2) for j in range(2)]
    fig = go.Figure(go.Heatmap(
        z=norm, x=labels, y=labels,
        colorscale=[[0,"#080c23"],[.5,"#1a3a8a"],[1,"#4F8BF9"]],
        showscale=True,
        colorbar=dict(title="Rate", thickness=12, tickfont=dict(color="#b0c4ee"))))
    fig.update_layout(**_base_layout(340, f"Confusion Matrix — {name}"),
                      annotations=ann)
    fig.update_xaxes(**_GRID, title_text="Predicted")
    fig.update_yaxes(**_GRID, title_text="Actual")
    return fig


def roc_chart(curves):
    palette = ["#4F8BF9","#7C4DFF","#ff4444","#69f0ae","#ffd700","#00d4ff"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",
                              line=dict(color="rgba(150,150,150,.35)",dash="dash",width=1.5)))
    for i,(model,data) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"], mode="lines",
            name=f"{model} (AUC={data['auc']:.4f})",
            line=dict(color=palette[i%len(palette)],
                      width=3 if "World Model" in model else 1.8,
                      dash="solid" if "World Model" in model else "dot")))
    fig.update_layout(**_base_layout(440, "ROC Curves — All Models"))
    fig.update_xaxes(**_GRID, range=[0,1], title_text="FPR")
    fig.update_yaxes(**_GRID, range=[0,1.02], title_text="TPR")
    return fig


def bench_bar(df, metric):
    sd = df.sort_values(metric, ascending=True)
    colors = ["#4F8BF9" if "World Model" in m else
              "#00d4ff" if m in ["Random Forest","XGBoost"] else
              "rgba(110,120,175,.6)" for m in sd["Model"]]
    fig = go.Figure(go.Bar(
        y=sd["Model"], x=sd[metric], orientation="h",
        marker=dict(color=colors, line_width=0),
        text=[f"{v:.4f}" for v in sd[metric]], textposition="outside",
        textfont=dict(color="#c8d8ff",size=12)))
    fig.update_layout(**_base_layout(340, f"Benchmark — {metric}"))
    fig.update_xaxes(**_GRID, range=[0,1.05], title_text=metric)
    fig.update_yaxes(**_GRID, tickfont=dict(color="#b0c4ee"))
    return fig


def pie_chart(counts):
    labels = list(counts.keys())
    values = list(counts.values())
    palette = ["#4F8BF9","#ff4444","#ffd700","#69f0ae","#CE93D8",
               "#FF8A65","#4FC3F7","#F06292","#00d4ff","#7C4DFF"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=.5,
        marker=dict(colors=palette[:len(labels)],
                    line=dict(color="#080c23",width=2)),
        textinfo="percent+label",
        textfont=dict(size=10,color="#c8d8ff"),
        hovertemplate="<b>%{label}</b><br>%{value} flows<br>%{percent}<extra></extra>"))
    fig.update_layout(
        paper_bgcolor=_PAPER, font=_FONT, height=360,
        margin=dict(l=10,r=10,t=30,b=10),
        legend=dict(**_LEGEND, font=dict(color="#b0c4ee",size=10)),
        title=dict(text="Traffic Label Distribution",
                   font=dict(color="#c8d8ff"), x=.01))
    return fig


def mitre_radar(stage_probs):
    cats = list(stage_probs.keys())
    vals = list(stage_probs.values())
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill="toself", fillcolor="rgba(79,139,249,.14)",
        line=dict(color="#4F8BF9",width=2.5),
        marker=dict(size=7,color="#7C4DFF")))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(8,12,35,.8)",
            radialaxis=dict(visible=True,range=[0,1],
                            tickfont=dict(color="#6a8cc0",size=9),
                            gridcolor="rgba(79,139,249,.12)"),
            angularaxis=dict(tickfont=dict(color="#c8d8ff",size=11),
                             gridcolor="rgba(79,139,249,.12)")),
        paper_bgcolor=_PAPER,
        title=dict(text="Stage Probability Distribution",
                   font=dict(color="#c8d8ff",size=14), x=.01),
        font=dict(family="Inter"), height=400,
        margin=dict(l=60,r=60,t=60,b=30))
    return fig


def scatter_3d(df, n=300):
    pdf = df.sample(min(n,len(df)), random_state=42)
    cx = "Flow Duration"   if "Flow Duration"   in df.columns else df.columns[0]
    cy = "Flow Byts/s"     if "Flow Byts/s"     in df.columns else df.columns[1]
    cz = "Tot Fwd Pkts"    if "Tot Fwd Pkts"    in df.columns else df.columns[2]
    colors = ["#ff4444" if l!="Benign" else "#4F8BF9"
              for l in pdf["Label"]] if "Label" in df.columns else "#4F8BF9"
    text   = pdf["Label"].tolist() if "Label" in df.columns else ["?"]*len(pdf)
    fig = go.Figure(go.Scatter3d(
        x=pdf[cx].clip(0,pdf[cx].quantile(.99)),
        y=pdf[cy].clip(0,pdf[cy].quantile(.99)),
        z=pdf[cz].clip(0,pdf[cz].quantile(.99)),
        mode="markers",
        marker=dict(size=4,color=colors,opacity=.8,
                    line=dict(color="rgba(255,255,255,.08)",width=.3)),
        text=text,
        hovertemplate=f"<b>%{{text}}</b><br>{cx}:%{{x:.1f}}<br>{cy}:%{{y:.1f}}<br>{cz}:%{{z:.1f}}<extra></extra>"))
    fig.update_layout(
        paper_bgcolor=_PAPER,
        scene=dict(
            bgcolor="rgba(8,12,35,.8)",
            xaxis=dict(title=cx[:18],backgroundcolor="rgba(8,12,35,.6)",
                       gridcolor="rgba(79,139,249,.12)",tickfont=dict(color="#6a8cc0")),
            yaxis=dict(title=cy[:18],backgroundcolor="rgba(8,12,35,.6)",
                       gridcolor="rgba(79,139,249,.12)",tickfont=dict(color="#6a8cc0")),
            zaxis=dict(title=cz[:18],backgroundcolor="rgba(8,12,35,.6)",
                       gridcolor="rgba(79,139,249,.12)",tickfont=dict(color="#6a8cc0"))),
        title=dict(text="3D Network Traffic Scatter",
                   font=dict(color="#c8d8ff"), x=.01),
        height=500, font=dict(family="Inter",color="#b0c4ee"),
        margin=dict(l=0,r=0,t=50,b=0))
    return fig


def bench_radar(metrics, models):
    cats = ["F1 Score","Precision","Recall","Accuracy","AUC-ROC"]
    palette = ["#4F8BF9","#7C4DFF","#ff4444","#69f0ae","#ffd700","#00d4ff"]
    fig = go.Figure()
    for i, model in enumerate(models):
        m = metrics[model]
        vals = [m["F1 Score"],m["Precision"],m["Recall"],m["Accuracy"],m["AUC-ROC"]]
        r,g,b = int(palette[i%len(palette)][1:3],16), \
                int(palette[i%len(palette)][3:5],16), \
                int(palette[i%len(palette)][5:7],16)
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill="toself", fillcolor=f"rgba({r},{g},{b},.07)",
            line=dict(color=palette[i%len(palette)],
                      width=3 if "World Model" in model else 1.8),
            name=model))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(8,12,35,.8)",
            radialaxis=dict(visible=True,range=[.8,1.],
                            tickfont=dict(color="#6a8cc0",size=9),
                            gridcolor="rgba(79,139,249,.12)"),
            angularaxis=dict(tickfont=dict(color="#c8d8ff",size=12),
                             gridcolor="rgba(79,139,249,.12)")),
        paper_bgcolor=_PAPER,
        title=dict(text="Multi-Model Performance Radar",
                   font=dict(color="#c8d8ff",size=14), x=.01),
        font=dict(family="Inter"), height=440,
        legend=dict(**_LEGEND, x=1.05))
    return fig


# ══════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════
def page_dashboard():
    ph("📊","Dashboard","Real-time network security overview and live attack forecasting")

    ts    = generate_time_series_data(60)
    df    = generate_mock_traffic_data(300)
    stats = compute_flow_statistics(df)
    prob  = float(ts["Attack Probability"].iloc[-1])

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(metric_card("Total Flows",  f"{stats['total_flows']:,}", "+12.4%", "#4F8BF9","rgba(79,139,249,.4)"),  unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Attack Flows", f"{stats['attack_flows']:,}", f"+{stats['attack_rate']}%","#ff4444","rgba(255,68,68,.4)"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Benign Flows", f"{stats['benign_flows']:,}", "-5.2%","#69f0ae","rgba(105,240,174,.4)"), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("Flow Rate",    f"{stats['avg_flow_bytes_s']/1000:.1f}K/s","+8.1%","#ffd700","rgba(255,215,0,.4)"), unsafe_allow_html=True)
    with c5: st.markdown(metric_card("Avg Duration", f"{stats['avg_duration_ms']:.0f}ms","-2.3%","#00d4ff","rgba(0,212,255,.4)"),  unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    l, r = st.columns([1,2])
    with l:
        co()
        sh("Current Threat Level")
        st.plotly_chart(gauge(prob), use_container_width=True, key="dg")
        ms = get_model_status()
        st.markdown(f"""
        <div style="display:flex;justify-content:space-around;margin-top:8px;
                    padding-top:12px;border-top:1px solid rgba(79,139,249,.12);">
            <div style="text-align:center;">
                <div style="font-family:'Orbitron',monospace;font-size:.88rem;font-weight:700;
                            color:#4F8BF9;text-shadow:0 0 10px rgba(79,139,249,.5);">{ms['version']}</div>
                <div style="font-size:.62rem;color:rgba(176,196,255,.5);text-transform:uppercase;letter-spacing:.5px;">Version</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'Orbitron',monospace;font-size:.88rem;font-weight:700;
                            color:#69f0ae;text-shadow:0 0 10px rgba(105,240,174,.5);">97.1%</div>
                <div style="font-size:.62rem;color:rgba(176,196,255,.5);text-transform:uppercase;letter-spacing:.5px;">Accuracy</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'Orbitron',monospace;font-size:.88rem;font-weight:700;
                            color:#ffd700;text-shadow:0 0 10px rgba(255,215,0,.5);">12ms</div>
                <div style="font-size:.62rem;color:rgba(176,196,255,.5);text-transform:uppercase;letter-spacing:.5px;">Latency</div>
            </div>
        </div>""", unsafe_allow_html=True)
        cc()
    with r:
        co()
        sh("Live Attack Probability Timeline")
        st.plotly_chart(timeline_chart(ts), use_container_width=True, key="dt")
        cc()

    st.markdown("<br>", unsafe_allow_html=True)
    fl, fr = st.columns([2,1])
    with fl:
        co()
        sh("Network Flow Rate")
        st.plotly_chart(flow_chart(ts), use_container_width=True, key="df")
        cc()
    with fr:
        co()
        sh("Attack Type Distribution")
        st.plotly_chart(pie_chart(stats["label_distribution"]), use_container_width=True, key="dp")
        cc()

    st.markdown("<br>", unsafe_allow_html=True)
    co()
    sh("Recent Security Alerts")
    alerts = get_recent_alerts(8)
    hcols = st.columns([1.2,2.5,1.8,1.2,1.2])
    for hc,h in zip(hcols,["Time","Alert Type","Source IP","Severity","Confidence"]):
        hc.markdown(f"<div style='font-size:.65rem;font-weight:700;color:rgba(176,196,255,.4);"
                    f"text-transform:uppercase;letter-spacing:1px;padding:4px 0;'>{h}</div>",
                    unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,"
                "rgba(79,139,249,.3),transparent);margin:4px 0 8px;'></div>",
                unsafe_allow_html=True)
    for a in alerts:
        ac = st.columns([1.2,2.5,1.8,1.2,1.2])
        ac[0].markdown(f"<div style='font-size:.78rem;color:rgba(176,196,255,.45);font-family:monospace;'>{a['time']}</div>", unsafe_allow_html=True)
        ac[1].markdown(f"<div style='font-size:.84rem;font-weight:500;color:#c8d8ff;'>{a['type']}</div>", unsafe_allow_html=True)
        ac[2].markdown(f"<div style='font-size:.78rem;color:#4F8BF9;font-family:monospace;'>{a['source_ip']}</div>", unsafe_allow_html=True)
        ac[3].markdown(badge(a["severity"]), unsafe_allow_html=True)
        gc = "#69f0ae" if a["confidence"]>.85 else "#ffd700"
        ac[4].markdown(f"<div style='font-size:.84rem;font-weight:700;color:{gc};'>{a['confidence']:.0%}</div>", unsafe_allow_html=True)
    cc()


# ══════════════════════════════════════════════════════════════
#  PAGE: DATA INGESTION
# ══════════════════════════════════════════════════════════════
def page_data_ingestion():
    ph("📂","Data Ingestion","Upload PCAP / CSV files and extract CIC-IDS2018 feature vectors")

    c1,c2 = st.columns(2)
    with c1:
        co()
        sh("Upload Network Data")
        st.markdown("""
        <div style="background:rgba(79,139,249,.05);border:2px dashed rgba(79,139,249,.28);
                    border-radius:12px;padding:22px;text-align:center;margin-bottom:14px;">
            <div style="font-size:2.2rem;filter:drop-shadow(0 0 14px rgba(79,139,249,.7));">📁</div>
            <div style="font-size:.88rem;font-weight:600;color:#4F8BF9;margin-top:6px;">Drop your file here</div>
            <div style="font-size:.72rem;color:rgba(176,196,255,.45);margin-top:3px;">CSV · PCAP · PCAPNG</div>
        </div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("Select file", type=["csv","pcap","pcapng"],
                                    label_visibility="collapsed")
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
                st.info("PCAP needs Scapy — loaded synthetic demo instead.")
        cc()

    with c2:
        co()
        sh("Feature Extraction Pipeline")
        items = [
            ("✅","Flow-level statistics (bytes, packets, duration)",True),
            ("✅","Packet-level features (TTL, window, payload)",True),
            ("✅","TCP flag counters (SYN ACK FIN RST PSH URG)",True),
            ("✅","IAT statistics (mean, std, max, min)",True),
            ("✅","Bidirectional flow ratios",True),
            ("✅","Window size & subflow features",True),
            ("✅","Active / Idle time features",True),
            ("✅","Label encoding — 13 attack categories",True),
            ("⏳","MinMax normalization",False),
            ("⏳","Sequence windowing (len=20)",False),
        ]
        for icon,feat,done in items:
            c  = "#69f0ae" if done else "#ffd700"
            bg = "rgba(105,240,174,.04)" if done else "rgba(255,215,0,.04)"
            bd = "rgba(105,240,174,.12)" if done else "rgba(255,215,0,.08)"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:9px;padding:6px 10px;"
                f"border-radius:8px;margin-bottom:3px;background:{bg};border:1px solid {bd};'>"
                f"<span>{icon}</span>"
                f"<span style='font-size:.81rem;color:#c8d8ff;flex:1;'>{feat}</span>"
                f"<span style='font-size:.68rem;font-weight:700;color:{c};text-transform:uppercase;"
                f"letter-spacing:.5px;'>{'Done' if done else 'Pending'}</span></div>",
                unsafe_allow_html=True)
        cc()

    st.markdown("<br>", unsafe_allow_html=True)
    df_show = st.session_state.get("ingested_df", generate_mock_traffic_data(100))
    co()
    sh("Data Explorer")
    t1,t2,t3 = st.tabs(["📋  Raw Data","📊  Statistics","🌐  3D Scatter"])
    with t1:
        n = st.slider("Rows to display",10,100,30,key="di_r")
        st.dataframe(df_show.head(n), use_container_width=True, height=320)
        ca,cb,cc_ = st.columns(3)
        ca.metric("Rows",f"{len(df_show):,}")
        cb.metric("Columns",len(df_show.columns))
        cc_.metric("Memory",f"{df_show.memory_usage(deep=True).sum()/1024:.1f} KB")
    with t2:
        num = df_show.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(df_show[num].describe().round(3), use_container_width=True, height=320)
    with t3:
        st.plotly_chart(scatter_3d(df_show), use_container_width=True, key="di3d")
    cc()

    st.markdown("<br>", unsafe_allow_html=True)
    co()
    sh("Extracted Feature Matrix (Top 16 Features)")
    st.dataframe(extract_features(df_show).head(20), use_container_width=True, height=280)
    cc()


# ══════════════════════════════════════════════════════════════
#  PAGE: MODEL TRAINING
# ══════════════════════════════════════════════════════════════
def page_model_training():
    ph("🧠","World Model Training","Configure & monitor LSTM / Transformer world model")

    c1,c2 = st.columns(2)
    with c1:
        co()
        sh("Training Configuration")
        arch   = st.selectbox("Architecture", list(MODEL_ARCHITECTURES.keys()), key="ta")
        epochs = st.slider("Epochs",10,100,50,key="te")
        st.selectbox("Batch Size",[32,64,128,256],index=1)
        st.select_slider("Learning Rate",[1e-4,5e-4,1e-3,5e-3],value=1e-3,
                         format_func=lambda x:f"{x:.0e}")
        st.slider("Sequence Length",5,50,20)
        st.slider("K-step Horizon",1,20,10)
        st.slider("Dropout",0.0,0.6,0.3,.05)
        st.checkbox("Early Stopping (patience=5)",value=True)
        if st.button("🚀 Start Training", use_container_width=True):
            pb   = st.progress(0)
            stxt = st.empty()
            for i in range(epochs):
                pb.progress((i+1)/epochs)
                stxt.markdown(
                    f"<div style='font-size:.8rem;color:#4F8BF9;font-family:monospace;'>"
                    f"Epoch {i+1:03d}/{epochs} &nbsp;│&nbsp; "
                    f"loss <span style='color:#ffd700;'>{0.72*np.exp(-0.065*(i+1))+0.08:.4f}</span>"
                    f" &nbsp;│&nbsp; val_loss "
                    f"<span style='color:#ff4444;'>{0.75*np.exp(-0.058*(i+1))+0.10:.4f}</span>"
                    f"</div>", unsafe_allow_html=True)
                time.sleep(0.03)
            stxt.markdown("<div style='color:#69f0ae;font-weight:700;'>✅ Training complete!</div>",
                          unsafe_allow_html=True)
        cc()

    with c2:
        co()
        sh("Model Architecture")
        adata = MODEL_ARCHITECTURES.get(arch, list(MODEL_ARCHITECTURES.values())[0])
        for i,layer in enumerate(adata["layers"]):
            hl  = "LSTM" in layer["name"] or "Attention" in layer["name"]
            bg  = "rgba(79,139,249,.1)" if hl else "rgba(255,255,255,.02)"
            brd = "rgba(79,139,249,.35)" if hl else "rgba(79,139,249,.08)"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:7px 12px;border-radius:8px;margin-bottom:4px;"
                f"background:{bg};border:1px solid {brd};'>"
                f"<span style='font-size:.81rem;color:#c8d8ff;font-weight:500;'>{i+1}. {layer['name']}</span>"
                f"<span style='font-size:.73rem;color:#4F8BF9;'>{layer['units']}</span>"
                f"<span style='font-size:.7rem;color:rgba(176,196,255,.45);'>{layer['activation']}</span>"
                f"</div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("Parameters", adata["params"])
        m2.metric("Optimizer",  adata["optimizer"].split(" ")[0])
        m3.metric("Loss","Binary CE")
        cc()

    st.markdown("<br>", unsafe_allow_html=True)
    co()
    sh("Training Curves")
    akey = "LSTM" if "LSTM" in arch else "Transformer"
    st.plotly_chart(training_chart(generate_training_curves(epochs=epochs, model_type=akey)),
                    use_container_width=True, key="tc")
    perf = get_model_performance_metrics(arch)
    cols = st.columns(6)
    for col,(lbl,key) in zip(cols,[("Accuracy","accuracy"),("Precision","precision"),
                                    ("Recall","recall"),("F1","f1_score"),
                                    ("FPR","fpr"),("AUC","auc_roc")]):
        col.metric(lbl, f"{perf[key]:.4f}")
    cc()


# ══════════════════════════════════════════════════════════════
#  PAGE: ATTACK PREDICTION
# ══════════════════════════════════════════════════════════════
def page_attack_prediction():
    ph("🔮","Attack Prediction","K-step forward simulation — predict infiltration before it completes")

    c1,c2 = st.columns(2)
    with c1:
        co()
        sh("Forecast Configuration")
        k    = st.slider("Forecast Horizon (K steps)",3,20,10,key="pk")
        base = st.slider("Current Base Probability",0.1,0.9,.45,.05,key="pb")
        st.selectbox("Attack Type Filter",["All"]+ATTACK_LABELS[1:],key="pat")
        if st.button("▶  Run Forecast", use_container_width=True):
            with st.spinner("Running K-step simulation…"):
                time.sleep(.6)
            st.success("✅ Forecast complete")
        fdf = generate_k_step_forecast(k=k, base_prob=base)
        sh("Step-by-Step Predictions")
        for _,row in fdf.iterrows():
            pv = float(row["Attack Probability"])
            pc = "#ff4444" if pv>.7 else "#ffd700" if pv>.5 else "#69f0ae"
            sc = SC.get(row["Predicted Stage"],"#c8d8ff")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:9px;padding:6px 10px;"
                f"border-radius:8px;margin-bottom:3px;background:rgba(79,139,249,.03);"
                f"border:1px solid rgba(79,139,249,.09);'>"
                f"<span style='font-size:.78rem;color:rgba(176,196,255,.45);font-family:monospace;"
                f"min-width:44px;'>t+{row['Step']}</span>"
                f"<div style='width:7px;height:7px;border-radius:50%;background:{sc};"
                f"box-shadow:0 0 5px {sc};flex-shrink:0;'></div>"
                f"<span style='font-size:.79rem;color:{sc};flex:1;'>{row['Predicted Stage']}</span>"
                f"<span style='font-size:.8rem;font-weight:700;color:{pc};font-family:monospace;'>"
                f"{pv:.4f}</span></div>", unsafe_allow_html=True)
        cc()

    with c2:
        co()
        sh(f"Step t+{k} Threat Gauge")
        fdf2 = generate_k_step_forecast(k=k, base_prob=base)
        st.plotly_chart(gauge(float(fdf2["Attack Probability"].iloc[-1]), f"Step t+{k}"),
                        use_container_width=True, key="pg")
        sh("Stage Frequency")
        for stage,cnt in fdf2["Predicted Stage"].value_counts().items():
            cc2 = SC.get(stage,"#4F8BF9")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:9px;padding:6px 0;"
                f"border-bottom:1px solid rgba(79,139,249,.07);'>"
                f"<div style='width:9px;height:9px;border-radius:50%;background:{cc2};"
                f"box-shadow:0 0 5px {cc2};'></div>"
                f"<span style='font-size:.83rem;color:#c8d8ff;flex:1;'>{stage}</span>"
                f"<span style='font-weight:700;color:{cc2};font-size:.88rem;'>{cnt} steps</span>"
                f"</div>", unsafe_allow_html=True)
        cc()

    st.markdown("<br>", unsafe_allow_html=True)
    co()
    sh("K-Step Forecast Chart")
    st.plotly_chart(kstep_chart(fdf2), use_container_width=True, key="pkc")
    cc()

    st.markdown("<br>", unsafe_allow_html=True)
    co()
    sh("Infiltration Summary")
    i1,i2,i3,i4 = st.columns(4)
    i1.metric("Min Prob",  f"{fdf2['Attack Probability'].min():.4f}")
    i2.metric("Max Prob",  f"{fdf2['Attack Probability'].max():.4f}")
    i3.metric("Mean Prob", f"{fdf2['Attack Probability'].mean():.4f}")
    i4.metric("Steps > 0.7", int((fdf2["Attack Probability"]>.7).sum()))
    cc()


# ══════════════════════════════════════════════════════════════
#  PAGE: MITRE ATT&CK
# ══════════════════════════════════════════════════════════════
def page_mitre_attack():
    ph("🗺️","MITRE ATT&CK Mapping","Kill chain progression, technique library & detection heatmap")

    c1,c2 = st.columns(2)
    with c1:
        co()
        sh("Kill Chain Progression")
        stage_sel = st.selectbox("Predicted Stage",[s["name"] for s in MITRE_STAGES],index=3)
        chain = get_kill_chain_progression(stage_sel)
        for s in chain:
            icon = "✅" if s["status"]=="completed" else ("🔴" if s["status"]=="active" else "⬜")
            sc   = "#69f0ae" if s["status"]=="completed" else ("#ff4444" if s["status"]=="active" else "rgba(176,196,255,.28)")
            st.markdown(
                f"<div class='kc kc-{s['status']}'>"
                f"<div style='display:flex;align-items:center;gap:11px;'>"
                f"<span style='font-size:1.1rem;filter:drop-shadow(0 0 7px {sc});'>{s['icon']}</span>"
                f"<div style='flex:1;'>"
                f"<div style='font-size:.86rem;font-weight:600;color:#c8d8ff;'>{icon} {s['name']}</div>"
                f"<div style='font-size:.7rem;color:rgba(176,196,255,.5);margin-top:2px;'>{s['description'][:62]}…</div>"
                f"</div>"
                f"<span style='font-size:.68rem;font-weight:700;color:{sc};"
                f"text-transform:uppercase;letter-spacing:.5px;'>{s['status']}</span>"
                f"</div></div>", unsafe_allow_html=True)
        cc()

    with c2:
        co()
        sh("Stage Probability Radar")
        sp = generate_stage_probability_vector(stage_sel)
        st.plotly_chart(mitre_radar(sp), use_container_width=True, key="mr")
        cc()

    st.markdown("<br>", unsafe_allow_html=True)
    c3,c4 = st.columns(2)
    with c3:
        co()
        sh(f"ATT&CK Techniques — {stage_sel}")
        st.dataframe(get_technique_details_for_stage(stage_sel),
                     use_container_width=True, hide_index=True)
        cc()
    with c4:
        co()
        sh("Network Indicators of Compromise")
        sdata = next(s for s in MITRE_STAGES if s["name"]==stage_sel)
        for ind in sdata["network_indicators"]:
            st.markdown(
                f"<div style='padding:7px 0;border-bottom:1px solid rgba(79,139,249,.07);"
                f"display:flex;align-items:center;gap:9px;'>"
                f"<span style='color:#ff4444;font-size:.85rem;'>⚠</span>"
                f"<span style='font-size:.83rem;color:#c8d8ff;'>{ind}</span></div>",
                unsafe_allow_html=True)
        sh("Detection Features")
        for feat in sdata["detection_features"]:
            st.markdown(
                f"<span style='display:inline-block;background:rgba(79,139,249,.13);"
                f"border:1px solid rgba(79,139,249,.38);border-radius:20px;"
                f"padding:3px 13px;margin:3px;font-size:.76rem;color:#4F8BF9;"
                f"font-weight:600;box-shadow:0 0 7px rgba(79,139,249,.2);'>{feat}</span>",
                unsafe_allow_html=True)
        cc()

    st.markdown("<br>", unsafe_allow_html=True)
    co()
    sh("MITRE ATT&CK Detection Frequency Heatmap")
    hdf   = get_mitre_heatmap_data()
    pivot = hdf.pivot(index="Stage",columns="Technique",values="Detection Count").fillna(0)
    fig_h = px.imshow(pivot,
                      color_continuous_scale=[[0,"#080c23"],[.4,"#1a3a8a"],[.7,"#4F8BF9"],[1,"#7C4DFF"]],
                      aspect="auto")
    fig_h.update_layout(
        paper_bgcolor=_PAPER, plot_bgcolor=_PLOT,
        font=_FONT, height=300,
        margin=dict(l=10,r=10,t=20,b=60),
        coloraxis_colorbar=dict(tickfont=dict(color="#b0c4ee")))
    fig_h.update_xaxes(tickfont=dict(size=9,color="#6a8cc0"), showgrid=False)
    fig_h.update_yaxes(tickfont=dict(color="#b0c4ee"), showgrid=False)
    st.plotly_chart(fig_h, use_container_width=True, key="mh")
    cc()


# ══════════════════════════════════════════════════════════════
#  PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════
def page_explainability():
    ph("💡","Explainability","SHAP values, attention weights & feature attribution")

    t1,t2,t3 = st.tabs(["🎯  SHAP Analysis","📊  Feature Importance","🔍  Attention Weights"])

    with t1:
        co()
        sh("SHAP Beeswarm Plot")
        n_s    = st.slider("Samples",10,50,20,key="xn")
        shdf   = get_shap_values(n_s)
        top_f  = shdf.abs().mean().sort_values(ascending=False).head(12).index.tolist()
        fig_bee = go.Figure()
        for i,feat in enumerate(top_f):
            vals = shdf[feat].values
            yj   = np.random.normal(i,.15,len(vals))
            clrs = ["#ff4444" if v>0 else "#4F8BF9" for v in vals]
            fig_bee.add_trace(go.Scatter(x=vals,y=yj,mode="markers",name=feat,
                                          marker=dict(color=clrs,size=6,opacity=.75),
                                          showlegend=False))
        fig_bee.update_layout(**_base_layout(440,"SHAP Beeswarm (red=pushes toward attack)"))
        fig_bee.update_xaxes(**_GRID, title_text="SHAP Value")
        fig_bee.update_yaxes(**_GRID,
                              tickvals=list(range(len(top_f))), ticktext=top_f,
                              tickfont=dict(color="#b0c4ee"))
        st.plotly_chart(fig_bee, use_container_width=True, key="xbee")

        sh("Mean |SHAP| Values")
        ms_vals = shdf.abs().mean().sort_values(ascending=False).head(10)
        fig_ms  = go.Figure(go.Bar(
            x=ms_vals.values, y=ms_vals.index, orientation="h",
            marker=dict(color=ms_vals.values,
                        colorscale=[[0,"#1a3a8a"],[1,"#7C4DFF"]],
                        line_width=0),
            text=[f"{v:.3f}" for v in ms_vals.values], textposition="outside",
            textfont=dict(color="#c8d8ff")))
        fig_ms.update_layout(**_base_layout(340,""))
        fig_ms.update_xaxes(**_GRID, title_text="Mean |SHAP|")
        fig_ms.update_yaxes(**_GRID, tickfont=dict(color="#b0c4ee"))
        st.plotly_chart(fig_ms, use_container_width=True, key="xms")
        cc()

    with t2:
        co()
        sh("Global Feature Importance")
        tn = st.slider("Top N",5,16,16,key="xtn")
        st.plotly_chart(feat_imp_chart(FEATURE_IMPORTANCE_DATA,tn),
                        use_container_width=True, key="xfi")
        fi_df = pd.DataFrame(list(FEATURE_IMPORTANCE_DATA.items()),
                              columns=["Feature","Importance"]) \
                  .sort_values("Importance",ascending=False).reset_index(drop=True)
        fi_df.index += 1
        fi_df["Importance"] = fi_df["Importance"].apply(lambda x:f"{x:.3f}")
        st.dataframe(fi_df, use_container_width=True, height=360)
        cc()

    with t3:
        co()
        sh("Transformer Self-Attention Heatmap")
        sl = st.slider("Sequence Length",5,20,15,key="xsl")
        st.plotly_chart(attn_heatmap(get_attention_weights(sl)),
                        use_container_width=True, key="xattn")
        st.info("Brighter cells = higher attention weight. The model focuses on recent "
                "timesteps when predicting attack progression.")
        cc()


# ══════════════════════════════════════════════════════════════
#  PAGE: BENCHMARK
# ══════════════════════════════════════════════════════════════
def page_benchmark():
    ph("📈","Benchmark","World Model vs LR vs ensemble baselines — full performance comparison")

    bdf = get_benchmark_dataframe()
    co()
    sh("Model Performance Comparison Table")
    st.dataframe(bdf.style.format({
        "F1 Score":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}",
        "Accuracy":"{:.4f}","FPR":"{:.4f}","AUC-ROC":"{:.4f}"}),
        use_container_width=True, hide_index=True, height=260)
    cc()

    st.markdown("<br>", unsafe_allow_html=True)
    bc1,bc2 = st.columns(2)
    with bc1:
        co()
        m1 = st.selectbox("Metric",["F1 Score","Precision","Recall","Accuracy","AUC-ROC"],key="bm1")
        st.plotly_chart(bench_bar(bdf,m1), use_container_width=True, key="bb1")
        cc()
    with bc2:
        co()
        m2 = st.selectbox("Metric",["AUC-ROC","FPR","Accuracy","F1 Score"],key="bm2")
        st.plotly_chart(bench_bar(bdf,m2), use_container_width=True, key="bb2")
        cc()

    st.markdown("<br>", unsafe_allow_html=True)
    r1,r2 = st.columns([3,2])
    with r1:
        co()
        sh("ROC Curves")
        st.plotly_chart(roc_chart(get_roc_curve_data()), use_container_width=True, key="broc")
        cc()
    with r2:
        co()
        sh("Confusion Matrix")
        cm_s = st.selectbox("Model",["World Model (LSTM)","World Model (Transformer)","Logistic Regression"],key="bcm")
        st.plotly_chart(conf_matrix(get_confusion_matrix(cm_s),cm_s),
                        use_container_width=True, key="bcmf")
        p = get_model_performance_metrics(cm_s)
        cc1,cc2 = st.columns(2)
        cc1.metric("Precision",f"{p['precision']:.4f}")
        cc2.metric("Recall",   f"{p['recall']:.4f}")
        cc1.metric("F1 Score", f"{p['f1_score']:.4f}")
        cc2.metric("FPR",      f"{p['fpr']:.4f}")
        cc()

    st.markdown("<br>", unsafe_allow_html=True)
    co()
    sh("Multi-Model Performance Radar")
    sel = st.multiselect("Select models",BENCHMARK_MODELS,
                         default=["World Model (LSTM)","World Model (Transformer)",
                                  "Logistic Regression","XGBoost"])
    if len(sel) >= 2:
        st.plotly_chart(bench_radar(BENCHMARK_METRICS,sel),
                        use_container_width=True, key="brad")
    else:
        st.info("Select at least 2 models.")
    cc()


# ══════════════════════════════════════════════════════════════
#  PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════
def page_settings():
    ph("⚙️","Settings","Model config, detection thresholds & system information")

    c1,c2 = st.columns(2)
    with c1:
        co()
        sh("Model Configuration")
        st.selectbox("Active Model",list(MODEL_ARCHITECTURES.keys()))
        st.number_input("Sequence Window Length",5,100,20)
        st.number_input("K-step Forecast Horizon",1,30,10)
        st.number_input("LSTM Hidden Units",64,512,256,step=64)
        st.number_input("LSTM Layers",1,6,2)
        st.select_slider("Dropout Rate",[0.0,.1,.2,.3,.4,.5],value=.3)
        st.selectbox("Optimizer",["Adam","AdamW","SGD","RMSprop"])
        if st.button("💾 Save Model Config", use_container_width=True):
            st.success("✅ Saved")
        cc()
    with c2:
        co()
        sh("Detection Thresholds")
        st.slider("Alert Threshold",.3,.95,.70,.05)
        st.slider("Critical Threshold",.5,.99,.85,.05)
        st.slider("Anomaly Score Threshold",.2,.95,.60,.05)
        st.slider("Min Confidence",.5,.99,.65,.05)
        st.markdown("<hr>", unsafe_allow_html=True)
        sh("Alert Settings")
        st.checkbox("Enable Email Alerts")
        st.checkbox("Enable Webhook Notifications")
        st.checkbox("Auto-block Suspicious IPs")
        st.selectbox("Aggregation Window",["1 min","5 min","15 min","1 hour"],index=1)
        if st.button("💾 Save Threshold Config", use_container_width=True):
            st.success("✅ Saved")
        cc()

    st.markdown("<br>", unsafe_allow_html=True)
    c3,c4 = st.columns(2)
    with c3:
        co()
        sh("Data Pipeline")
        st.selectbox("Input Source",["File Upload","Live PCAP","Network Tap","SIEM Integration"])
        st.number_input("Max Flow Buffer",1000,100000,10000,step=1000)
        st.selectbox("Normalization",["MinMax","Standard","Robust","None"])
        st.checkbox("Apply PCA Reduction")
        st.number_input("Train Split (%)",60,80,70)
        if st.button("💾 Save Data Config", use_container_width=True):
            st.success("✅ Saved")
        cc()
    with c4:
        co()
        sh("System Information")
        ms = get_model_status()
        for k,v in [("App Version","1.0.0"),("Model",ms["model_name"]),
                    ("Version",ms["version"]),("Dataset",ms["dataset"]),
                    ("Parameters",ms["total_params"]),("Last Trained",ms["last_trained"]),
                    ("Epochs",str(ms["epochs_trained"])),("Device",ms["device"])]:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:7px 0;"
                f"border-bottom:1px solid rgba(79,139,249,.07);'>"
                f"<span style='font-size:.8rem;color:rgba(176,196,255,.5);'>{k}</span>"
                f"<span style='font-size:.8rem;font-weight:600;color:#c8d8ff;'>{v}</span>"
                f"</div>", unsafe_allow_html=True)
        cc()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    inject_css()
    if "ingested_df" not in st.session_state:
        st.session_state["ingested_df"] = generate_mock_traffic_data(300)

    page = render_sidebar()

    {
        "Dashboard":       page_dashboard,
        "Data Ingestion":  page_data_ingestion,
        "Model Training":  page_model_training,
        "Attack Prediction": page_attack_prediction,
        "MITRE ATT&CK":    page_mitre_attack,
        "Explainability":  page_explainability,
        "Benchmark":       page_benchmark,
        "Settings":        page_settings,
    }.get(page, page_dashboard)()


if __name__ == "__main__":
    main()
