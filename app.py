"""
CyberSentinel AI — Network Attack Forecasting
Fixed: no chart errors, proper sidebar navigation
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
    generate_k_step_forecast, get_recent_alerts, ATTACK_LABELS,
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

# ─────────────────────────────────────────────────────────────
#  CHART FACTORY — single safe update_layout call per figure
# ─────────────────────────────────────────────────────────────
PBG  = "rgba(0,0,0,0)"
PLT  = "rgba(7,11,30,0.75)"
FNT  = dict(family="Inter,sans-serif", color="#c8d8ff")
HOV  = dict(bgcolor="#0d1235", font_size=12, font_family="Inter",
            font_color="#e0eaff", bordercolor="rgba(79,139,249,.5)")
GRD  = dict(showgrid=True, gridcolor="rgba(79,139,249,.09)",
            zeroline=False, linecolor="rgba(79,139,249,.18)",
            tickfont=dict(color="#5a7ab0", size=11))
LEG  = dict(bgcolor="rgba(7,11,30,.9)", bordercolor="rgba(79,139,249,.25)",
            borderwidth=1, font=dict(color="#b0c4ee", size=11))


def fig_layout(fig, h=380, title="", legend_h=False):
    """Apply layout ONCE — no duplicate keys ever."""
    lg = dict(**LEG)
    if legend_h:
        lg.update(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    upd = dict(
        paper_bgcolor=PBG,
        plot_bgcolor=PLT,
        font=FNT,
        hoverlabel=HOV,
        legend=lg,
        height=h,
        margin=dict(l=44, r=28, t=48, b=36),
    )
    if title:
        upd["title"] = dict(text=title, font=dict(size=14, color="#c8d8ff"), x=0.01)
    fig.update_layout(**upd)
    fig.update_xaxes(**GRD)
    fig.update_yaxes(**GRD)
    return fig


SC = {
    "Reconnaissance": "#4FC3F7", "Initial Access": "#81C784",
    "Execution": "#FFB74D",      "Lateral Movement": "#FF8A65",
    "Command & Control": "#CE93D8", "Exfiltration": "#ff4444",
}

# ─────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Orbitron:wght@600;700;900&display=swap');

html,body,[class*="css"]{font-family:'Inter',sans-serif!important}

/* BACKGROUND */
.stApp{
  background:
    radial-gradient(ellipse at 12% 38%,rgba(79,139,249,.16) 0%,transparent 52%),
    radial-gradient(ellipse at 88% 14%,rgba(124,77,255,.13) 0%,transparent 44%),
    radial-gradient(ellipse at 52% 88%,rgba(0,212,255,.09) 0%,transparent 44%),
    radial-gradient(ellipse at 88% 82%,rgba(229,57,53,.07) 0%,transparent 34%),
    #03060f!important;
  background-attachment:fixed!important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#05091a 0%,#070c22 60%,#05091a 100%)!important;
  border-right:1px solid rgba(79,139,249,.22)!important;
  box-shadow:4px 0 40px rgba(0,0,0,.7)!important;
  padding:0!important;
}
[data-testid="stSidebar"]>div{background:transparent!important;padding-top:0!important}
[data-testid="stSidebar"] *{color:#c8d8ff!important}

/* hide every default radio label & dot */
[data-testid="stSidebar"] .stRadio>label{display:none!important}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]>div:first-child{display:none!important}
[data-testid="stSidebar"] .stRadio label span[data-testid="stMarkdownContainer"]{display:none!important}

/* NAV ITEMS */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{
  display:flex;flex-direction:column;gap:1px;padding:0 12px;
}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]{
  background:transparent!important;
  border:none!important;
  border-radius:10px!important;
  padding:10px 14px!important;
  cursor:pointer!important;
  transition:all .18s ease!important;
  display:flex!important;align-items:center!important;
  font-size:.87rem!important;font-weight:500!important;
  color:rgba(170,192,255,.6)!important;
  margin:0!important;width:100%!important;
  letter-spacing:.2px!important;
}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover{
  background:rgba(79,139,249,.13)!important;
  color:#fff!important;padding-left:18px!important;
}
[data-testid="stSidebar"] .stRadio label[aria-checked="true"]{
  background:linear-gradient(135deg,rgba(79,139,249,.3),rgba(124,77,255,.2))!important;
  border-left:3px solid #4F8BF9!important;
  color:#fff!important;
  box-shadow:0 0 16px rgba(79,139,249,.25),inset 0 0 16px rgba(79,139,249,.04)!important;
  padding-left:11px!important;
}

/* MAIN */
.main .block-container{padding:1.4rem 2rem 3rem!important;max-width:100%!important}

/* PAGE HEADER */
.phead{
  background:linear-gradient(135deg,rgba(79,139,249,.17),rgba(124,77,255,.11) 60%,rgba(0,212,255,.07));
  border:1px solid rgba(79,139,249,.32);border-radius:18px;
  padding:24px 28px;margin-bottom:20px;position:relative;overflow:hidden;
  box-shadow:0 8px 36px rgba(79,139,249,.16),inset 0 0 50px rgba(79,139,249,.03);
}
.phead::after{content:'';position:absolute;top:-55%;right:-14%;width:320px;height:320px;
  background:radial-gradient(circle,rgba(124,77,255,.11) 0%,transparent 70%);pointer-events:none}
.phead h1{font-family:'Orbitron',monospace!important;font-size:1.38rem!important;
  font-weight:800!important;color:#fff!important;margin:0!important;
  text-shadow:0 0 22px rgba(79,139,249,.85)!important;letter-spacing:.4px!important}
.phead p{font-size:.83rem!important;color:rgba(200,216,255,.72)!important;margin:6px 0 0!important}

/* CARDS */
.card{
  background:linear-gradient(135deg,rgba(7,11,30,.97),rgba(10,15,42,.93));
  border:1px solid rgba(79,139,249,.15);border-radius:15px;
  padding:18px 20px;margin-bottom:14px;position:relative;overflow:hidden;
  box-shadow:0 5px 28px rgba(0,0,0,.42),inset 0 1px 0 rgba(255,255,255,.03);
  transition:transform .2s,border-color .2s,box-shadow .2s;
}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(79,139,249,.45),transparent)}
.card:hover{transform:translateY(-2px);border-color:rgba(79,139,249,.36)!important;
  box-shadow:0 10px 38px rgba(0,0,0,.5),0 0 18px rgba(79,139,249,.1)!important}

/* METRIC CARDS */
.mcard{
  background:linear-gradient(135deg,rgba(7,11,30,.98),rgba(10,15,42,.94));
  border:1px solid rgba(79,139,249,.13);border-radius:13px;
  padding:16px 12px;text-align:center;position:relative;overflow:hidden;
  box-shadow:0 4px 22px rgba(0,0,0,.35);
  transition:transform .2s,box-shadow .2s;
  height:114px;display:flex;flex-direction:column;justify-content:center;align-items:center;
}
.mcard:hover{transform:translateY(-3px);
  box-shadow:0 9px 32px rgba(0,0,0,.5),0 0 16px var(--gc,rgba(79,139,249,.3))!important}
.mbar{position:absolute;top:0;left:0;right:0;height:2px;
  box-shadow:0 0 7px var(--gc,rgba(79,139,249,.5))}
.mval{font-family:'Orbitron',monospace!important;font-size:1.68rem!important;
  font-weight:800!important;line-height:1.1!important}
.mlbl{font-size:.63rem!important;font-weight:600!important;
  color:rgba(170,192,255,.52)!important;margin-top:3px!important;
  text-transform:uppercase!important;letter-spacing:1px!important}
.mdelta{font-size:.68rem!important;margin-top:2px!important;font-weight:600!important}

/* SECTION HEAD */
.sh{font-size:.84rem;font-weight:700;color:#d0e0ff;
  border-left:3px solid #4F8BF9;padding-left:10px;
  margin:14px 0 10px;text-transform:uppercase;letter-spacing:.4px}

/* BADGES */
.bc{background:rgba(229,57,53,.22);color:#ff6b6b;border:1px solid rgba(229,57,53,.45);
    border-radius:20px;padding:2px 10px;font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.4px}
.bh{background:rgba(255,143,0,.18);color:#ffb347;border:1px solid rgba(255,143,0,.38);
    border-radius:20px;padding:2px 10px;font-size:.68rem;font-weight:700;text-transform:uppercase}
.bm{background:rgba(255,215,0,.13);color:#ffd700;border:1px solid rgba(255,215,0,.3);
    border-radius:20px;padding:2px 10px;font-size:.68rem;font-weight:700;text-transform:uppercase}
.bl{background:rgba(67,160,71,.18);color:#69f0ae;border:1px solid rgba(67,160,71,.35);
    border-radius:20px;padding:2px 10px;font-size:.68rem;font-weight:700;text-transform:uppercase}

/* KILL CHAIN */
.kc{border-radius:9px;padding:10px 13px;margin-bottom:5px;
    border:1px solid rgba(79,139,249,.09);background:rgba(255,255,255,.02);transition:all .18s}
.kc-completed{border-left:3px solid #43A047;opacity:.76}
.kc-active{border-left:3px solid #E53935;background:rgba(229,57,53,.06);
           border-color:rgba(229,57,53,.32);box-shadow:0 0 16px rgba(229,57,53,.22)}
.kc-pending{border-left:3px solid rgba(79,139,249,.18);opacity:.38}

/* BUTTONS */
.stButton>button{
  background:linear-gradient(135deg,#4F8BF9,#7C4DFF)!important;
  color:#fff!important;border:none!important;border-radius:9px!important;
  padding:9px 20px!important;font-weight:700!important;font-size:.83rem!important;
  letter-spacing:.4px!important;text-transform:uppercase!important;
  transition:all .22s!important;box-shadow:0 4px 16px rgba(79,139,249,.38)!important}
.stButton>button:hover{transform:translateY(-2px)!important;
  box-shadow:0 7px 26px rgba(79,139,249,.58)!important}

/* INPUTS */
.stSelectbox>div>div,.stNumberInput>div>div>input,.stTextInput>div>div>input{
  background:rgba(7,11,30,.88)!important;
  border:1px solid rgba(79,139,249,.22)!important;
  border-radius:8px!important;color:#c8d8ff!important}

/* TABS */
.stTabs [data-baseweb="tab-list"]{
  background:rgba(7,11,30,.82)!important;border-radius:10px!important;
  padding:4px!important;border:1px solid rgba(79,139,249,.16)!important}
.stTabs [data-baseweb="tab"]{border-radius:7px!important;
  color:rgba(200,216,255,.52)!important;font-weight:500!important;
  font-size:.82rem!important;padding:7px 15px!important}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,#4F8BF9,#7C4DFF)!important;
  color:#fff!important;box-shadow:0 4px 12px rgba(79,139,249,.38)!important}

/* PROGRESS */
.stProgress>div>div>div{
  background:linear-gradient(90deg,#4F8BF9,#7C4DFF)!important;
  box-shadow:0 0 9px rgba(79,139,249,.45)!important}

/* SCROLLBAR */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:rgba(7,11,30,.5)}
::-webkit-scrollbar-thumb{background:linear-gradient(#4F8BF9,#7C4DFF);border-radius:10px}

/* HIDE CLUTTER */
#MainMenu,footer,.stDeployButton,[data-testid="stToolbar"]{display:none!important}

/* TEXT */
h1,h2,h3,h4,h5,h6{color:#e0eaff!important}
p,label{color:#b0c4ee!important}
.stMarkdown p{color:#b0c4ee!important}
[data-testid="stMetricValue"]{
  color:#7EB8FF!important;font-family:'Orbitron',monospace!important;font-weight:700!important}
[data-testid="stMetricLabel"]{
  color:rgba(176,196,255,.62)!important;font-size:.7rem!important;
  text-transform:uppercase!important;letter-spacing:.4px!important}
hr{border:none!important;height:1px!important;
   background:linear-gradient(90deg,transparent,rgba(79,139,249,.38),transparent)!important;
   margin:12px 0!important}
.stAlert{background:rgba(79,139,249,.07)!important;
  border-left:3px solid #4F8BF9!important;border-radius:8px!important;color:#c8d8ff!important}
.stCheckbox label span{color:#c8d8ff!important}
.stDataFrame{border-radius:10px!important;overflow:hidden!important}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SIDEBAR — proper app-style navigation
# ─────────────────────────────────────────────────────────────
NAV_ITEMS = [
    ("📊", "Dashboard"),
    ("📂", "Data Ingestion"),
    ("🧠", "Model Training"),
    ("🔮", "Attack Prediction"),
    ("🗺️", "MITRE ATT&CK"),
    ("💡", "Explainability"),
    ("📈", "Benchmark"),
    ("⚙️", "Settings"),
]

def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
<div style="padding:22px 16px 14px;text-align:center">
  <div style="font-size:2.4rem;filter:drop-shadow(0 0 20px rgba(79,139,249,.9))">🔐</div>
  <div style="font-family:'Orbitron',monospace;font-size:1rem;font-weight:900;
              color:#fff;letter-spacing:2.5px;margin-top:7px;
              text-shadow:0 0 20px rgba(79,139,249,.85)">CYBERSENTINEL</div>
  <div style="font-family:'Orbitron',monospace;font-size:.55rem;font-weight:400;
              color:rgba(170,192,255,.5);letter-spacing:3px;margin-top:2px">AI · NETWORK DEFENSE</div>
</div>
<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(79,139,249,.4),transparent);margin:0 12px 14px"></div>
<div style="font-size:.6rem;color:rgba(170,192,255,.32);text-transform:uppercase;
            letter-spacing:2px;padding:0 16px 7px">Navigation</div>
""", unsafe_allow_html=True)

        # Build label list for radio
        labels = [f"{icon}   {name}" for icon, name in NAV_ITEMS]
        choice = st.radio("_nav", labels, label_visibility="collapsed")

        # Model status
        ms = get_model_status()
        st.markdown(f"""
<div style="background:rgba(79,139,249,.06);border:1px solid rgba(79,139,249,.16);
            border-radius:11px;padding:12px 14px;margin:16px 12px 8px">
  <div style="font-size:.58rem;color:rgba(170,192,255,.4);text-transform:uppercase;
              letter-spacing:1.5px;margin-bottom:6px">⚡ Model Status</div>
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
    <div style="width:6px;height:6px;border-radius:50%;background:#43A047;
                box-shadow:0 0 7px #43A047;flex-shrink:0"></div>
    <span style="font-size:.78rem;font-weight:700;color:#69f0ae">OPERATIONAL</span>
  </div>
  <div style="font-size:.7rem;color:rgba(170,192,255,.6);line-height:1.6">
    {ms['model_name']}<br>
    <span style="color:#4F8BF9;font-weight:600">{ms['version']}</span> · {ms['dataset']}
  </div>
</div>
<div style="text-align:center;padding:10px 0 4px;font-size:.58rem;
            color:rgba(170,192,255,.22);letter-spacing:1px">SIH 2024 · NTRO CHALLENGE</div>
""", unsafe_allow_html=True)

    # Extract page name from choice
    for icon, name in NAV_ITEMS:
        if f"{icon}   {name}" == choice:
            return name
    return "Dashboard"


# ─────────────────────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────────────────────
def ph(icon, title, sub):
    st.markdown(f'<div class="phead"><h1>{icon} {title}</h1><p>{sub}</p></div>',
                unsafe_allow_html=True)

def sh(txt):
    st.markdown(f'<div class="sh">{txt}</div>', unsafe_allow_html=True)

def co():
    st.markdown('<div class="card">', unsafe_allow_html=True)

def cx():
    st.markdown('</div>', unsafe_allow_html=True)

def mc(label, value, delta="", color="#4F8BF9", glow="rgba(79,139,249,.38)"):
    dh = ""
    if delta:
        dc = "#69f0ae" if delta.startswith("+") else "#ff6b6b"
        dh = f'<div class="mdelta" style="color:{dc}">{delta}</div>'
    return (f'<div class="mcard" style="--gc:{glow}">'
            f'<div class="mbar" style="background:{color}"></div>'
            f'<div class="mval" style="color:{color};text-shadow:0 0 16px {glow}">{value}</div>'
            f'<div class="mlbl">{label}</div>{dh}</div>')

def badge(sev):
    return f'<span class="{"bc" if sev=="Critical" else "bh" if sev=="High" else "bm" if sev=="Medium" else "bl"}">{sev}</span>'


# ─────────────────────────────────────────────────────────────
#  CHARTS — every one uses fig_layout() ONCE, no merging
# ─────────────────────────────────────────────────────────────
def chart_gauge(prob, title="Attack Probability"):
    pct = round(prob * 100, 1)
    c, lv = (("#69f0ae","LOW") if pct<25 else ("#ffd700","MODERATE") if pct<50
              else ("#ff9800","HIGH") if pct<75 else ("#ff4444","CRITICAL"))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        title={"text": f"<span style='font-size:.83em;color:#c8d8ff'>{title}</span><br>"
                       f"<span style='font-size:.72em;color:{c};font-weight:700'>{lv}</span>",
               "font": {"size":13}},
        number={"suffix":"%","font":{"size":38,"color":c,"family":"Orbitron"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#4a6fa5","tickfont":{"size":9,"color":"#6a8cc0"}},
               "bar":{"color":c,"thickness":.24},
               "bgcolor":"rgba(7,11,30,.82)","borderwidth":2,"bordercolor":"rgba(79,139,249,.22)",
               "steps":[{"range":[0,25],"color":"rgba(105,240,174,.04)"},
                         {"range":[25,50],"color":"rgba(255,215,0,.04)"},
                         {"range":[50,75],"color":"rgba(255,152,0,.04)"},
                         {"range":[75,100],"color":"rgba(255,68,68,.04)"}],
               "threshold":{"line":{"color":"#ff4444","width":3},"thickness":.78,"value":70}},
    ))
    fig.update_layout(paper_bgcolor=PBG, height=295,
                      margin=dict(l=28,r=28,t=56,b=18),
                      font=dict(family="Inter",color="#b0c4ee"))
    return fig


def chart_timeline(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Timestamp"], y=df["Attack Probability"],
                              mode="lines", name="Attack Prob",
                              line=dict(color="#ff4444",width=2.4),
                              fill="tozeroy", fillcolor="rgba(255,68,68,.07)"))
    fig.add_trace(go.Scatter(x=df["Timestamp"], y=df["Anomaly Score"],
                              mode="lines", name="Anomaly Score",
                              line=dict(color="#7C4DFF",width=1.7,dash="dot")))
    fig.add_hline(y=.7, line_dash="dash", line_color="rgba(255,68,68,.55)",
                  annotation_text="Threshold 0.70",
                  annotation_font_color="#ff6b6b", annotation_position="top right")
    fig_layout(fig, h=310, title="Live Attack Probability Timeline", legend_h=True)
    fig.update_xaxes(**GRD, title_text="Time")
    fig.update_yaxes(**GRD, range=[0,1], title_text="Probability")
    return fig


def chart_flow(df):
    fig = go.Figure(go.Bar(
        x=df["Timestamp"], y=df["Flow Rate (flows/min)"],
        marker=dict(color=df["Flow Rate (flows/min)"],
                    colorscale=[[0,"#101a50"],[.5,"#4F8BF9"],[1,"#7C4DFF"]],
                    line_width=0), opacity=.88))
    fig_layout(fig, h=295, title="Network Flow Rate (flows/min)")
    fig.update_xaxes(**GRD, title_text="Time")
    fig.update_yaxes(**GRD, title_text="Flows / min")
    return fig


def chart_pie(counts):
    labels = list(counts.keys())
    values = list(counts.values())
    pal = ["#4F8BF9","#ff4444","#ffd700","#69f0ae","#CE93D8","#FF8A65","#4FC3F7","#F06292","#00d4ff","#7C4DFF"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=.48,
        marker=dict(colors=pal[:len(labels)], line=dict(color="#070b1e",width=2)),
        textinfo="percent+label", textfont=dict(size=9,color="#c8d8ff"),
        hovertemplate="<b>%{label}</b><br>%{value}<br>%{percent}<extra></extra>"))
    fig.update_layout(paper_bgcolor=PBG, font=FNT, height=350,
                      margin=dict(l=8,r=8,t=28,b=8),
                      legend=dict(**LEG, font=dict(color="#b0c4ee",size=9)),
                      title=dict(text="Traffic Label Distribution",
                                 font=dict(color="#c8d8ff",size=13), x=.01))
    return fig


def chart_kstep(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(df["Step"])+list(df["Step"])[::-1],
        y=list(df["Upper Bound"])+list(df["Lower Bound"])[::-1],
        fill="toself", fillcolor="rgba(79,139,249,.07)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
    fig.add_trace(go.Scatter(
        x=df["Step"], y=df["Attack Probability"],
        mode="lines+markers", name="Attack Prob",
        line=dict(color="#4F8BF9",width=2.8),
        marker=dict(size=8,color="#7C4DFF",line=dict(color="#c8d8ff",width=1.8))))
    fig.add_hline(y=.7, line_dash="dash", line_color="rgba(255,68,68,.55)",
                  annotation_text="Alert Threshold", annotation_font_color="#ff6b6b")
    prev = None
    for _, row in df.iterrows():
        s = row["Predicted Stage"]
        if s != prev:
            fig.add_annotation(x=row["Step"], y=min(row["Attack Probability"]+.09,.96),
                                text=s, showarrow=False,
                                font=dict(size=7,color=SC.get(s,"#c8d8ff")),
                                bgcolor="rgba(7,11,30,.88)",
                                bordercolor=SC.get(s,"#4F8BF9"),borderwidth=1,borderpad=3)
            prev = s
    fig_layout(fig, h=395, title="K-Step Forward Simulation — Attack Probability Forecast", legend_h=True)
    fig.update_xaxes(**GRD, title_text="Step (t+k)", tickmode="linear")
    fig.update_yaxes(**GRD, range=[0,1.06], title_text="Attack Probability")
    return fig


def chart_training(df):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Loss Curves", "Accuracy & F1"])
    pairs = [
        (df["Epoch"],df["Train Loss"],  "Train Loss",   "#4F8BF9", "solid",   1,1),
        (df["Epoch"],df["Val Loss"],    "Val Loss",     "#ff4444", "dot",     1,1),
        (df["Epoch"],df["Train Accuracy"],"Train Acc",  "#69f0ae", "solid",   1,2),
        (df["Epoch"],df["Val Accuracy"],  "Val Acc",    "#00d4ff", "dot",     1,2),
        (df["Epoch"],df["Train F1"],      "Train F1",   "#ffd700", "dash",    1,2),
    ]
    for x,y,name,col,dash,r,c in pairs:
        fig.add_trace(go.Scatter(x=x,y=y,name=name,
                                  line=dict(color=col,width=2.1,dash=dash)),row=r,col=c)
    # subplot titles color
    for ann in fig.layout.annotations:
        ann.font.color = "#c8d8ff"
    fig.update_layout(
        paper_bgcolor=PBG, plot_bgcolor=PLT, font=FNT, hoverlabel=HOV,
        height=350, margin=dict(l=44,r=28,t=48,b=36),
        legend=dict(**LEG, orientation="h", y=-.22, x=.5, xanchor="center"))
    fig.update_xaxes(**GRD)
    fig.update_yaxes(**GRD)
    return fig


def chart_feat_imp(data, top_n=16):
    items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    feats = [x[0] for x in items]
    vals  = [x[1] for x in items]
    fig = go.Figure(go.Bar(
        x=vals[::-1], y=feats[::-1], orientation="h",
        marker=dict(color=vals[::-1],
                    colorscale=[[0,"#101a50"],[.5,"#4F8BF9"],[1,"#7C4DFF"]],
                    line_width=0),
        text=[f"{v:.3f}" for v in vals[::-1]], textposition="outside",
        textfont=dict(color="#c8d8ff",size=10)))
    fig_layout(fig, h=440, title="Global Feature Importance (SHAP-based)")
    fig.update_xaxes(**GRD, title_text="Importance Score")
    fig.update_yaxes(**GRD, tickfont=dict(color="#b0c4ee",size=10))
    return fig


def chart_shap_bar(shdf):
    ms = shdf.abs().mean().sort_values(ascending=False).head(10)
    fig = go.Figure(go.Bar(
        x=ms.values, y=ms.index, orientation="h",
        marker=dict(color=ms.values,
                    colorscale=[[0,"#1a3a8a"],[1,"#7C4DFF"]],
                    line_width=0),
        text=[f"{v:.3f}" for v in ms.values], textposition="outside",
        textfont=dict(color="#c8d8ff",size=10)))
    fig_layout(fig, h=330, title="Mean |SHAP| per Feature")
    fig.update_xaxes(**GRD, title_text="Mean |SHAP|")
    fig.update_yaxes(**GRD, tickfont=dict(color="#b0c4ee",size=10))
    return fig


def chart_beeswarm(shdf):
    top_f = shdf.abs().mean().sort_values(ascending=False).head(12).index.tolist()
    fig = go.Figure()
    for i, feat in enumerate(top_f):
        vals = shdf[feat].values
        yj   = np.random.normal(i, .14, len(vals))
        clrs = ["#ff4444" if v>0 else "#4F8BF9" for v in vals]
        fig.add_trace(go.Scatter(x=vals, y=yj, mode="markers", name=feat,
                                  marker=dict(color=clrs,size=5,opacity=.72),
                                  showlegend=False))
    fig_layout(fig, h=430, title="SHAP Beeswarm  (red = pushes toward attack)")
    fig.update_xaxes(**GRD, title_text="SHAP Value")
    fig.update_yaxes(**GRD, tickvals=list(range(len(top_f))),
                     ticktext=top_f, tickfont=dict(color="#b0c4ee",size=10))
    return fig


def chart_attention(attn):
    n = attn.shape[0]
    labels = [f"t-{n-1-i}" if i < n-1 else "t" for i in range(n)]
    fig = go.Figure(go.Heatmap(
        z=attn, x=labels, y=labels,
        colorscale=[[0,"#07091e"],[.35,"#1a3a8a"],[.7,"#4F8BF9"],[1,"#7C4DFF"]],
        showscale=True,
        colorbar=dict(title="Weight",thickness=11,tickfont=dict(color="#b0c4ee"))))
    fig_layout(fig, h=415, title="Transformer Attention Weights (Self-Attention)")
    fig.update_xaxes(**GRD, title_text="Key")
    fig.update_yaxes(**GRD, title_text="Query")
    return fig


def chart_confusion(cm, name):
    labels = ["Benign","Attack"]
    norm = cm.astype(float) / cm.sum(axis=1,keepdims=True)
    anns = [dict(text=f"<b>{cm[i][j]:,}</b><br>({norm[i][j]:.1%})",
                 x=labels[j],y=labels[i],xref="x",yref="y",showarrow=False,
                 font=dict(size=13,color="white" if norm[i][j]>.4 else "#c8d8ff"))
            for i in range(2) for j in range(2)]
    fig = go.Figure(go.Heatmap(
        z=norm, x=labels, y=labels,
        colorscale=[[0,"#07091e"],[.5,"#1a3a8a"],[1,"#4F8BF9"]],
        showscale=True, colorbar=dict(title="Rate",thickness=11,tickfont=dict(color="#b0c4ee"))))
    fig_layout(fig, h=335, title=f"Confusion Matrix — {name}")
    fig.update_layout(annotations=anns)
    fig.update_xaxes(**GRD, title_text="Predicted")
    fig.update_yaxes(**GRD, title_text="Actual")
    return fig


def chart_roc(curves):
    pal = ["#4F8BF9","#7C4DFF","#ff4444","#69f0ae","#ffd700","#00d4ff"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",
                              line=dict(color="rgba(140,140,140,.3)",dash="dash",width=1.4)))
    for i,(model,data) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=data["fpr"],y=data["tpr"],mode="lines",
            name=f"{model} (AUC={data['auc']:.4f})",
            line=dict(color=pal[i%len(pal)],
                      width=2.8 if "World Model" in model else 1.6,
                      dash="solid" if "World Model" in model else "dot")))
    fig_layout(fig, h=435, title="ROC Curves — All Models")
    fig.update_xaxes(**GRD, range=[0,1], title_text="FPR")
    fig.update_yaxes(**GRD, range=[0,1.02], title_text="TPR")
    return fig


def chart_bench_bar(df, metric):
    sd = df.sort_values(metric, ascending=True)
    clrs = ["#4F8BF9" if "World Model" in m else
            "#00d4ff" if m in ["Random Forest","XGBoost"] else
            "rgba(100,110,165,.6)" for m in sd["Model"]]
    fig = go.Figure(go.Bar(
        y=sd["Model"], x=sd[metric], orientation="h",
        marker=dict(color=clrs,line_width=0),
        text=[f"{v:.4f}" for v in sd[metric]], textposition="outside",
        textfont=dict(color="#c8d8ff",size=11)))
    fig_layout(fig, h=335, title=f"Benchmark — {metric}")
    fig.update_xaxes(**GRD, range=[0,1.06], title_text=metric)
    fig.update_yaxes(**GRD, tickfont=dict(color="#b0c4ee",size=10))
    return fig


def chart_bench_radar(metrics, models):
    cats = ["F1 Score","Precision","Recall","Accuracy","AUC-ROC"]
    pal  = ["#4F8BF9","#7C4DFF","#ff4444","#69f0ae","#ffd700","#00d4ff"]
    fig  = go.Figure()
    for i, model in enumerate(models):
        m    = metrics[model]
        vals = [m["F1 Score"],m["Precision"],m["Recall"],m["Accuracy"],m["AUC-ROC"]]
        r,g,b = int(pal[i%len(pal)][1:3],16),int(pal[i%len(pal)][3:5],16),int(pal[i%len(pal)][5:7],16)
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill="toself", fillcolor=f"rgba({r},{g},{b},.06)",
            line=dict(color=pal[i%len(pal)],width=2.6 if "World Model" in model else 1.6),
            name=model))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(7,11,30,.8)",
            radialaxis=dict(visible=True,range=[.8,1.],
                            tickfont=dict(color="#5a7ab0",size=9),
                            gridcolor="rgba(79,139,249,.1)"),
            angularaxis=dict(tickfont=dict(color="#c8d8ff",size=11),
                             gridcolor="rgba(79,139,249,.1)")),
        paper_bgcolor=PBG,
        title=dict(text="Multi-Model Performance Radar",
                   font=dict(color="#c8d8ff",size=13),x=.01),
        font=dict(family="Inter"),height=430,
        legend=dict(**LEG,x=1.04))
    return fig


def chart_mitre_radar(sp):
    cats = list(sp.keys()); vals = list(sp.values())
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill="toself", fillcolor="rgba(79,139,249,.12)",
        line=dict(color="#4F8BF9",width=2.3),
        marker=dict(size=6,color="#7C4DFF")))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(7,11,30,.8)",
            radialaxis=dict(visible=True,range=[0,1],
                            tickfont=dict(color="#5a7ab0",size=9),
                            gridcolor="rgba(79,139,249,.1)"),
            angularaxis=dict(tickfont=dict(color="#c8d8ff",size=10),
                             gridcolor="rgba(79,139,249,.1)")),
        paper_bgcolor=PBG,
        title=dict(text="Stage Probability Distribution",
                   font=dict(color="#c8d8ff",size=13),x=.01),
        font=dict(family="Inter"),height=390,
        margin=dict(l=55,r=55,t=55,b=28))
    return fig


def chart_scatter3d(df, n=300):
    pdf = df.sample(min(n,len(df)), random_state=42)
    cx = "Flow Duration"  if "Flow Duration"  in df.columns else df.columns[0]
    cy = "Flow Byts/s"    if "Flow Byts/s"    in df.columns else df.columns[1]
    cz = "Tot Fwd Pkts"   if "Tot Fwd Pkts"   in df.columns else df.columns[2]
    clrs = (["#ff4444" if l!="Benign" else "#4F8BF9" for l in pdf["Label"]]
            if "Label" in df.columns else "#4F8BF9")
    txt  = pdf["Label"].tolist() if "Label" in df.columns else ["?"]*len(pdf)
    fig = go.Figure(go.Scatter3d(
        x=pdf[cx].clip(0,pdf[cx].quantile(.99)),
        y=pdf[cy].clip(0,pdf[cy].quantile(.99)),
        z=pdf[cz].clip(0,pdf[cz].quantile(.99)),
        mode="markers",
        marker=dict(size=4,color=clrs,opacity=.8,
                    line=dict(color="rgba(255,255,255,.06)",width=.3)),
        text=txt,
        hovertemplate=f"<b>%{{text}}</b><br>{cx}:%{{x:.1f}}<br>{cy}:%{{y:.1f}}<br>{cz}:%{{z:.1f}}<extra></extra>"))
    fig.update_layout(
        paper_bgcolor=PBG,
        scene=dict(bgcolor="rgba(7,11,30,.8)",
                   xaxis=dict(title=cx[:18],backgroundcolor="rgba(7,11,30,.6)",
                              gridcolor="rgba(79,139,249,.1)",tickfont=dict(color="#5a7ab0")),
                   yaxis=dict(title=cy[:18],backgroundcolor="rgba(7,11,30,.6)",
                              gridcolor="rgba(79,139,249,.1)",tickfont=dict(color="#5a7ab0")),
                   zaxis=dict(title=cz[:18],backgroundcolor="rgba(7,11,30,.6)",
                              gridcolor="rgba(79,139,249,.1)",tickfont=dict(color="#5a7ab0"))),
        title=dict(text="3D Network Traffic Scatter",font=dict(color="#c8d8ff"),x=.01),
        height=490, font=dict(family="Inter",color="#b0c4ee"),
        margin=dict(l=0,r=0,t=46,b=0))
    return fig


def chart_mitre_heatmap():
    hdf   = get_mitre_heatmap_data()
    pivot = hdf.pivot(index="Stage",columns="Technique",values="Detection Count").fillna(0)
    fig   = px.imshow(pivot,
                      color_continuous_scale=[[0,"#07091e"],[.4,"#1a3a8a"],[.7,"#4F8BF9"],[1,"#7C4DFF"]],
                      aspect="auto")
    fig.update_layout(paper_bgcolor=PBG, plot_bgcolor=PLT, font=FNT,
                      height=295, margin=dict(l=8,r=8,t=18,b=58),
                      coloraxis_colorbar=dict(tickfont=dict(color="#b0c4ee")))
    fig.update_xaxes(tickfont=dict(size=8,color="#5a7ab0"), showgrid=False)
    fig.update_yaxes(tickfont=dict(color="#b0c4ee",size=10), showgrid=False)
    return fig


# ─────────────────────────────────────────────────────────────
#  PAGES
# ─────────────────────────────────────────────────────────────
def page_dashboard():
    ph("📊","Dashboard","Real-time network security overview · live attack forecasting")

    ts    = generate_time_series_data(60)
    df    = generate_mock_traffic_data(300)
    stats = compute_flow_statistics(df)
    prob  = float(ts["Attack Probability"].iloc[-1])

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(mc("Total Flows",  f"{stats['total_flows']:,}", "+12.4%","#4F8BF9","rgba(79,139,249,.38)"),  unsafe_allow_html=True)
    with c2: st.markdown(mc("Attack Flows", f"{stats['attack_flows']:,}",f"+{stats['attack_rate']}%","#ff4444","rgba(255,68,68,.38)"), unsafe_allow_html=True)
    with c3: st.markdown(mc("Benign Flows", f"{stats['benign_flows']:,}","-5.2%","#69f0ae","rgba(105,240,174,.38)"), unsafe_allow_html=True)
    with c4: st.markdown(mc("Flow Rate",    f"{stats['avg_flow_bytes_s']/1000:.1f}K/s","+8.1%","#ffd700","rgba(255,215,0,.38)"), unsafe_allow_html=True)
    with c5: st.markdown(mc("Avg Duration", f"{stats['avg_duration_ms']:.0f}ms","-2.3%","#00d4ff","rgba(0,212,255,.38)"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    l,r = st.columns([1,2])
    with l:
        co(); sh("Current Threat Level")
        st.plotly_chart(chart_gauge(prob), use_container_width=True, key="dg")
        ms = get_model_status()
        st.markdown(f"""
<div style="display:flex;justify-content:space-around;margin-top:8px;
            padding-top:11px;border-top:1px solid rgba(79,139,249,.1)">
  <div style="text-align:center">
    <div style="font-family:'Orbitron',monospace;font-size:.84rem;font-weight:700;
                color:#4F8BF9;text-shadow:0 0 9px rgba(79,139,249,.5)">{ms['version']}</div>
    <div style="font-size:.6rem;color:rgba(170,192,255,.46);text-transform:uppercase;letter-spacing:.4px">Version</div>
  </div>
  <div style="text-align:center">
    <div style="font-family:'Orbitron',monospace;font-size:.84rem;font-weight:700;
                color:#69f0ae;text-shadow:0 0 9px rgba(105,240,174,.5)">97.1%</div>
    <div style="font-size:.6rem;color:rgba(170,192,255,.46);text-transform:uppercase;letter-spacing:.4px">Accuracy</div>
  </div>
  <div style="text-align:center">
    <div style="font-family:'Orbitron',monospace;font-size:.84rem;font-weight:700;
                color:#ffd700;text-shadow:0 0 9px rgba(255,215,0,.5)">12ms</div>
    <div style="font-size:.6rem;color:rgba(170,192,255,.46);text-transform:uppercase;letter-spacing:.4px">Latency</div>
  </div>
</div>""", unsafe_allow_html=True)
        cx()
    with r:
        co(); sh("Live Attack Probability Timeline")
        st.plotly_chart(chart_timeline(ts), use_container_width=True, key="dt")
        cx()

    st.markdown("<br>", unsafe_allow_html=True)
    fl,fr = st.columns([2,1])
    with fl:
        co(); sh("Network Flow Rate")
        st.plotly_chart(chart_flow(ts), use_container_width=True, key="dfl")
        cx()
    with fr:
        co(); sh("Attack Type Distribution")
        st.plotly_chart(chart_pie(stats["label_distribution"]), use_container_width=True, key="dp")
        cx()

    st.markdown("<br>", unsafe_allow_html=True)
    co(); sh("Recent Security Alerts")
    alerts = get_recent_alerts(8)
    hh = st.columns([1.2,2.5,1.8,1.2,1.2])
    for hc,h in zip(hh,["Time","Alert Type","Source IP","Severity","Confidence"]):
        hc.markdown(f"<div style='font-size:.62rem;font-weight:700;color:rgba(170,192,255,.38);"
                    f"text-transform:uppercase;letter-spacing:1px;padding:3px 0'>{h}</div>",
                    unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,"
                "rgba(79,139,249,.28),transparent);margin:3px 0 7px'></div>",
                unsafe_allow_html=True)
    for a in alerts:
        ac = st.columns([1.2,2.5,1.8,1.2,1.2])
        ac[0].markdown(f"<div style='font-size:.75rem;color:rgba(170,192,255,.42);font-family:monospace'>{a['time']}</div>", unsafe_allow_html=True)
        ac[1].markdown(f"<div style='font-size:.82rem;font-weight:500;color:#c8d8ff'>{a['type']}</div>", unsafe_allow_html=True)
        ac[2].markdown(f"<div style='font-size:.76rem;color:#4F8BF9;font-family:monospace'>{a['source_ip']}</div>", unsafe_allow_html=True)
        ac[3].markdown(badge(a["severity"]), unsafe_allow_html=True)
        gc = "#69f0ae" if a["confidence"]>.85 else "#ffd700"
        ac[4].markdown(f"<div style='font-size:.82rem;font-weight:700;color:{gc}'>{a['confidence']:.0%}</div>", unsafe_allow_html=True)
    cx()


def page_data_ingestion():
    ph("📂","Data Ingestion","Upload PCAP / CSV and extract CIC-IDS2018 feature vectors")

    c1,c2 = st.columns(2)
    with c1:
        co(); sh("Upload Network Data")
        st.markdown("""
<div style="background:rgba(79,139,249,.04);border:2px dashed rgba(79,139,249,.26);
            border-radius:11px;padding:20px;text-align:center;margin-bottom:13px">
  <div style="font-size:2rem;filter:drop-shadow(0 0 12px rgba(79,139,249,.65))">📁</div>
  <div style="font-size:.86rem;font-weight:600;color:#4F8BF9;margin-top:5px">Drop your file here</div>
  <div style="font-size:.7rem;color:rgba(170,192,255,.42);margin-top:2px">CSV · PCAP · PCAPNG</div>
</div>""", unsafe_allow_html=True)
        up = st.file_uploader("f", type=["csv","pcap","pcapng"], label_visibility="collapsed")
        if st.button("🔄  Load Demo Dataset (CIC-IDS2018)", use_container_width=True):
            st.session_state["df"] = generate_mock_traffic_data(500)
            st.success("✅ Demo dataset loaded — 500 flows")
        if up:
            if up.name.endswith(".csv"):
                try:
                    st.session_state["df"] = parse_uploaded_csv(up)
                    st.success(f"✅ Loaded {len(st.session_state['df']):,} rows")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.session_state["df"] = generate_mock_traffic_data(500)
                st.info("PCAP parsing needs Scapy — loaded synthetic demo.")
        cx()

    with c2:
        co(); sh("Feature Extraction Pipeline")
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
            bg = "rgba(105,240,174,.035)" if done else "rgba(255,215,0,.035)"
            bd = "rgba(105,240,174,.1)"   if done else "rgba(255,215,0,.07)"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;padding:5px 9px;"
                f"border-radius:7px;margin-bottom:3px;background:{bg};border:1px solid {bd}'>"
                f"<span>{icon}</span>"
                f"<span style='font-size:.8rem;color:#c8d8ff;flex:1'>{feat}</span>"
                f"<span style='font-size:.65rem;font-weight:700;color:{c};"
                f"text-transform:uppercase;letter-spacing:.4px'>{'Done' if done else 'Pending'}</span></div>",
                unsafe_allow_html=True)
        cx()

    st.markdown("<br>", unsafe_allow_html=True)
    dfs = st.session_state.get("df", generate_mock_traffic_data(100))
    co(); sh("Data Explorer")
    t1,t2,t3 = st.tabs(["📋  Raw Data","📊  Statistics","🌐  3D Scatter"])
    with t1:
        n = st.slider("Rows",10,100,30,key="din")
        st.dataframe(dfs.head(n), use_container_width=True, height=310)
        ca,cb,cc_ = st.columns(3)
        ca.metric("Rows",f"{len(dfs):,}"); cb.metric("Columns",len(dfs.columns))
        cc_.metric("Memory",f"{dfs.memory_usage(deep=True).sum()/1024:.1f} KB")
    with t2:
        num = dfs.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(dfs[num].describe().round(3), use_container_width=True, height=310)
    with t3:
        st.plotly_chart(chart_scatter3d(dfs), use_container_width=True, key="d3d")
    cx()

    st.markdown("<br>", unsafe_allow_html=True)
    co(); sh("Extracted Feature Matrix (Top 16)")
    st.dataframe(extract_features(dfs).head(20), use_container_width=True, height=270)
    cx()


def page_model_training():
    ph("🧠","World Model Training","Configure & monitor LSTM / Transformer world model")

    c1,c2 = st.columns(2)
    with c1:
        co(); sh("Training Configuration")
        arch   = st.selectbox("Architecture", list(MODEL_ARCHITECTURES.keys()), key="ta")
        epochs = st.slider("Epochs",10,100,50,key="te")
        st.selectbox("Batch Size",[32,64,128,256],index=1)
        st.select_slider("Learning Rate",[1e-4,5e-4,1e-3,5e-3],value=1e-3,
                         format_func=lambda x:f"{x:.0e}")
        st.slider("Sequence Length",5,50,20)
        st.slider("K-step Horizon",1,20,10)
        st.slider("Dropout",0.0,0.6,0.3,.05)
        st.checkbox("Early Stopping (patience=5)",value=True)
        if st.button("🚀  Start Training", use_container_width=True):
            pb   = st.progress(0)
            stxt = st.empty()
            for i in range(epochs):
                pb.progress((i+1)/epochs)
                stxt.markdown(
                    f"<div style='font-size:.78rem;color:#4F8BF9;font-family:monospace'>"
                    f"Epoch {i+1:03d}/{epochs} &nbsp;│&nbsp; "
                    f"loss <span style='color:#ffd700'>{0.72*np.exp(-0.065*(i+1))+0.08:.4f}</span>"
                    f" &nbsp;│&nbsp; val_loss "
                    f"<span style='color:#ff4444'>{0.75*np.exp(-0.058*(i+1))+0.10:.4f}</span>"
                    f"</div>", unsafe_allow_html=True)
                time.sleep(0.03)
            stxt.markdown("<div style='color:#69f0ae;font-weight:700'>✅ Training complete!</div>",
                          unsafe_allow_html=True)
        cx()

    with c2:
        co(); sh("Model Architecture")
        ad = MODEL_ARCHITECTURES.get(arch, list(MODEL_ARCHITECTURES.values())[0])
        for i,layer in enumerate(ad["layers"]):
            hl  = "LSTM" in layer["name"] or "Attention" in layer["name"]
            bg  = "rgba(79,139,249,.09)" if hl else "rgba(255,255,255,.02)"
            brd = "rgba(79,139,249,.3)"  if hl else "rgba(79,139,249,.07)"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:7px 11px;border-radius:7px;margin-bottom:3px;"
                f"background:{bg};border:1px solid {brd}'>"
                f"<span style='font-size:.79rem;color:#c8d8ff;font-weight:500'>{i+1}. {layer['name']}</span>"
                f"<span style='font-size:.72rem;color:#4F8BF9'>{layer['units']}</span>"
                f"<span style='font-size:.68rem;color:rgba(170,192,255,.42)'>{layer['activation']}</span></div>",
                unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("Parameters",ad["params"]); m2.metric("Optimizer",ad["optimizer"].split(" ")[0]); m3.metric("Loss","Binary CE")
        cx()

    st.markdown("<br>", unsafe_allow_html=True)
    co(); sh("Training Curves")
    akey = "LSTM" if "LSTM" in arch else "Transformer"
    st.plotly_chart(chart_training(generate_training_curves(epochs=epochs,model_type=akey)),
                    use_container_width=True, key="tc")
    perf = get_model_performance_metrics(arch)
    for col,(lbl,key) in zip(st.columns(6),
            [("Accuracy","accuracy"),("Precision","precision"),("Recall","recall"),
             ("F1","f1_score"),("FPR","fpr"),("AUC","auc_roc")]):
        col.metric(lbl, f"{perf[key]:.4f}")
    cx()


def page_attack_prediction():
    ph("🔮","Attack Prediction","K-step forward simulation — predict infiltration before it completes")

    c1,c2 = st.columns(2)
    with c1:
        co(); sh("Forecast Configuration")
        k    = st.slider("Forecast Horizon (K steps)",3,20,10,key="pk")
        base = st.slider("Current Base Probability",.1,.9,.45,.05,key="pb")
        st.selectbox("Attack Type Filter",["All"]+ATTACK_LABELS[1:],key="pat")
        if st.button("▶  Run Forecast", use_container_width=True):
            with st.spinner("Running K-step world model simulation…"):
                time.sleep(.5)
            st.success("✅ Forecast complete")
        fdf = generate_k_step_forecast(k=k, base_prob=base)
        sh("Step-by-Step Predictions")
        for _, row in fdf.iterrows():
            pv = float(row["Attack Probability"])
            pc = "#ff4444" if pv>.7 else "#ffd700" if pv>.5 else "#69f0ae"
            sc = SC.get(row["Predicted Stage"],"#c8d8ff")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;padding:5px 9px;"
                f"border-radius:7px;margin-bottom:3px;background:rgba(79,139,249,.025);"
                f"border:1px solid rgba(79,139,249,.08)'>"
                f"<span style='font-size:.76rem;color:rgba(170,192,255,.4);font-family:monospace;"
                f"min-width:40px'>t+{row['Step']}</span>"
                f"<div style='width:6px;height:6px;border-radius:50%;background:{sc};"
                f"box-shadow:0 0 5px {sc};flex-shrink:0'></div>"
                f"<span style='font-size:.77rem;color:{sc};flex:1'>{row['Predicted Stage']}</span>"
                f"<span style='font-size:.78rem;font-weight:700;color:{pc};font-family:monospace'>"
                f"{pv:.4f}</span></div>", unsafe_allow_html=True)
        cx()

    with c2:
        co(); sh(f"Step t+{k} Threat Gauge")
        fdf2 = generate_k_step_forecast(k=k, base_prob=base)
        st.plotly_chart(chart_gauge(float(fdf2["Attack Probability"].iloc[-1]),f"Step t+{k}"),
                        use_container_width=True, key="pg")
        sh("Stage Frequency")
        for stage,cnt in fdf2["Predicted Stage"].value_counts().items():
            cc2 = SC.get(stage,"#4F8BF9")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;padding:5px 0;"
                f"border-bottom:1px solid rgba(79,139,249,.06)'>"
                f"<div style='width:8px;height:8px;border-radius:50%;background:{cc2};"
                f"box-shadow:0 0 4px {cc2}'></div>"
                f"<span style='font-size:.81rem;color:#c8d8ff;flex:1'>{stage}</span>"
                f"<span style='font-weight:700;color:{cc2};font-size:.86rem'>{cnt} steps</span></div>",
                unsafe_allow_html=True)
        cx()

    st.markdown("<br>", unsafe_allow_html=True)
    co(); sh("K-Step Forecast Chart")
    st.plotly_chart(chart_kstep(fdf2), use_container_width=True, key="pkc")
    cx()

    st.markdown("<br>", unsafe_allow_html=True)
    co(); sh("Infiltration Probability Summary")
    i1,i2,i3,i4 = st.columns(4)
    i1.metric("Min",  f"{fdf2['Attack Probability'].min():.4f}")
    i2.metric("Max",  f"{fdf2['Attack Probability'].max():.4f}")
    i3.metric("Mean", f"{fdf2['Attack Probability'].mean():.4f}")
    i4.metric("Steps > 0.7", int((fdf2["Attack Probability"]>.7).sum()))
    cx()


def page_mitre_attack():
    ph("🗺️","MITRE ATT&CK Mapping","Kill chain progression · technique library · detection heatmap")

    c1,c2 = st.columns(2)
    with c1:
        co(); sh("Kill Chain Progression")
        stage_sel = st.selectbox("Predicted Stage",[s["name"] for s in MITRE_STAGES],index=3)
        for s in get_kill_chain_progression(stage_sel):
            icon = "✅" if s["status"]=="completed" else ("🔴" if s["status"]=="active" else "⬜")
            sc   = "#69f0ae" if s["status"]=="completed" else ("#ff4444" if s["status"]=="active" else "rgba(170,192,255,.26)")
            st.markdown(
                f"<div class='kc kc-{s['status']}'>"
                f"<div style='display:flex;align-items:center;gap:10px'>"
                f"<span style='font-size:1.05rem;filter:drop-shadow(0 0 6px {sc})'>{s['icon']}</span>"
                f"<div style='flex:1'>"
                f"<div style='font-size:.84rem;font-weight:600;color:#c8d8ff'>{icon} {s['name']}</div>"
                f"<div style='font-size:.68rem;color:rgba(170,192,255,.48);margin-top:1px'>{s['description'][:60]}…</div></div>"
                f"<span style='font-size:.65rem;font-weight:700;color:{sc};text-transform:uppercase;letter-spacing:.4px'>{s['status']}</span>"
                f"</div></div>", unsafe_allow_html=True)
        cx()
    with c2:
        co(); sh("Stage Probability Radar")
        st.plotly_chart(chart_mitre_radar(generate_stage_probability_vector(stage_sel)),
                        use_container_width=True, key="mr")
        cx()

    st.markdown("<br>", unsafe_allow_html=True)
    c3,c4 = st.columns(2)
    with c3:
        co(); sh(f"ATT&CK Techniques — {stage_sel}")
        st.dataframe(get_technique_details_for_stage(stage_sel), use_container_width=True, hide_index=True)
        cx()
    with c4:
        co(); sh("Network Indicators of Compromise")
        sd = next(s for s in MITRE_STAGES if s["name"]==stage_sel)
        for ind in sd["network_indicators"]:
            st.markdown(
                f"<div style='padding:6px 0;border-bottom:1px solid rgba(79,139,249,.06);"
                f"display:flex;align-items:center;gap:8px'>"
                f"<span style='color:#ff4444;font-size:.82rem'>⚠</span>"
                f"<span style='font-size:.8rem;color:#c8d8ff'>{ind}</span></div>",
                unsafe_allow_html=True)
        sh("Detection Features")
        for feat in sd["detection_features"]:
            st.markdown(
                f"<span style='display:inline-block;background:rgba(79,139,249,.12);"
                f"border:1px solid rgba(79,139,249,.35);border-radius:20px;"
                f"padding:2px 12px;margin:3px;font-size:.74rem;color:#4F8BF9;"
                f"font-weight:600;box-shadow:0 0 6px rgba(79,139,249,.18)'>{feat}</span>",
                unsafe_allow_html=True)
        cx()

    st.markdown("<br>", unsafe_allow_html=True)
    co(); sh("MITRE ATT&CK Detection Frequency Heatmap")
    st.plotly_chart(chart_mitre_heatmap(), use_container_width=True, key="mh")
    cx()


def page_explainability():
    ph("💡","Explainability","SHAP values · attention weights · feature attribution")

    t1,t2,t3 = st.tabs(["🎯  SHAP Analysis","📊  Feature Importance","🔍  Attention Weights"])

    with t1:
        co(); sh("SHAP Beeswarm Plot")
        n_s  = st.slider("Samples",10,50,20,key="xn")
        shdf = get_shap_values(n_s)
        st.plotly_chart(chart_beeswarm(shdf), use_container_width=True, key="xbee")
        sh("Mean |SHAP| Values")
        st.plotly_chart(chart_shap_bar(shdf), use_container_width=True, key="xms")
        cx()

    with t2:
        co(); sh("Global Feature Importance")
        tn = st.slider("Top N Features",5,16,16,key="xtn")
        st.plotly_chart(chart_feat_imp(FEATURE_IMPORTANCE_DATA,tn), use_container_width=True, key="xfi")
        fi_df = (pd.DataFrame(list(FEATURE_IMPORTANCE_DATA.items()),columns=["Feature","Importance"])
                   .sort_values("Importance",ascending=False).reset_index(drop=True))
        fi_df.index += 1
        fi_df["Importance"] = fi_df["Importance"].apply(lambda x:f"{x:.3f}")
        st.dataframe(fi_df, use_container_width=True, height=350)
        cx()

    with t3:
        co(); sh("Transformer Self-Attention Heatmap")
        sl = st.slider("Sequence Length",5,20,15,key="xsl")
        st.plotly_chart(chart_attention(get_attention_weights(sl)), use_container_width=True, key="xattn")
        st.info("Brighter cells = higher attention weight. The model focuses on recent "
                "timesteps when predicting attack progression through the kill chain.")
        cx()


def page_benchmark():
    ph("📈","Benchmark","World Model vs LR vs ensemble baselines — full performance analysis")

    bdf = get_benchmark_dataframe()
    co(); sh("Model Performance Comparison")
    st.dataframe(bdf.style.format(
        {"F1 Score":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}",
         "Accuracy":"{:.4f}","FPR":"{:.4f}","AUC-ROC":"{:.4f}"}),
        use_container_width=True, hide_index=True, height=255)
    cx()

    st.markdown("<br>", unsafe_allow_html=True)
    bc1,bc2 = st.columns(2)
    with bc1:
        co()
        m1 = st.selectbox("Metric",["F1 Score","Precision","Recall","Accuracy","AUC-ROC"],key="bm1")
        st.plotly_chart(chart_bench_bar(bdf,m1), use_container_width=True, key="bb1")
        cx()
    with bc2:
        co()
        m2 = st.selectbox("Metric",["AUC-ROC","FPR","Accuracy","F1 Score"],key="bm2")
        st.plotly_chart(chart_bench_bar(bdf,m2), use_container_width=True, key="bb2")
        cx()

    st.markdown("<br>", unsafe_allow_html=True)
    r1,r2 = st.columns([3,2])
    with r1:
        co(); sh("ROC Curves")
        st.plotly_chart(chart_roc(get_roc_curve_data()), use_container_width=True, key="broc")
        cx()
    with r2:
        co(); sh("Confusion Matrix")
        cms = st.selectbox("Model",["World Model (LSTM)","World Model (Transformer)","Logistic Regression"],key="bcm")
        st.plotly_chart(chart_confusion(get_confusion_matrix(cms),cms), use_container_width=True, key="bcmf")
        p = get_model_performance_metrics(cms)
        cc1,cc2 = st.columns(2)
        cc1.metric("Precision",f"{p['precision']:.4f}"); cc2.metric("Recall",f"{p['recall']:.4f}")
        cc1.metric("F1 Score", f"{p['f1_score']:.4f}"); cc2.metric("FPR",   f"{p['fpr']:.4f}")
        cx()

    st.markdown("<br>", unsafe_allow_html=True)
    co(); sh("Multi-Model Performance Radar")
    sel = st.multiselect("Select models",BENCHMARK_MODELS,
                         default=["World Model (LSTM)","World Model (Transformer)","Logistic Regression","XGBoost"])
    if len(sel) >= 2:
        st.plotly_chart(chart_bench_radar(BENCHMARK_METRICS,sel), use_container_width=True, key="brad")
    else:
        st.info("Select at least 2 models to compare.")
    cx()


def page_settings():
    ph("⚙️","Settings","Model config · detection thresholds · system information")

    c1,c2 = st.columns(2)
    with c1:
        co(); sh("Model Configuration")
        st.selectbox("Active Model",list(MODEL_ARCHITECTURES.keys()))
        st.number_input("Sequence Window Length",5,100,20)
        st.number_input("K-step Forecast Horizon",1,30,10)
        st.number_input("LSTM Hidden Units",64,512,256,step=64)
        st.number_input("LSTM Layers",1,6,2)
        st.select_slider("Dropout",[0.0,.1,.2,.3,.4,.5],value=.3)
        st.selectbox("Optimizer",["Adam","AdamW","SGD","RMSprop"])
        if st.button("💾  Save Model Config", use_container_width=True): st.success("✅ Saved")
        cx()
    with c2:
        co(); sh("Detection Thresholds")
        st.slider("Alert Threshold",.3,.95,.70,.05)
        st.slider("Critical Threshold",.5,.99,.85,.05)
        st.slider("Anomaly Score Threshold",.2,.95,.60,.05)
        st.slider("Min Confidence",.5,.99,.65,.05)
        st.markdown("<hr>", unsafe_allow_html=True)
        sh("Alert Settings")
        st.checkbox("Enable Email Alerts"); st.checkbox("Enable Webhook Notifications")
        st.checkbox("Auto-block Suspicious IPs")
        st.selectbox("Aggregation Window",["1 min","5 min","15 min","1 hour"],index=1)
        if st.button("💾  Save Threshold Config", use_container_width=True): st.success("✅ Saved")
        cx()

    st.markdown("<br>", unsafe_allow_html=True)
    c3,c4 = st.columns(2)
    with c3:
        co(); sh("Data Pipeline")
        st.selectbox("Input Source",["File Upload","Live PCAP","Network Tap","SIEM Integration"])
        st.number_input("Max Flow Buffer",1000,100000,10000,step=1000)
        st.selectbox("Normalization",["MinMax","Standard","Robust","None"])
        st.checkbox("Apply PCA Reduction")
        st.number_input("Train Split (%)",60,80,70)
        if st.button("💾  Save Data Config", use_container_width=True): st.success("✅ Saved")
        cx()
    with c4:
        co(); sh("System Information")
        ms = get_model_status()
        for k,v in [("App Version","1.0.0"),("Model",ms["model_name"]),("Version",ms["version"]),
                    ("Dataset",ms["dataset"]),("Parameters",ms["total_params"]),
                    ("Last Trained",ms["last_trained"]),("Epochs",str(ms["epochs_trained"])),
                    ("Device",ms["device"])]:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
                f"border-bottom:1px solid rgba(79,139,249,.06)'>"
                f"<span style='font-size:.78rem;color:rgba(170,192,255,.48)'>{k}</span>"
                f"<span style='font-size:.78rem;font-weight:600;color:#c8d8ff'>{v}</span></div>",
                unsafe_allow_html=True)
        cx()


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    inject_css()
    if "df" not in st.session_state:
        st.session_state["df"] = generate_mock_traffic_data(300)

    page = render_sidebar()

    {
        "Dashboard":         page_dashboard,
        "Data Ingestion":    page_data_ingestion,
        "Model Training":    page_model_training,
        "Attack Prediction": page_attack_prediction,
        "MITRE ATT&CK":      page_mitre_attack,
        "Explainability":    page_explainability,
        "Benchmark":         page_benchmark,
        "Settings":          page_settings,
    }.get(page, page_dashboard)()


if __name__ == "__main__":
    main()
