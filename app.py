"""
AI-Based Network Attack Forecasting from Network Traffic Data
SIH Project — Main Streamlit Application
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── page config must be first ──────────────────────────────────
st.set_page_config(
    page_title="NetGuard AI — Attack Forecasting",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── utils ──────────────────────────────────────────────────────
from utils.data_processor import (
    generate_mock_traffic_data,
    parse_uploaded_csv,
    extract_features,
    compute_flow_statistics,
    generate_time_series_data,
    generate_k_step_forecast,
    get_recent_alerts,
    TOP_FEATURES,
    ATTACK_LABELS,
)
from utils.model_utils import (
    generate_training_curves,
    get_model_performance_metrics,
    get_confusion_matrix,
    FEATURE_IMPORTANCE_DATA,
    get_shap_values,
    get_attention_weights,
    get_benchmark_dataframe,
    get_roc_curve_data,
    BENCHMARK_METRICS,
    MODEL_ARCHITECTURES,
    get_model_status,
    BENCHMARK_MODELS,
)
from utils.mitre_attack import (
    MITRE_STAGES,
    get_kill_chain_progression,
    generate_stage_probability_vector,
    get_technique_details_for_stage,
    get_mitre_heatmap_data,
    map_label_to_stage,
)
from utils.visualizations import (
    attack_probability_gauge,
    live_timeline_chart,
    flow_rate_chart,
    k_step_forecast_chart,
    training_curves_chart,
    feature_importance_chart,
    shap_beeswarm_chart,
    attention_heatmap,
    confusion_matrix_chart,
    benchmark_bar_chart,
    roc_curve_chart,
    label_distribution_pie,
    mitre_stage_radar,
    network_traffic_3d,
    benchmark_radar_chart,
    COLORS,
)


# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main background ── */
    .stApp {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 50%, #f5f0ff 100%);
        background-attachment: fixed;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        border-right: 1px solid rgba(79,139,249,0.3);
    }
    [data-testid="stSidebar"] * {
        color: #e0e8ff !important;
    }
    [data-testid="stSidebarNav"] {
        padding-top: 0.5rem;
    }

    /* ── Radio buttons (nav) ── */
    [data-testid="stSidebar"] .stRadio > label {
        display: none;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 10px 14px;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid transparent;
        display: flex !important;
        align-items: center;
        gap: 8px;
        font-size: 0.92rem;
        font-weight: 500;
        color: #b0c4ff !important;
    }
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
        background: rgba(79,139,249,0.2);
        border-color: rgba(79,139,249,0.4);
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stRadio label[aria-checked="true"] {
        background: linear-gradient(135deg, rgba(79,139,249,0.35), rgba(124,77,255,0.25));
        border-color: rgba(79,139,249,0.6);
        color: #ffffff !important;
        box-shadow: 0 0 12px rgba(79,139,249,0.3);
    }
    [data-testid="stSidebar"] .stRadio span[data-testid="stMarkdownContainer"] {
        display: none;
    }

    /* ── Cards ── */
    .ng-card {
        background: rgba(255,255,255,0.92);
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 4px 24px rgba(79,139,249,0.10), 0 1px 4px rgba(0,0,0,0.04);
        border: 1px solid rgba(79,139,249,0.12);
        backdrop-filter: blur(8px);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        margin-bottom: 6px;
    }
    .ng-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(79,139,249,0.16), 0 2px 8px rgba(0,0,0,0.06);
    }

    /* ── Metric cards ── */
    .metric-card {
        background: rgba(255,255,255,0.95);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(79,139,249,0.10);
        border: 1px solid rgba(79,139,249,0.12);
        transition: transform 0.15s ease;
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #5a6a8a;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-delta {
        font-size: 0.72rem;
        margin-top: 2px;
    }

    /* ── Page headers ── */
    .page-header {
        background: linear-gradient(135deg, #4F8BF9 0%, #7C4DFF 100%);
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 6px 30px rgba(79,139,249,0.3);
    }
    .page-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    .page-header p {
        font-size: 0.88rem;
        opacity: 0.85;
        margin: 6px 0 0 0;
        color: rgba(255,255,255,0.9);
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.0rem;
        font-weight: 600;
        color: #1a1a2e;
        border-left: 3px solid #4F8BF9;
        padding-left: 10px;
        margin: 18px 0 12px 0;
    }

    /* ── Alert badges ── */
    .badge-critical { background: #ffebee; color: #c62828; border: 1px solid #ef9a9a;
                      border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; }
    .badge-high     { background: #fff3e0; color: #e65100; border: 1px solid #ffcc80;
                      border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; }
    .badge-medium   { background: #fffde7; color: #f57f17; border: 1px solid #fff176;
                      border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; }
    .badge-low      { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7;
                      border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; font-weight: 600; }

    /* ── Kill chain ── */
    .kc-stage {
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 8px;
        border: 1px solid rgba(79,139,249,0.15);
        background: rgba(255,255,255,0.9);
        transition: all 0.2s ease;
    }
    .kc-completed {
        border-left: 4px solid #43A047;
        opacity: 0.85;
    }
    .kc-active {
        border-left: 4px solid #E53935;
        box-shadow: 0 0 16px rgba(229,57,53,0.25);
        background: rgba(255,235,238,0.6);
    }
    .kc-pending {
        border-left: 4px solid rgba(79,139,249,0.3);
        opacity: 0.6;
    }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid rgba(79,139,249,0.15); margin: 16px 0; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #f0f4ff; }
    ::-webkit-scrollbar-thumb { background: rgba(79,139,249,0.4); border-radius: 10px; }

    /* ── Tab styles ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.7);
        border-radius: 10px;
        padding: 4px;
        border: 1px solid rgba(79,139,249,0.15);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
        color: #5a6a8a;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4F8BF9, #7C4DFF);
        color: white !important;
    }

    /* ── Selectbox / inputs ── */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(79,139,249,0.3);
        border-radius: 8px;
    }
    .stSlider > div { color: #4F8BF9; }
    .stButton > button {
        background: linear-gradient(135deg, #4F8BF9 0%, #7C4DFF 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 0.88rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(79,139,249,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(79,139,249,0.4);
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div { background: linear-gradient(90deg, #4F8BF9, #7C4DFF); }

    /* ── DataFrames ── */
    .dataframe { border-radius: 10px !important; }

    /* ── Hide Streamlit default elements ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stToolbar"] { display: none; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  ANIMATED BACKGROUND
# ══════════════════════════════════════════════════════════════

def inject_animated_background():
    st.markdown("""
    <div id="ng-canvas-wrapper" style="
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none; z-index: 0; overflow: hidden;">
      <canvas id="ngCanvas"></canvas>
    </div>
    <script>
    (function() {
      const canvas = document.getElementById('ngCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');

      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;

      window.addEventListener('resize', () => {
        canvas.width  = window.innerWidth;
        canvas.height = window.innerHeight;
      });

      const NODE_COUNT  = 55;
      const MAX_DIST    = 160;
      const PULSE_SPEED = 0.018;

      const nodes = Array.from({ length: NODE_COUNT }, () => ({
        x:   Math.random() * canvas.width,
        y:   Math.random() * canvas.height,
        vx:  (Math.random() - 0.5) * 0.55,
        vy:  (Math.random() - 0.5) * 0.55,
        r:   2 + Math.random() * 2.5,
        phase: Math.random() * Math.PI * 2,
        isRed: Math.random() < 0.12,
      }));

      let frame = 0;

      function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        frame++;

        // ── Move nodes ──
        nodes.forEach(n => {
          n.x += n.vx;
          n.y += n.vy;
          if (n.x < 0 || n.x > canvas.width)  n.vx *= -1;
          if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
        });

        // ── Draw edges ──
        for (let i = 0; i < nodes.length; i++) {
          for (let j = i + 1; j < nodes.length; j++) {
            const dx   = nodes[i].x - nodes[j].x;
            const dy   = nodes[i].y - nodes[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < MAX_DIST) {
              const alpha = (1 - dist / MAX_DIST) * 0.18;
              const hasRed = nodes[i].isRed || nodes[j].isRed;
              ctx.beginPath();
              ctx.moveTo(nodes[i].x, nodes[i].y);
              ctx.lineTo(nodes[j].x, nodes[j].y);
              ctx.strokeStyle = hasRed
                ? `rgba(229,57,53,${alpha * 1.5})`
                : `rgba(79,139,249,${alpha})`;
              ctx.lineWidth = hasRed ? 1.2 : 0.8;
              ctx.stroke();
            }
          }
        }

        // ── Draw nodes ──
        nodes.forEach(n => {
          const pulse = 0.7 + 0.3 * Math.sin(frame * PULSE_SPEED + n.phase);
          const radius = n.r * pulse;

          // Outer glow
          const grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, radius * 4);
          if (n.isRed) {
            grad.addColorStop(0, `rgba(229,57,53,0.25)`);
            grad.addColorStop(1, 'rgba(229,57,53,0)');
          } else {
            grad.addColorStop(0, `rgba(79,139,249,0.18)`);
            grad.addColorStop(1, 'rgba(79,139,249,0)');
          }
          ctx.beginPath();
          ctx.arc(n.x, n.y, radius * 4, 0, Math.PI * 2);
          ctx.fillStyle = grad;
          ctx.fill();

          // Core dot
          ctx.beginPath();
          ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
          ctx.fillStyle = n.isRed ? 'rgba(229,57,53,0.75)' : 'rgba(79,139,249,0.65)';
          ctx.fill();
        });

        requestAnimationFrame(draw);
      }
      draw();
    })();
    </script>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    with st.sidebar:
        # Logo / title
        st.markdown("""
        <div style="text-align:center; padding: 20px 0 18px 0;">
          <div style="font-size:2.4rem; margin-bottom:6px;">🛡️</div>
          <div style="font-size:1.05rem; font-weight:700; color:#fff; letter-spacing:0.5px;">NetGuard AI</div>
          <div style="font-size:0.72rem; color:rgba(176,196,255,0.75); margin-top:2px;">Network Attack Forecasting</div>
        </div>
        <hr style="border-color:rgba(79,139,249,0.2); margin:0 0 14px 0;">
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

        selected = st.radio("Navigation", pages, label_visibility="collapsed")

        # Status indicator at the bottom
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        model_status = get_model_status()
        st.markdown(f"""
        <div style="background:rgba(79,139,249,0.12); border-radius:12px; padding:12px 14px;
                    border:1px solid rgba(79,139,249,0.25); margin-top:10px;">
          <div style="font-size:0.7rem; color:rgba(176,196,255,0.8); text-transform:uppercase;
                      letter-spacing:0.5px; margin-bottom:6px;">Model Status</div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="width:8px; height:8px; border-radius:50%; background:#43A047;
                         box-shadow:0 0 6px #43A047; display:inline-block;"></span>
            <span style="font-size:0.82rem; font-weight:600; color:#e0e8ff;">{model_status['status']}</span>
          </div>
          <div style="font-size:0.72rem; color:rgba(176,196,255,0.65); margin-top:4px;">
            {model_status['model_name']}<br>{model_status['version']}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; margin-top:20px; font-size:0.68rem; color:rgba(176,196,255,0.4);">
          SIH Project · CIC-IDS2018
        </div>
        """, unsafe_allow_html=True)

    return selected.split("  ", 1)[1].strip()


# ══════════════════════════════════════════════════════════════
#  HELPER COMPONENTS
# ══════════════════════════════════════════════════════════════

def metric_card(label: str, value: str, delta: str = "", color: str = "#4F8BF9") -> str:
    delta_html = ""
    if delta:
        up = delta.startswith("+")
        delta_color = "#43A047" if up else "#E53935"
        delta_html = f'<div class="metric-delta" style="color:{delta_color};">{delta}</div>'
    return f"""
    <div class="metric-card">
      <div class="metric-value" style="color:{color};">{value}</div>
      <div class="metric-label">{label}</div>
      {delta_html}
    </div>"""


def page_header(icon: str, title: str, subtitle: str):
    st.markdown(f"""
    <div class="page-header">
      <h1>{icon} {title}</h1>
      <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)


def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def alert_badge(severity: str) -> str:
    cls_map = {"Critical": "badge-critical", "High": "badge-high",
               "Medium": "badge-medium", "Low": "badge-low"}
    return f'<span class="{cls_map.get(severity, "badge-low")}">{severity}</span>'


# ══════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════

def page_dashboard():
    page_header("📊", "Dashboard", "Real-time overview of network security posture and attack forecasts")

    # ── Top metrics row ──
    ts_data = generate_time_series_data(60)
    df_mock = generate_mock_traffic_data(300)
    stats = compute_flow_statistics(df_mock)
    latest_prob = float(ts_data["Attack Probability"].iloc[-1])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(metric_card("Total Flows", f"{stats['total_flows']:,}", "+12.4%", "#4F8BF9"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Attack Flows", f"{stats['attack_flows']:,}", f"+{stats['attack_rate']}%", "#E53935"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Benign Flows", f"{stats['benign_flows']:,}", "-5.2%", "#43A047"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Avg Flow Rate", f"{stats['avg_flow_bytes_s']/1000:.1f}K B/s", "+8.1%", "#FF8F00"), unsafe_allow_html=True)
    with c5:
        st.markdown(metric_card("Avg Duration", f"{stats['avg_duration_ms']:.0f} ms", "-2.3%", "#00ACC1"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + Live timeline ──
    col_gauge, col_timeline = st.columns([1, 2])
    with col_gauge:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Current Threat Level")
        fig_gauge = attack_probability_gauge(latest_prob)
        st.plotly_chart(fig_gauge, use_container_width=True, key="dash_gauge")
        # Quick stats below gauge
        model_s = get_model_status()
        st.markdown(f"""
        <div style="display:flex; justify-content:space-around; margin-top:8px;">
          <div style="text-align:center;">
            <div style="font-size:1.0rem; font-weight:700; color:#4F8BF9;">{model_s['version']}</div>
            <div style="font-size:0.68rem; color:#5a6a8a;">Model Version</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:1.0rem; font-weight:700; color:#43A047;">97.1%</div>
            <div style="font-size:0.68rem; color:#5a6a8a;">Accuracy</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:1.0rem; font-weight:700; color:#FF8F00;">12.4ms</div>
            <div style="font-size:0.68rem; color:#5a6a8a;">Inference</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_timeline:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Live Attack Probability Timeline")
        fig_tl = live_timeline_chart(ts_data)
        st.plotly_chart(fig_tl, use_container_width=True, key="dash_timeline")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Flow rate + Label distribution ──
    col_flow, col_pie = st.columns([2, 1])
    with col_flow:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Network Flow Rate")
        fig_flow = flow_rate_chart(ts_data)
        st.plotly_chart(fig_flow, use_container_width=True, key="dash_flow")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_pie:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Attack Type Distribution")
        fig_pie = label_distribution_pie(stats["label_distribution"])
        st.plotly_chart(fig_pie, use_container_width=True, key="dash_pie")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Recent Alerts ──
    st.markdown('<div class="ng-card">', unsafe_allow_html=True)
    section_header("Recent Alerts")
    alerts = get_recent_alerts(8)

    header_cols = st.columns([1.2, 2.5, 1.8, 1.2, 1.2])
    for hc, h in zip(header_cols, ["Time", "Alert Type", "Source IP", "Severity", "Confidence"]):
        hc.markdown(f"<div style='font-size:0.72rem; font-weight:600; color:#5a6a8a; text-transform:uppercase; letter-spacing:0.5px;'>{h}</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:6px 0;'>", unsafe_allow_html=True)

    for alert in alerts:
        ac1, ac2, ac3, ac4, ac5 = st.columns([1.2, 2.5, 1.8, 1.2, 1.2])
        ac1.markdown(f"<div style='font-size:0.83rem; color:#5a6a8a; font-family:monospace;'>{alert['time']}</div>", unsafe_allow_html=True)
        ac2.markdown(f"<div style='font-size:0.85rem; font-weight:500; color:#1a1a2e;'>{alert['type']}</div>", unsafe_allow_html=True)
        ac3.markdown(f"<div style='font-size:0.83rem; color:#4F8BF9; font-family:monospace;'>{alert['source_ip']}</div>", unsafe_allow_html=True)
        ac4.markdown(alert_badge(alert["severity"]), unsafe_allow_html=True)
        conf_color = "#43A047" if alert["confidence"] > 0.85 else "#FF8F00"
        ac5.markdown(f"<div style='font-size:0.85rem; font-weight:600; color:{conf_color};'>{alert['confidence']:.0%}</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: DATA INGESTION
# ══════════════════════════════════════════════════════════════

def page_data_ingestion():
    page_header("📂", "Data Ingestion", "Upload PCAP or CSV files, preview parsed data, and extract features")

    col_up, col_info = st.columns([1, 1])

    with col_up:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Upload Network Data")

        st.markdown("""
        <div style="background:rgba(79,139,249,0.05); border:2px dashed rgba(79,139,249,0.3);
                    border-radius:12px; padding:24px; text-align:center; margin-bottom:14px;">
          <div style="font-size:2rem; margin-bottom:8px;">📁</div>
          <div style="font-size:0.9rem; font-weight:500; color:#4F8BF9;">Drop your file here</div>
          <div style="font-size:0.75rem; color:#5a6a8a; margin-top:4px;">Supported: CSV, PCAP, PCAPNG</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Select file",
            type=["csv", "pcap", "pcapng"],
            label_visibility="collapsed",
        )

        if st.button("🔄 Use Demo Dataset (CIC-IDS2018)", use_container_width=True):
            st.session_state["ingested_df"] = generate_mock_traffic_data(500)
            st.success("Demo dataset loaded — 500 flows from CIC-IDS2018 format.")

        if uploaded is not None:
            if uploaded.name.endswith(".csv"):
                try:
                    df_up = parse_uploaded_csv(uploaded)
                    st.session_state["ingested_df"] = df_up
                    st.success(f"CSV loaded: {len(df_up):,} rows × {len(df_up.columns)} columns")
                except Exception as e:
                    st.error(f"Parse error: {e}")
            else:
                st.warning("PCAP parsing requires scapy (not installed in demo). Loading synthetic data instead.")
                st.session_state["ingested_df"] = generate_mock_traffic_data(500)
                st.info("Synthetic CIC-IDS2018 data loaded for demo.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Feature Extraction Status")

        features_status = [
            ("Flow-level statistics", True),
            ("Packet-level features", True),
            ("TCP flag counters", True),
            ("IAT (Inter-Arrival Time) features", True),
            ("Window size features", True),
            ("Subflow features", True),
            ("Active/Idle time features", True),
            ("Label encoding", True),
            ("Normalization (MinMax)", False),
            ("Sequence windowing (len=20)", False),
        ]

        for feat_name, done in features_status:
            icon = "✅" if done else "⏳"
            color = "#43A047" if done else "#FF8F00"
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:10px; padding:6px 0; "
                f"border-bottom:1px solid rgba(79,139,249,0.08);'>"
                f"<span>{icon}</span>"
                f"<span style='font-size:0.85rem; color:#1a1a2e;'>{feat_name}</span>"
                f"<span style='margin-left:auto; font-size:0.72rem; font-weight:600; color:{color};'>"
                f"{'Done' if done else 'Pending'}</span></div>",
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Data preview ──
    df_show = st.session_state.get("ingested_df", generate_mock_traffic_data(100))

    st.markdown('<div class="ng-card">', unsafe_allow_html=True)
    section_header("Data Preview")

    t1, t2, t3 = st.tabs(["📋 Raw Data", "📊 Statistics", "🔬 3D Scatter"])

    with t1:
        n_show = st.slider("Rows to display", 10, 100, 30, key="di_rows")
        st.dataframe(df_show.head(n_show), use_container_width=True, height=320)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", f"{len(df_show):,}")
        c2.metric("Columns", len(df_show.columns))
        c3.metric("Memory", f"{df_show.memory_usage(deep=True).sum() / 1024:.1f} KB")

    with t2:
        numeric_cols = df_show.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(df_show[numeric_cols].describe().round(3), use_container_width=True, height=320)

    with t3:
        fig3d = network_traffic_3d(df_show, sample_n=300)
        st.plotly_chart(fig3d, use_container_width=True, key="di_3d")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Feature extraction output ──
    st.markdown('<div class="ng-card">', unsafe_allow_html=True)
    section_header("Extracted Feature Matrix (Top 16 Features)")
    feat_df = extract_features(df_show)
    st.dataframe(feat_df.head(20), use_container_width=True, height=280)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: MODEL TRAINING
# ══════════════════════════════════════════════════════════════

def page_model_training():
    page_header("🧠", "World Model Training", "Configure and monitor LSTM/Transformer world model training")

    col_cfg, col_status = st.columns([1, 1])

    with col_cfg:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Training Configuration")

        model_arch   = st.selectbox("Model Architecture", list(MODEL_ARCHITECTURES.keys()), key="train_arch")
        epochs       = st.slider("Epochs", 10, 100, 50, key="train_epochs")
        batch_size   = st.selectbox("Batch Size", [32, 64, 128, 256], index=1, key="train_bs")
        lr           = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                                        value=1e-3, format_func=lambda x: f"{x:.0e}", key="train_lr")
        seq_len      = st.slider("Sequence Length", 5, 50, 20, key="train_seq")
        k_steps_cfg  = st.slider("K-step Forecast Horizon", 1, 20, 10, key="train_k")
        dropout      = st.slider("Dropout Rate", 0.0, 0.6, 0.3, step=0.05, key="train_drop")
        early_stop   = st.checkbox("Early Stopping (patience=5)", value=True, key="train_es")

        if st.button("🚀 Start Training", use_container_width=True, key="btn_train"):
            st.session_state["training_done"] = False
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i in range(epochs):
                progress_bar.progress((i + 1) / epochs)
                status_text.markdown(
                    f"<div style='font-size:0.85rem; color:#4F8BF9;'>Epoch {i+1}/{epochs} — "
                    f"loss: {0.72*np.exp(-0.065*(i+1))+0.08:.4f} — "
                    f"val_loss: {0.75*np.exp(-0.058*(i+1))+0.10:.4f}</div>",
                    unsafe_allow_html=True
                )
                time.sleep(0.04)
            st.session_state["training_done"] = True
            st.session_state["trained_model"] = model_arch
            status_text.markdown(
                "<div style='color:#43A047; font-weight:600;'>✅ Training complete!</div>",
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with col_status:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Model Architecture")

        arch = MODEL_ARCHITECTURES.get(model_arch, list(MODEL_ARCHITECTURES.values())[0])
        for layer in arch["layers"]:
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; padding:7px 0; "
                f"border-bottom:1px solid rgba(79,139,249,0.08);'>"
                f"<span style='font-size:0.84rem; color:#1a1a2e; font-weight:500;'>{layer['name']}</span>"
                f"<span style='font-size:0.78rem; color:#4F8BF9;'>units: {layer['units']}</span>"
                f"<span style='font-size:0.78rem; color:#5a6a8a;'>{layer['activation']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        mc1.metric("Total Parameters", arch["params"])
        mc2.metric("Optimizer", arch["optimizer"].split(" ")[0])
        mc1.metric("Loss Function", "Binary CE")
        mc2.metric("Sequence Length", seq_len)
        mc1.metric("K-step Horizon", k_steps_cfg)
        mc2.metric("Device", "CPU / GPU")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Training curves ──
    st.markdown('<div class="ng-card">', unsafe_allow_html=True)
    section_header("Training Curves")

    arch_key = "LSTM" if "LSTM" in model_arch else "Transformer"
    curves_df = generate_training_curves(epochs=epochs, model_type=arch_key)
    fig_curves = training_curves_chart(curves_df)
    st.plotly_chart(fig_curves, use_container_width=True, key="train_curves")

    # Final metrics
    perf = get_model_performance_metrics(model_arch)
    mc = st.columns(6)
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "FPR", "AUC-ROC"]
    metric_keys   = ["accuracy", "precision", "recall", "f1_score", "fpr", "auc_roc"]
    for col, lbl, key in zip(mc, metric_labels, metric_keys):
        val = perf[key]
        col.metric(lbl, f"{val:.4f}")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: ATTACK PREDICTION
# ══════════════════════════════════════════════════════════════

def page_attack_prediction():
    page_header("🔮", "Attack Prediction", "K-step forward simulation and infiltration probability forecasting")

    col_cfg, col_gauge = st.columns([1, 1])

    with col_cfg:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Forecast Configuration")

        k_steps = st.slider("Forecast Horizon (K steps)", 3, 20, 10, key="pred_k")
        base_prob = st.slider("Current Base Attack Probability", 0.1, 0.9, 0.45, step=0.05, key="pred_base")
        attack_type_filter = st.selectbox("Filter by Attack Type", ["All"] + ATTACK_LABELS[1:], key="pred_at")
        show_stages = st.checkbox("Annotate Attack Stages", value=True, key="pred_ann")

        if st.button("▶ Run Forecast", use_container_width=True, key="btn_pred"):
            with st.spinner("Running K-step forward simulation..."):
                time.sleep(0.8)
            st.success("Forecast complete.")

        forecast_df = generate_k_step_forecast(k=k_steps, base_prob=base_prob)

        # Stage prediction table
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Step-by-Step Stage Predictions")
        display_cols = ["Step", "Attack Probability", "Predicted Stage"]
        styled = forecast_df[display_cols].copy()
        styled["Attack Probability"] = styled["Attack Probability"].apply(lambda x: f"{x:.4f}")
        st.dataframe(styled, use_container_width=True, height=280, hide_index=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_gauge:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Final Step Attack Probability")
        final_prob = float(forecast_df["Attack Probability"].iloc[-1])
        fig_g = attack_probability_gauge(final_prob, f"Step t+{k_steps} Probability")
        st.plotly_chart(fig_g, use_container_width=True, key="pred_gauge")

        # Stage counts
        stage_counts = forecast_df["Predicted Stage"].value_counts()
        section_header("Predicted Stage Frequency")
        for stage, cnt in stage_counts.items():
            col_sc = COLORS["stage_colors"].get(stage, "#4F8BF9")
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:10px; padding:5px 0; "
                f"border-bottom:1px solid rgba(79,139,249,0.08);'>"
                f"<div style='width:10px; height:10px; border-radius:50%; background:{col_sc};'></div>"
                f"<span style='font-size:0.85rem; color:#1a1a2e;'>{stage}</span>"
                f"<span style='margin-left:auto; font-weight:600; color:{col_sc}; font-size:0.9rem;'>{cnt} steps</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main K-step chart ──
    st.markdown('<div class="ng-card">', unsafe_allow_html=True)
    section_header("K-Step Attack Probability Forecast")
    fig_kstep = k_step_forecast_chart(forecast_df)
    st.plotly_chart(fig_kstep, use_container_width=True, key="pred_kstep")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Infiltration probability summary ──
    st.markdown('<div class="ng-card">', unsafe_allow_html=True)
    section_header("Infiltration Probability Summary")
    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.metric("Min Probability", f"{forecast_df['Attack Probability'].min():.4f}")
    ic2.metric("Max Probability", f"{forecast_df['Attack Probability'].max():.4f}")
    ic3.metric("Mean Probability", f"{forecast_df['Attack Probability'].mean():.4f}")
    ic4.metric("Steps Above Threshold (0.7)", int((forecast_df["Attack Probability"] > 0.7).sum()))
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: MITRE ATT&CK
# ══════════════════════════════════════════════════════════════

def page_mitre_attack():
    page_header("🗺️", "MITRE ATT&CK Mapping", "Visual kill chain progression and attack stage classification")

    col_sel, col_radar = st.columns([1, 1])

    with col_sel:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Current Attack Stage")

        predicted_stage = st.selectbox(
            "Select predicted stage",
            [s["name"] for s in MITRE_STAGES],
            index=3,
            key="mitre_stage"
        )
        stage_probs = generate_stage_probability_vector(predicted_stage)

        # Kill chain visual
        section_header("Kill Chain Progression")
        chain = get_kill_chain_progression(predicted_stage)

        for stage in chain:
            css_class = f"kc-{stage['status']}"
            icon = "✅" if stage["status"] == "completed" else ("🔴" if stage["status"] == "active" else "⬜")
            st.markdown(f"""
            <div class="kc-stage {css_class}">
              <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.1rem;">{stage['icon']}</span>
                <div>
                  <div style="font-size:0.88rem; font-weight:600; color:#1a1a2e;">{icon} {stage['name']}</div>
                  <div style="font-size:0.72rem; color:#5a6a8a; margin-top:1px;">{stage['description'][:65]}…</div>
                </div>
                <span style="margin-left:auto; font-size:0.75rem; font-weight:600;
                      color:{'#43A047' if stage['status']=='completed' else ('#E53935' if stage['status']=='active' else '#9E9E9E')};">
                  {stage['status'].upper()}
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_radar:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Stage Probability Distribution")
        fig_radar = mitre_stage_radar(stage_probs)
        st.plotly_chart(fig_radar, use_container_width=True, key="mitre_radar")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stage details ──
    col_tech, col_ioc = st.columns([1, 1])

    with col_tech:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header(f"Techniques for: {predicted_stage}")
        tech_df = get_technique_details_for_stage(predicted_stage)
        st.dataframe(tech_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ioc:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Network Indicators of Compromise")

        stage_data = next(s for s in MITRE_STAGES if s["name"] == predicted_stage)
        for indicator in stage_data["network_indicators"]:
            st.markdown(
                f"<div style='padding:7px 0; border-bottom:1px solid rgba(79,139,249,0.08); "
                f"display:flex; align-items:center; gap:8px;'>"
                f"<span style='color:#E53935; font-size:0.85rem;'>⚠</span>"
                f"<span style='font-size:0.84rem; color:#1a1a2e;'>{indicator}</span></div>",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Key Detection Features")
        for feat in stage_data["detection_features"]:
            st.markdown(
                f"<div style='display:inline-block; background:rgba(79,139,249,0.1); "
                f"border:1px solid rgba(79,139,249,0.25); border-radius:20px; padding:3px 12px; "
                f"margin:3px; font-size:0.78rem; color:#4F8BF9; font-weight:500;'>{feat}</div>",
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MITRE Heatmap ──
    st.markdown('<div class="ng-card">', unsafe_allow_html=True)
    section_header("MITRE ATT&CK Detection Frequency Heatmap")
    heatmap_df = get_mitre_heatmap_data()
    pivot = heatmap_df.pivot(index="Stage", columns="Technique", values="Detection Count").fillna(0)

    import plotly.express as px
    fig_heat = px.imshow(
        pivot,
        color_continuous_scale=[[0, "#EEF2FF"], [0.5, "#4F8BF9"], [1, "#7C4DFF"]],
        aspect="auto",
        title="",
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,244,255,0.5)",
        font=dict(family="Inter, sans-serif"),
        height=280,
        margin=dict(l=20, r=20, t=20, b=60),
        coloraxis_colorbar=dict(title="Count", thickness=12),
        xaxis=dict(tickfont=dict(size=9)),
    )
    st.plotly_chart(fig_heat, use_container_width=True, key="mitre_heat")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════

def page_explainability():
    page_header("💡", "Explainability", "SHAP values, attention weights, and feature importance analysis")

    tab_shap, tab_fi, tab_attn = st.tabs(["🎯 SHAP Analysis", "📊 Feature Importance", "🔍 Attention Weights"])

    with tab_shap:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("SHAP Value Distribution (Beeswarm)")

        n_samples = st.slider("Number of samples for SHAP", 10, 50, 20, key="xai_shap_n")
        shap_df = get_shap_values(n_samples)

        fig_bee = shap_beeswarm_chart(shap_df)
        st.plotly_chart(fig_bee, use_container_width=True, key="xai_bee")

        st.markdown("<hr>", unsafe_allow_html=True)
        section_header("Mean |SHAP| Values")
        mean_shap = shap_df.abs().mean().sort_values(ascending=False).head(10)

        import plotly.express as px
        fig_ms = px.bar(
            x=mean_shap.values, y=mean_shap.index,
            orientation="h",
            color=mean_shap.values,
            color_continuous_scale=[[0, "#4F8BF9"], [1, "#7C4DFF"]],
            labels={"x": "Mean |SHAP|", "y": "Feature"},
        )
        fig_ms.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(240,244,255,0.5)",
            font=dict(family="Inter, sans-serif"),
            height=320,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_ms, use_container_width=True, key="xai_mean_shap")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_fi:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Global Feature Importance")

        top_n = st.slider("Top N features", 5, 16, 16, key="xai_fi_n")
        fig_fi = feature_importance_chart(FEATURE_IMPORTANCE_DATA, top_n=top_n)
        st.plotly_chart(fig_fi, use_container_width=True, key="xai_fi")

        st.markdown("<hr>", unsafe_allow_html=True)
        section_header("Feature Importance Table")
        fi_df = pd.DataFrame(
            list(FEATURE_IMPORTANCE_DATA.items()),
            columns=["Feature", "Importance Score"]
        ).sort_values("Importance Score", ascending=False).reset_index(drop=True)
        fi_df["Rank"] = fi_df.index + 1
        fi_df["Importance Score"] = fi_df["Importance Score"].apply(lambda x: f"{x:.3f}")
        st.dataframe(fi_df[["Rank", "Feature", "Importance Score"]], use_container_width=True,
                     hide_index=True, height=360)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_attn:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Transformer Attention Weights (Self-Attention)")

        seq_len = st.slider("Sequence Length (timesteps)", 5, 20, 15, key="xai_attn_len")
        attn = get_attention_weights(seq_len)
        fig_attn = attention_heatmap(attn)
        st.plotly_chart(fig_attn, use_container_width=True, key="xai_attn")

        st.info(
            "Attention weights show how much the model attends to each past timestep when "
            "making predictions. Bright cells indicate higher attention. Recent timesteps "
            "typically receive more attention in network attack detection."
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: BENCHMARK
# ══════════════════════════════════════════════════════════════

def page_benchmark():
    page_header("📈", "Benchmark", "Compare World Model against logistic regression and other baselines")

    bench_df = get_benchmark_dataframe()

    # ── Top model comparison metrics ──
    st.markdown('<div class="ng-card">', unsafe_allow_html=True)
    section_header("Model Performance Overview")
    st.dataframe(
        bench_df.style.highlight_max(
            subset=["F1 Score", "Precision", "Recall", "Accuracy", "AUC-ROC"],
            color="rgba(79,139,249,0.2)"
        ).highlight_min(
            subset=["FPR"],
            color="rgba(79,139,249,0.2)"
        ).format({
            "F1 Score": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "Accuracy": "{:.4f}", "FPR": "{:.4f}", "AUC-ROC": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
        height=260,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar charts ──
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        metric_sel1 = st.selectbox("Metric (left chart)", ["F1 Score", "Precision", "Recall", "Accuracy", "AUC-ROC"], key="bm_m1")
        fig_bc1 = benchmark_bar_chart(bench_df, metric_sel1)
        st.plotly_chart(fig_bc1, use_container_width=True, key="bm_bar1")
        st.markdown('</div>', unsafe_allow_html=True)

    with bc2:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        metric_sel2 = st.selectbox("Metric (right chart)", ["AUC-ROC", "FPR", "Accuracy", "F1 Score"], key="bm_m2")
        fig_bc2 = benchmark_bar_chart(bench_df, metric_sel2)
        st.plotly_chart(fig_bc2, use_container_width=True, key="bm_bar2")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROC Curves ──
    col_roc, col_cm = st.columns([3, 2])
    with col_roc:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("ROC Curves")
        roc_data = get_roc_curve_data()
        fig_roc = roc_curve_chart(roc_data)
        st.plotly_chart(fig_roc, use_container_width=True, key="bm_roc")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_cm:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Confusion Matrix")
        cm_model = st.selectbox("Model", ["World Model (LSTM)", "World Model (Transformer)", "Logistic Regression"], key="bm_cm_sel")
        cm = get_confusion_matrix(cm_model)
        fig_cm = confusion_matrix_chart(cm, cm_model)
        st.plotly_chart(fig_cm, use_container_width=True, key="bm_cm")
        perf = get_model_performance_metrics(cm_model)
        cc1, cc2 = st.columns(2)
        cc1.metric("Precision", f"{perf['precision']:.4f}")
        cc2.metric("Recall", f"{perf['recall']:.4f}")
        cc1.metric("F1 Score", f"{perf['f1_score']:.4f}")
        cc2.metric("FPR", f"{perf['fpr']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Radar benchmark ──
    st.markdown('<div class="ng-card">', unsafe_allow_html=True)
    section_header("Multi-Model Performance Radar")
    selected_models = st.multiselect(
        "Select models to compare",
        BENCHMARK_MODELS,
        default=["World Model (LSTM)", "World Model (Transformer)", "Logistic Regression", "XGBoost"],
        key="bm_radar_sel"
    )
    if len(selected_models) >= 2:
        fig_radar = benchmark_radar_chart(BENCHMARK_METRICS, selected_models)
        st.plotly_chart(fig_radar, use_container_width=True, key="bm_radar")
    else:
        st.info("Select at least 2 models for the radar chart.")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════

def page_settings():
    page_header("⚙️", "Settings", "Model configuration, detection thresholds, and system settings")

    col_model, col_thresh = st.columns([1, 1])

    with col_model:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Model Configuration")

        st.selectbox("Active Model", list(MODEL_ARCHITECTURES.keys()), key="cfg_model")
        st.number_input("Sequence Window Length", 5, 100, 20, key="cfg_seq")
        st.number_input("K-step Forecast Horizon", 1, 30, 10, key="cfg_k")
        st.number_input("LSTM Hidden Units", 64, 512, 256, step=64, key="cfg_hidden")
        st.number_input("LSTM Layers", 1, 6, 2, key="cfg_layers")
        st.select_slider("Dropout Rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], value=0.3, key="cfg_drop")
        st.selectbox("Optimizer", ["Adam", "AdamW", "SGD", "RMSprop"], key="cfg_opt")
        st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3,
                         format_func=lambda x: f"{x:.0e}", key="cfg_lr")

        if st.button("💾 Save Model Config", use_container_width=True, key="btn_save_model"):
            st.success("Model configuration saved.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_thresh:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Detection Thresholds")

        alert_thresh = st.slider("Alert Threshold", 0.3, 0.95, 0.70, step=0.05, key="cfg_alert")
        critical_thresh = st.slider("Critical Alert Threshold", 0.5, 0.99, 0.85, step=0.05, key="cfg_crit")
        anomaly_thresh = st.slider("Anomaly Score Threshold", 0.2, 0.95, 0.60, step=0.05, key="cfg_anom")
        min_confidence = st.slider("Minimum Prediction Confidence", 0.5, 0.99, 0.65, step=0.05, key="cfg_conf")

        st.markdown("<hr>", unsafe_allow_html=True)
        section_header("Alert Settings")
        st.checkbox("Enable Email Alerts", value=False, key="cfg_email")
        st.checkbox("Enable Webhook Notifications", value=False, key="cfg_webhook")
        st.checkbox("Auto-block suspicious IPs", value=False, key="cfg_block")
        st.selectbox("Alert Aggregation Window", ["1 min", "5 min", "15 min", "1 hour"], index=1, key="cfg_agg")

        if st.button("💾 Save Threshold Config", use_container_width=True, key="btn_save_thresh"):
            st.success("Threshold configuration saved.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Data settings ──
    col_data, col_sys = st.columns([1, 1])

    with col_data:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("Data Pipeline Settings")
        st.selectbox("Input Data Source", ["File Upload", "Live PCAP Capture", "Network Tap", "SIEM Integration"], key="cfg_src")
        st.number_input("Max Flow Buffer Size", 1000, 100000, 10000, step=1000, key="cfg_buf")
        st.selectbox("Feature Normalization", ["MinMax Scaler", "Standard Scaler", "Robust Scaler", "None"], key="cfg_norm")
        st.checkbox("Apply PCA Dimensionality Reduction", value=False, key="cfg_pca")
        st.number_input("Train/Val/Test Split (%)", 60, 80, 70, key="cfg_split")
        if st.button("💾 Save Data Config", use_container_width=True, key="btn_save_data"):
            st.success("Data configuration saved.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sys:
        st.markdown('<div class="ng-card">', unsafe_allow_html=True)
        section_header("System Information")

        model_s = get_model_status()
        info_items = [
            ("App Version", "1.0.0"),
            ("Model Name", model_s["model_name"]),
            ("Model Version", model_s["version"]),
            ("Dataset", model_s["dataset"]),
            ("Total Parameters", model_s["total_params"]),
            ("Last Trained", model_s["last_trained"]),
            ("Epochs Trained", str(model_s["epochs_trained"])),
            ("Device", model_s["device"]),
        ]
        for k, v in info_items:
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; padding:7px 0; "
                f"border-bottom:1px solid rgba(79,139,249,0.08);'>"
                f"<span style='font-size:0.84rem; color:#5a6a8a;'>{k}</span>"
                f"<span style='font-size:0.84rem; font-weight:600; color:#1a1a2e;'>{v}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    inject_css()
    inject_animated_background()

    if "ingested_df" not in st.session_state:
        st.session_state["ingested_df"] = generate_mock_traffic_data(300)
    if "training_done" not in st.session_state:
        st.session_state["training_done"] = False

    page = render_sidebar()

    # Wrap all content in a z-index container so it sits above the canvas
    st.markdown(
        '<div style="position:relative; z-index:1;">',
        unsafe_allow_html=True
    )

    if page == "Dashboard":
        page_dashboard()
    elif page == "Data Ingestion":
        page_data_ingestion()
    elif page == "Model Training":
        page_model_training()
    elif page == "Attack Prediction":
        page_attack_prediction()
    elif page == "MITRE ATT&CK":
        page_mitre_attack()
    elif page == "Explainability":
        page_explainability()
    elif page == "Benchmark":
        page_benchmark()
    elif page == "Settings":
        page_settings()

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
