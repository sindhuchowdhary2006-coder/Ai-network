"""
All Plotly chart functions for Network Attack Forecasting dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────
#  Color Palette
# ─────────────────────────────────────────────────────────────

COLORS = {
    "primary": "#4F8BF9",
    "secondary": "#7C4DFF",
    "success": "#43A047",
    "warning": "#FF8F00",
    "danger": "#E53935",
    "info": "#00ACC1",
    "light_bg": "#F0F4FF",
    "card_bg": "#FFFFFF",
    "text_primary": "#1A1A2E",
    "text_secondary": "#5A6A8A",
    "gradient_start": "#4F8BF9",
    "gradient_end": "#7C4DFF",
    "stage_colors": {
        "Reconnaissance": "#4FC3F7",
        "Initial Access": "#81C784",
        "Execution": "#FFB74D",
        "Lateral Movement": "#FF8A65",
        "Command & Control": "#CE93D8",
        "Exfiltration": "#EF5350",
    },
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(240,244,255,0.6)",
    font=dict(family="Inter, sans-serif", color=COLORS["text_primary"]),
    margin=dict(l=40, r=40, t=50, b=40),
    hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter"),
)


def _apply_defaults(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(**LAYOUT_DEFAULTS)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=16, color=COLORS["text_primary"]), x=0.01))
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(79,139,249,0.12)", zeroline=False,
        linecolor="rgba(79,139,249,0.3)"
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(79,139,249,0.12)", zeroline=False,
        linecolor="rgba(79,139,249,0.3)"
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  Gauge Chart – Attack Probability
# ─────────────────────────────────────────────────────────────

def attack_probability_gauge(probability: float, title: str = "Attack Probability") -> go.Figure:
    pct = round(probability * 100, 1)
    if pct < 25:
        color = COLORS["success"]
        level = "LOW"
    elif pct < 50:
        color = "#CDDC39"
        level = "MODERATE"
    elif pct < 75:
        color = COLORS["warning"]
        level = "HIGH"
    else:
        color = COLORS["danger"]
        level = "CRITICAL"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        delta={"reference": 30, "increasing": {"color": COLORS["danger"]}, "decreasing": {"color": COLORS["success"]}},
        title={"text": f"{title}<br><span style='font-size:0.75em;color:{color}'>{level}</span>",
               "font": {"size": 16}},
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["text_secondary"],
                     "tickfont": {"size": 11}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "rgba(79,139,249,0.2)",
            "steps": [
                {"range": [0, 25], "color": "rgba(67,160,71,0.12)"},
                {"range": [25, 50], "color": "rgba(205,220,57,0.12)"},
                {"range": [50, 75], "color": "rgba(255,143,0,0.12)"},
                {"range": [75, 100], "color": "rgba(229,57,53,0.12)"},
            ],
            "threshold": {
                "line": {"color": COLORS["danger"], "width": 3},
                "thickness": 0.75,
                "value": 70,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  Timeline – Live Attack Probability
# ─────────────────────────────────────────────────────────────

def live_timeline_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Timestamp"], y=df["Attack Probability"],
        mode="lines",
        name="Attack Probability",
        line=dict(color=COLORS["danger"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(229,57,53,0.08)",
    ))

    fig.add_trace(go.Scatter(
        x=df["Timestamp"], y=df["Anomaly Score"],
        mode="lines",
        name="Anomaly Score",
        line=dict(color=COLORS["secondary"], width=1.8, dash="dot"),
    ))

    fig.add_hline(y=0.7, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text="Alert Threshold (0.70)",
                  annotation_position="top right",
                  annotation_font_color=COLORS["danger"])

    _apply_defaults(fig, "Live Attack Probability Timeline")
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320,
        yaxis=dict(range=[0, 1], title="Probability"),
        xaxis=dict(title="Time"),
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  Flow Rate Chart
# ─────────────────────────────────────────────────────────────

def flow_rate_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Timestamp"], y=df["Flow Rate (flows/min)"],
        name="Flow Rate",
        marker_color=COLORS["primary"],
        marker_line_width=0,
        opacity=0.85,
    ))
    _apply_defaults(fig, "Network Flow Rate (flows/min)")
    fig.update_layout(height=300, yaxis_title="Flows/min", xaxis_title="Time")
    return fig


# ─────────────────────────────────────────────────────────────
#  K-Step Forecast Chart
# ─────────────────────────────────────────────────────────────

def k_step_forecast_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(df["Step"]) + list(df["Step"])[::-1],
        y=list(df["Upper Bound"]) + list(df["Lower Bound"])[::-1],
        fill="toself",
        fillcolor="rgba(79,139,249,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Confidence Band",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=df["Step"], y=df["Attack Probability"],
        mode="lines+markers",
        name="Predicted Attack Prob.",
        line=dict(color=COLORS["primary"], width=2.8),
        marker=dict(size=8, color=COLORS["secondary"], line=dict(color="white", width=2)),
    ))

    # Color-coded stage annotations
    stage_colors = COLORS["stage_colors"]
    prev_stage = None
    for _, row in df.iterrows():
        stage = row["Predicted Stage"]
        if stage != prev_stage:
            fig.add_annotation(
                x=row["Step"], y=row["Attack Probability"] + 0.06,
                text=stage,
                showarrow=False,
                font=dict(size=9, color=stage_colors.get(stage, "#555")),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=stage_colors.get(stage, "#555"),
                borderwidth=1,
                borderpad=3,
            )
            prev_stage = stage

    fig.add_hline(y=0.7, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text="Alert Threshold",
                  annotation_position="top left",
                  annotation_font_color=COLORS["danger"])

    _apply_defaults(fig, "K-Step Forward Attack Probability Forecast")
    fig.update_layout(
        height=380,
        xaxis=dict(title="Forecast Step (t+k)", tickmode="linear"),
        yaxis=dict(title="Attack Probability", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  Training Loss Curves
# ─────────────────────────────────────────────────────────────

def training_curves_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Loss Curves", "Accuracy / F1 Curves"))

    fig.add_trace(go.Scatter(x=df["Epoch"], y=df["Train Loss"],
                              mode="lines", name="Train Loss",
                              line=dict(color=COLORS["primary"], width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Epoch"], y=df["Val Loss"],
                              mode="lines", name="Val Loss",
                              line=dict(color=COLORS["danger"], width=2.2, dash="dot")), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["Epoch"], y=df["Train Accuracy"],
                              mode="lines", name="Train Acc",
                              line=dict(color=COLORS["success"], width=2.2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["Epoch"], y=df["Val Accuracy"],
                              mode="lines", name="Val Acc",
                              line=dict(color=COLORS["info"], width=2.2, dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["Epoch"], y=df["Train F1"],
                              mode="lines", name="Train F1",
                              line=dict(color=COLORS["warning"], width=2.0, dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["Epoch"], y=df["Val F1"],
                              mode="lines", name="Val F1",
                              line=dict(color=COLORS["secondary"], width=2.0, dash="longdash")), row=1, col=2)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(79,139,249,0.12)", title_text="Epoch")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(79,139,249,0.12)")
    return fig


# ─────────────────────────────────────────────────────────────
#  Feature Importance Bar Chart
# ─────────────────────────────────────────────────────────────

def feature_importance_chart(importance_dict: dict, top_n: int = 16) -> go.Figure:
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    # Color gradient based on importance
    max_val = max(values) if values else 1
    bar_colors = [
        f"rgba({int(79 + (229-79) * v/max_val)}, {int(139 - (139-57) * v/max_val)}, {int(249 - (249-53) * v/max_val)}, 0.85)"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=features[::-1],
        orientation="h",
        marker_color=bar_colors[::-1],
        marker_line_width=0,
        text=[f"{v:.3f}" for v in values[::-1]],
        textposition="outside",
        textfont=dict(size=11),
    ))

    _apply_defaults(fig, "Feature Importance (SHAP-based)")
    fig.update_layout(
        height=420,
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  SHAP Summary Plot
# ─────────────────────────────────────────────────────────────

def shap_beeswarm_chart(shap_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    feature_means = shap_df.abs().mean().sort_values(ascending=False)
    top_features = feature_means.head(12).index.tolist()

    for i, feat in enumerate(top_features):
        vals = shap_df[feat].values
        y_jitter = np.random.normal(i, 0.15, len(vals))
        colors = [COLORS["danger"] if v > 0 else COLORS["primary"] for v in vals]

        fig.add_trace(go.Scatter(
            x=vals,
            y=y_jitter,
            mode="markers",
            name=feat,
            marker=dict(color=colors, size=6, opacity=0.7,
                        line=dict(color="white", width=0.5)),
            showlegend=False,
        ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=420,
        title="SHAP Value Distribution (Beeswarm)",
        xaxis_title="SHAP Value (impact on model output)",
        yaxis=dict(
            tickvals=list(range(len(top_features))),
            ticktext=top_features,
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  Attention Weights Heatmap
# ─────────────────────────────────────────────────────────────

def attention_heatmap(attention_matrix: np.ndarray) -> go.Figure:
    seq_len = attention_matrix.shape[0]
    labels = [f"t-{seq_len-1-i}" if i < seq_len-1 else "t" for i in range(seq_len)]

    fig = go.Figure(go.Heatmap(
        z=attention_matrix,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "rgba(240,244,255,1)"],
            [0.3, "rgba(79,139,249,0.4)"],
            [0.7, "rgba(124,77,255,0.7)"],
            [1.0, "rgba(79,139,249,1)"],
        ],
        showscale=True,
        colorbar=dict(title="Attn Weight", thickness=12),
        hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>",
    ))

    _apply_defaults(fig, "Transformer Attention Weights (Last Layer)")
    fig.update_layout(height=400, xaxis_title="Key (Time Step)", yaxis_title="Query (Time Step)")
    return fig


# ─────────────────────────────────────────────────────────────
#  Confusion Matrix Heatmap
# ─────────────────────────────────────────────────────────────

def confusion_matrix_chart(cm: np.ndarray, model_name: str) -> go.Figure:
    labels = ["Benign", "Attack"]
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                text=f"<b>{cm[i][j]:,}</b><br>({cm_normalized[i][j]:.1%})",
                x=labels[j], y=labels[i],
                xref="x", yref="y",
                showarrow=False,
                font=dict(size=14, color="white" if cm_normalized[i][j] > 0.4 else COLORS["text_primary"]),
            ))

    fig = go.Figure(go.Heatmap(
        z=cm_normalized,
        x=labels, y=labels,
        colorscale=[
            [0.0, "rgba(240,244,255,1)"],
            [0.5, "rgba(79,139,249,0.5)"],
            [1.0, "rgba(79,139,249,1)"],
        ],
        showscale=True,
        colorbar=dict(title="Rate", thickness=12),
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"Confusion Matrix — {model_name}",
        annotations=annotations,
        height=340,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  Benchmark Comparison Bar Chart
# ─────────────────────────────────────────────────────────────

def benchmark_bar_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    sorted_df = df.sort_values(metric, ascending=True)
    colors = []
    for m in sorted_df["Model"]:
        if "World Model" in m:
            colors.append(COLORS["primary"])
        elif m in ["Random Forest", "XGBoost"]:
            colors.append(COLORS["info"])
        else:
            colors.append("rgba(150,150,180,0.7)")

    fig = go.Figure(go.Bar(
        y=sorted_df["Model"],
        x=sorted_df[metric],
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.4f}" for v in sorted_df[metric]],
        textposition="outside",
        textfont=dict(size=12),
    ))

    _apply_defaults(fig, f"Benchmark Comparison — {metric}")
    fig.update_layout(
        height=340,
        xaxis=dict(title=metric, range=[0, 1.05]),
        yaxis_title="",
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  ROC Curve
# ─────────────────────────────────────────────────────────────

def roc_curve_chart(curves: dict) -> go.Figure:
    fig = go.Figure()

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", name="Random Classifier",
        line=dict(color="rgba(150,150,150,0.5)", dash="dash", width=1.5),
    ))

    palette = [COLORS["primary"], COLORS["secondary"], COLORS["danger"],
               COLORS["success"], COLORS["warning"], COLORS["info"]]

    for i, (model, data) in enumerate(curves.items()):
        is_world = "World Model" in model
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"],
            mode="lines",
            name=f"{model} (AUC={data['auc']:.4f})",
            line=dict(
                color=palette[i % len(palette)],
                width=3.0 if is_world else 1.8,
                dash="solid" if is_world else "dot",
            ),
        ))

    _apply_defaults(fig, "ROC Curves — All Models")
    fig.update_layout(
        height=420,
        xaxis=dict(title="False Positive Rate", range=[0, 1]),
        yaxis=dict(title="True Positive Rate", range=[0, 1.02]),
        legend=dict(x=0.55, y=0.12, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(79,139,249,0.3)", borderwidth=1),
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  Attack Label Distribution Pie
# ─────────────────────────────────────────────────────────────

def label_distribution_pie(label_counts: dict) -> go.Figure:
    labels = list(label_counts.keys())
    values = list(label_counts.values())

    colors_pie = [
        "#4F8BF9", "#EF5350", "#FFB74D", "#81C784", "#CE93D8",
        "#FF8A65", "#4FC3F7", "#F06292", "#AED581", "#FFF176",
        "#80DEEA", "#BCAAA4", "#B0BEC5"
    ]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.45,
        marker=dict(colors=colors_pie[:len(labels)], line=dict(color="white", width=2)),
        textinfo="percent+label",
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Traffic Label Distribution",
        height=360,
        legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=11)),
        showlegend=True,
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  MITRE Stage Probability Radar
# ─────────────────────────────────────────────────────────────

def mitre_stage_radar(stage_probs: dict) -> go.Figure:
    categories = list(stage_probs.keys())
    values = list(stage_probs.values())
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(79,139,249,0.18)",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=7, color=COLORS["secondary"]),
        name="Attack Stage Probability",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(240,244,255,0.6)",
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10),
                            gridcolor="rgba(79,139,249,0.2)"),
            angularaxis=dict(tickfont=dict(size=12), gridcolor="rgba(79,139,249,0.2)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        title="MITRE ATT&CK Stage Probability",
        font=dict(family="Inter, sans-serif"),
        height=380,
        margin=dict(l=50, r=50, t=60, b=30),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  3D Network Traffic Scatter
# ─────────────────────────────────────────────────────────────

def network_traffic_3d(df: pd.DataFrame, sample_n: int = 300) -> go.Figure:
    if len(df) > sample_n:
        plot_df = df.sample(sample_n, random_state=42)
    else:
        plot_df = df.copy()

    col_x = "Flow Duration" if "Flow Duration" in df.columns else df.columns[0]
    col_y = "Flow Byts/s" if "Flow Byts/s" in df.columns else df.columns[1]
    col_z = "Tot Fwd Pkts" if "Tot Fwd Pkts" in df.columns else df.columns[2]
    col_c = "Label" if "Label" in df.columns else None

    if col_c:
        is_attack = plot_df[col_c] != "Benign"
        colors = [COLORS["danger"] if a else COLORS["primary"] for a in is_attack]
        text_labels = plot_df[col_c].tolist()
    else:
        colors = COLORS["primary"]
        text_labels = ["Unknown"] * len(plot_df)

    fig = go.Figure(go.Scatter3d(
        x=plot_df[col_x].clip(0, plot_df[col_x].quantile(0.99)),
        y=plot_df[col_y].clip(0, plot_df[col_y].quantile(0.99)),
        z=plot_df[col_z].clip(0, plot_df[col_z].quantile(0.99)),
        mode="markers",
        marker=dict(size=4, color=colors, opacity=0.75,
                    line=dict(color="white", width=0.3)),
        text=text_labels,
        hovertemplate=f"<b>%{{text}}</b><br>{col_x}: %{{x:.1f}}<br>{col_y}: %{{y:.1f}}<br>{col_z}: %{{z:.1f}}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(title=col_x[:20], backgroundcolor="rgba(240,244,255,0.6)",
                       gridcolor="rgba(79,139,249,0.2)"),
            yaxis=dict(title=col_y[:20], backgroundcolor="rgba(240,244,255,0.6)",
                       gridcolor="rgba(79,139,249,0.2)"),
            zaxis=dict(title=col_z[:20], backgroundcolor="rgba(240,244,255,0.6)",
                       gridcolor="rgba(79,139,249,0.2)"),
        ),
        title="3D Network Traffic Scatter",
        height=480,
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  Benchmark Metrics Radar
# ─────────────────────────────────────────────────────────────

def benchmark_radar_chart(metrics_dict: dict, model_names: list) -> go.Figure:
    categories = ["F1 Score", "Precision", "Recall", "Accuracy", "AUC-ROC"]

    fig = go.Figure()
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["danger"],
               COLORS["success"], COLORS["warning"], COLORS["info"]]

    for i, model in enumerate(model_names):
        m = metrics_dict[model]
        vals = [m["F1 Score"], m["Precision"], m["Recall"], m["Accuracy"], m["AUC-ROC"]]
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself",
            fillcolor=f"rgba{tuple(list(bytes.fromhex(palette[i % len(palette)].lstrip('#'))) + [40])}",
            line=dict(color=palette[i % len(palette)],
                      width=2.5 if "World Model" in model else 1.8),
            name=model,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(240,244,255,0.6)",
            radialaxis=dict(visible=True, range=[0.8, 1.0], tickfont=dict(size=10),
                            gridcolor="rgba(79,139,249,0.2)"),
            angularaxis=dict(tickfont=dict(size=12), gridcolor="rgba(79,139,249,0.2)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        title="Model Performance Radar",
        font=dict(family="Inter, sans-serif"),
        height=420,
        legend=dict(x=1.1, y=0.5),
        showlegend=True,
    )
    return fig
