"""
Data processing utilities for Network Attack Forecasting.
Handles CIC-IDS2018 feature set, CSV/PCAP ingestion, and feature extraction.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import io

# CIC-IDS2018 feature names
CIC_IDS_FEATURES = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
    "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts",
    "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"
]

TOP_FEATURES = [
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "Flow Byts/s",
    "Flow Pkts/s", "Pkt Len Mean", "Fwd Pkt Len Mean", "Bwd Pkt Len Mean",
    "SYN Flag Cnt", "ACK Flag Cnt", "FIN Flag Cnt", "RST Flag Cnt",
    "Flow IAT Mean", "Fwd IAT Mean", "Bwd IAT Mean", "Init Fwd Win Byts"
]

ATTACK_LABELS = [
    "Benign", "DDoS attacks-LOIC-HTTP", "DDoS-LOIC-UDP",
    "DoS attacks-Hulk", "DoS attacks-SlowHTTPTest", "DoS attacks-GoldenEye",
    "FTP-BruteForce", "SSH-Bruteforce", "Infilteration", "Bot",
    "SQL Injection", "Brute Force -XSS", "Brute Force -Web"
]


def generate_mock_traffic_data(n_rows: int = 500) -> pd.DataFrame:
    """Generate synthetic CIC-IDS2018-style network traffic data."""
    np.random.seed(42)
    rng = np.random.default_rng(42)

    # Time index
    base_time = datetime(2024, 1, 15, 8, 0, 0)
    timestamps = [base_time + timedelta(seconds=i * 2) for i in range(n_rows)]

    data = {
        "Timestamp": timestamps,
        "Src IP": [f"192.168.{rng.integers(1,5)}.{rng.integers(1,254)}" for _ in range(n_rows)],
        "Dst IP": [f"10.0.{rng.integers(0,3)}.{rng.integers(1,254)}" for _ in range(n_rows)],
        "Dst Port": rng.choice([80, 443, 22, 21, 8080, 3306, 8443, 445], n_rows),
        "Protocol": rng.choice([6, 17, 1], n_rows, p=[0.7, 0.2, 0.1]),
        "Flow Duration": rng.exponential(50000, n_rows).astype(int),
        "Tot Fwd Pkts": rng.integers(1, 200, n_rows),
        "Tot Bwd Pkts": rng.integers(0, 150, n_rows),
        "TotLen Fwd Pkts": rng.integers(40, 150000, n_rows),
        "TotLen Bwd Pkts": rng.integers(0, 100000, n_rows),
        "Fwd Pkt Len Max": rng.integers(40, 1500, n_rows),
        "Fwd Pkt Len Min": rng.integers(20, 100, n_rows),
        "Fwd Pkt Len Mean": rng.uniform(40, 800, n_rows).round(2),
        "Fwd Pkt Len Std": rng.uniform(0, 400, n_rows).round(2),
        "Bwd Pkt Len Max": rng.integers(0, 1500, n_rows),
        "Bwd Pkt Len Min": rng.integers(0, 100, n_rows),
        "Bwd Pkt Len Mean": rng.uniform(0, 800, n_rows).round(2),
        "Bwd Pkt Len Std": rng.uniform(0, 400, n_rows).round(2),
        "Flow Byts/s": rng.exponential(50000, n_rows).round(2),
        "Flow Pkts/s": rng.exponential(100, n_rows).round(2),
        "Flow IAT Mean": rng.exponential(10000, n_rows).round(2),
        "Flow IAT Std": rng.exponential(5000, n_rows).round(2),
        "Flow IAT Max": rng.integers(1000, 500000, n_rows),
        "Flow IAT Min": rng.integers(0, 1000, n_rows),
        "Fwd IAT Tot": rng.integers(0, 1000000, n_rows),
        "Fwd IAT Mean": rng.exponential(10000, n_rows).round(2),
        "Fwd IAT Std": rng.exponential(5000, n_rows).round(2),
        "Fwd IAT Max": rng.integers(0, 500000, n_rows),
        "Fwd IAT Min": rng.integers(0, 1000, n_rows),
        "Bwd IAT Tot": rng.integers(0, 1000000, n_rows),
        "Bwd IAT Mean": rng.exponential(10000, n_rows).round(2),
        "Bwd IAT Std": rng.exponential(5000, n_rows).round(2),
        "Bwd IAT Max": rng.integers(0, 500000, n_rows),
        "Bwd IAT Min": rng.integers(0, 1000, n_rows),
        "Fwd PSH Flags": rng.integers(0, 2, n_rows),
        "Bwd PSH Flags": rng.integers(0, 2, n_rows),
        "Fwd URG Flags": rng.integers(0, 2, n_rows),
        "Bwd URG Flags": rng.integers(0, 2, n_rows),
        "Fwd Header Len": rng.integers(20, 60, n_rows),
        "Bwd Header Len": rng.integers(0, 60, n_rows),
        "Fwd Pkts/s": rng.exponential(50, n_rows).round(2),
        "Bwd Pkts/s": rng.exponential(30, n_rows).round(2),
        "Pkt Len Min": rng.integers(20, 100, n_rows),
        "Pkt Len Max": rng.integers(100, 1500, n_rows),
        "Pkt Len Mean": rng.uniform(40, 800, n_rows).round(2),
        "Pkt Len Std": rng.uniform(0, 400, n_rows).round(2),
        "Pkt Len Var": rng.uniform(0, 160000, n_rows).round(2),
        "FIN Flag Cnt": rng.integers(0, 2, n_rows),
        "SYN Flag Cnt": rng.integers(0, 3, n_rows),
        "RST Flag Cnt": rng.integers(0, 2, n_rows),
        "PSH Flag Cnt": rng.integers(0, 5, n_rows),
        "ACK Flag Cnt": rng.integers(0, 10, n_rows),
        "URG Flag Cnt": rng.integers(0, 2, n_rows),
        "CWE Flag Count": rng.integers(0, 2, n_rows),
        "ECE Flag Cnt": rng.integers(0, 2, n_rows),
        "Down/Up Ratio": rng.uniform(0, 5, n_rows).round(2),
        "Pkt Size Avg": rng.uniform(40, 800, n_rows).round(2),
        "Fwd Seg Size Avg": rng.uniform(40, 800, n_rows).round(2),
        "Bwd Seg Size Avg": rng.uniform(0, 800, n_rows).round(2),
        "Init Fwd Win Byts": rng.choice([65535, 8192, 16384, 32768, -1], n_rows),
        "Init Bwd Win Byts": rng.choice([65535, 8192, 16384, 32768, -1], n_rows),
        "Fwd Act Data Pkts": rng.integers(0, 50, n_rows),
        "Fwd Seg Size Min": rng.integers(8, 60, n_rows),
        "Active Mean": rng.exponential(5000, n_rows).round(2),
        "Active Std": rng.exponential(2000, n_rows).round(2),
        "Active Max": rng.integers(1000, 100000, n_rows),
        "Active Min": rng.integers(0, 1000, n_rows),
        "Idle Mean": rng.exponential(200000, n_rows).round(2),
        "Idle Std": rng.exponential(100000, n_rows).round(2),
        "Idle Max": rng.integers(10000, 1000000, n_rows),
        "Idle Min": rng.integers(0, 10000, n_rows),
    }

    # Assign labels (80% benign, 20% attacks)
    attack_dist = [0.80] + [0.20 / (len(ATTACK_LABELS) - 1)] * (len(ATTACK_LABELS) - 1)
    data["Label"] = rng.choice(ATTACK_LABELS, n_rows, p=attack_dist)

    df = pd.DataFrame(data)
    return df


def parse_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Parse an uploaded CSV file and align columns to CIC-IDS2018 features."""
    try:
        content = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(content))
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and clean top features from raw dataframe."""
    available = [f for f in TOP_FEATURES if f in df.columns]
    if len(available) < 5:
        # fall back to numeric columns
        available = df.select_dtypes(include=[np.number]).columns.tolist()[:16]

    feature_df = df[available].copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    feature_df = feature_df.clip(lower=0)
    return feature_df


def compute_flow_statistics(df: pd.DataFrame) -> dict:
    """Compute aggregate flow-level statistics for dashboard metrics."""
    stats = {}
    total = len(df)
    stats["total_flows"] = total

    if "Label" in df.columns:
        attack_mask = df["Label"] != "Benign"
        stats["attack_flows"] = int(attack_mask.sum())
        stats["benign_flows"] = total - stats["attack_flows"]
        stats["attack_rate"] = round(stats["attack_flows"] / total * 100, 2) if total > 0 else 0.0
        label_counts = df["Label"].value_counts().to_dict()
        stats["label_distribution"] = label_counts
    else:
        stats["attack_flows"] = 0
        stats["benign_flows"] = total
        stats["attack_rate"] = 0.0
        stats["label_distribution"] = {"Unknown": total}

    if "Flow Byts/s" in df.columns:
        stats["avg_flow_bytes_s"] = round(df["Flow Byts/s"].mean(), 2)
        stats["max_flow_bytes_s"] = round(df["Flow Byts/s"].max(), 2)
    else:
        stats["avg_flow_bytes_s"] = 0.0
        stats["max_flow_bytes_s"] = 0.0

    if "Flow Duration" in df.columns:
        stats["avg_duration_ms"] = round(df["Flow Duration"].mean() / 1000, 2)
    else:
        stats["avg_duration_ms"] = 0.0

    return stats


def generate_time_series_data(n_points: int = 60) -> pd.DataFrame:
    """Generate time-series traffic data for live dashboard metrics."""
    np.random.seed(int(datetime.now().second))
    base = datetime.now() - timedelta(minutes=n_points)
    timestamps = [base + timedelta(minutes=i) for i in range(n_points)]

    # Simulate a rising attack scenario in the last 15 minutes
    attack_probs = np.concatenate([
        np.random.uniform(0.02, 0.12, n_points - 20),
        np.linspace(0.12, 0.72, 10),
        np.random.uniform(0.65, 0.85, 10)
    ])

    flow_rates = np.concatenate([
        np.random.uniform(800, 1200, n_points - 15),
        np.random.uniform(2500, 6000, 15)
    ])

    anomaly_scores = attack_probs + np.random.normal(0, 0.03, n_points)
    anomaly_scores = np.clip(anomaly_scores, 0, 1)

    return pd.DataFrame({
        "Timestamp": timestamps,
        "Attack Probability": np.round(attack_probs, 4),
        "Flow Rate (flows/min)": np.round(flow_rates, 1),
        "Anomaly Score": np.round(anomaly_scores, 4),
    })


def generate_k_step_forecast(k: int = 10, base_prob: float = 0.45) -> pd.DataFrame:
    """Generate K-step ahead attack probability forecast."""
    np.random.seed(99)
    steps = list(range(1, k + 1))
    # Simulate escalating attack probability
    probs = base_prob + np.cumsum(np.random.uniform(0.01, 0.06, k))
    probs = np.clip(probs, 0.0, 0.98)

    upper = np.clip(probs + np.random.uniform(0.03, 0.08, k), 0, 1)
    lower = np.clip(probs - np.random.uniform(0.03, 0.08, k), 0, 1)

    stage_names = [
        "Reconnaissance", "Initial Access", "Execution",
        "Lateral Movement", "Command & Control", "Exfiltration"
    ]
    # Map steps to stages
    stage_map = [stage_names[min(int((p / 1.0) * len(stage_names)), len(stage_names) - 1)] for p in probs]

    return pd.DataFrame({
        "Step": steps,
        "Attack Probability": np.round(probs, 4),
        "Upper Bound": np.round(upper, 4),
        "Lower Bound": np.round(lower, 4),
        "Predicted Stage": stage_map,
    })


def get_recent_alerts(n: int = 8) -> list:
    """Generate mock recent alert entries."""
    rng = np.random.default_rng(42)
    alert_types = [
        "DDoS Attack Detected", "Brute Force Attempt", "Port Scan",
        "SQL Injection", "Data Exfiltration", "Lateral Movement",
        "C&C Communication", "Anomalous Traffic Spike"
    ]
    severities = ["Critical", "High", "Medium", "Low"]
    severity_weights = [0.15, 0.30, 0.35, 0.20]
    ips = [f"192.168.{rng.integers(1,5)}.{rng.integers(1,254)}" for _ in range(n)]

    alerts = []
    base_time = datetime.now()
    for i in range(n):
        sev = rng.choice(severities, p=severity_weights)
        alerts.append({
            "time": (base_time - timedelta(minutes=int(rng.integers(1, 30)))).strftime("%H:%M:%S"),
            "type": rng.choice(alert_types),
            "source_ip": ips[i],
            "severity": sev,
            "confidence": round(float(rng.uniform(0.65, 0.99)), 2),
        })
    return sorted(alerts, key=lambda x: x["time"], reverse=True)
