"""
MITRE ATT&CK mapping utilities for network attack stage classification.
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
#  MITRE ATT&CK Stage Definitions
# ─────────────────────────────────────────────────────────────

MITRE_STAGES = [
    {
        "id": "TA0043",
        "name": "Reconnaissance",
        "icon": "🔍",
        "color": "#4FC3F7",
        "description": "Gathering information to plan future adversary operations.",
        "techniques": [
            {"id": "T1595", "name": "Active Scanning"},
            {"id": "T1592", "name": "Gather Victim Host Info"},
            {"id": "T1589", "name": "Gather Victim Identity Info"},
            {"id": "T1590", "name": "Gather Victim Network Info"},
        ],
        "network_indicators": [
            "Port scan activity", "ICMP sweeps", "DNS enumeration",
            "High rate of SYN packets", "Banner grabbing",
        ],
        "detection_features": ["SYN Flag Cnt", "Dst Port", "Tot Fwd Pkts", "Flow Duration"],
    },
    {
        "id": "TA0001",
        "name": "Initial Access",
        "icon": "🚪",
        "color": "#81C784",
        "description": "Gaining initial foothold in the target network.",
        "techniques": [
            {"id": "T1190", "name": "Exploit Public-Facing Application"},
            {"id": "T1133", "name": "External Remote Services"},
            {"id": "T1566", "name": "Phishing"},
            {"id": "T1078", "name": "Valid Accounts"},
        ],
        "network_indicators": [
            "Brute-force login attempts", "Exploit traffic patterns",
            "Unusual authentication flows", "HTTP error rate spikes",
        ],
        "detection_features": ["Dst Port", "FIN Flag Cnt", "RST Flag Cnt", "Flow Byts/s"],
    },
    {
        "id": "TA0002",
        "name": "Execution",
        "icon": "⚙️",
        "color": "#FFB74D",
        "description": "Running malicious code on victim systems.",
        "techniques": [
            {"id": "T1059", "name": "Command and Scripting Interpreter"},
            {"id": "T1203", "name": "Exploitation for Client Execution"},
            {"id": "T1072", "name": "Software Deployment Tools"},
            {"id": "T1569", "name": "System Services"},
        ],
        "network_indicators": [
            "Abnormal process network connections", "Unusual port usage",
            "Increased outbound connections", "Script download traffic",
        ],
        "detection_features": ["Flow Duration", "PSH Flag Cnt", "Fwd Pkt Len Mean", "ACK Flag Cnt"],
    },
    {
        "id": "TA0008",
        "name": "Lateral Movement",
        "icon": "↔️",
        "color": "#FF8A65",
        "description": "Moving through the environment to reach the target.",
        "techniques": [
            {"id": "T1021", "name": "Remote Services"},
            {"id": "T1534", "name": "Internal Spearphishing"},
            {"id": "T1570", "name": "Lateral Tool Transfer"},
            {"id": "T1563", "name": "Remote Service Session Hijacking"},
        ],
        "network_indicators": [
            "Internal east-west traffic spikes", "SMB/RDP connections",
            "Pass-the-hash patterns", "Unusual admin share access",
        ],
        "detection_features": ["Dst Port", "Init Fwd Win Byts", "Flow Pkts/s", "Tot Bwd Pkts"],
    },
    {
        "id": "TA0011",
        "name": "Command & Control",
        "icon": "📡",
        "color": "#CE93D8",
        "description": "Communicating with compromised systems to control them.",
        "techniques": [
            {"id": "T1071", "name": "Application Layer Protocol"},
            {"id": "T1573", "name": "Encrypted Channel"},
            {"id": "T1090", "name": "Proxy"},
            {"id": "T1095", "name": "Non-Application Layer Protocol"},
        ],
        "network_indicators": [
            "Beaconing patterns", "Unusual DNS queries", "Encrypted C&C traffic",
            "Periodic outbound connections", "Domain generation algorithm (DGA)",
        ],
        "detection_features": ["Flow IAT Mean", "Bwd IAT Mean", "Idle Mean", "Fwd IAT Std"],
    },
    {
        "id": "TA0010",
        "name": "Exfiltration",
        "icon": "📤",
        "color": "#EF5350",
        "description": "Stealing data from the target network.",
        "techniques": [
            {"id": "T1041", "name": "Exfiltration Over C2 Channel"},
            {"id": "T1048", "name": "Exfiltration Over Alternative Protocol"},
            {"id": "T1567", "name": "Exfiltration Over Web Service"},
            {"id": "T1030", "name": "Data Transfer Size Limits"},
        ],
        "network_indicators": [
            "Large outbound data transfers", "DNS tunneling", "HTTPS POST spikes",
            "Off-hours transfer activity", "Compressed archive transfers",
        ],
        "detection_features": ["TotLen Fwd Pkts", "Flow Byts/s", "Down/Up Ratio", "Bwd Pkt Len Max"],
    },
]

# Map attack labels to MITRE stages
LABEL_TO_STAGE = {
    "DDoS attacks-LOIC-HTTP": "Execution",
    "DDoS-LOIC-UDP": "Execution",
    "DoS attacks-Hulk": "Execution",
    "DoS attacks-SlowHTTPTest": "Initial Access",
    "DoS attacks-GoldenEye": "Execution",
    "FTP-BruteForce": "Initial Access",
    "SSH-Bruteforce": "Initial Access",
    "Infilteration": "Lateral Movement",
    "Bot": "Command & Control",
    "SQL Injection": "Execution",
    "Brute Force -XSS": "Initial Access",
    "Brute Force -Web": "Initial Access",
    "Benign": None,
}


def get_stage_by_name(name: str) -> dict:
    """Return stage metadata by name."""
    for stage in MITRE_STAGES:
        if stage["name"] == name:
            return stage
    return MITRE_STAGES[0]


def map_label_to_stage(label: str) -> str:
    """Map a CIC-IDS attack label to a MITRE stage name."""
    return LABEL_TO_STAGE.get(label, "Reconnaissance")


def get_kill_chain_progression(current_stage: str) -> list:
    """Return kill chain stages with active/inactive flags."""
    stage_names = [s["name"] for s in MITRE_STAGES]
    current_idx = next(
        (i for i, s in enumerate(MITRE_STAGES) if s["name"] == current_stage),
        0
    )
    progression = []
    for i, stage in enumerate(MITRE_STAGES):
        progression.append({
            **stage,
            "status": "completed" if i < current_idx else ("active" if i == current_idx else "pending"),
            "order": i + 1,
        })
    return progression


def generate_stage_probability_vector(predicted_stage: str) -> dict:
    """Generate probabilities for each MITRE stage given predicted stage."""
    np.random.seed(42)
    stage_names = [s["name"] for s in MITRE_STAGES]
    predicted_idx = stage_names.index(predicted_stage) if predicted_stage in stage_names else 0

    # Assign high probability to predicted stage and lower to adjacent
    probs = np.array([0.02] * len(stage_names))
    probs[predicted_idx] = 0.72
    if predicted_idx > 0:
        probs[predicted_idx - 1] = 0.14
    if predicted_idx < len(stage_names) - 1:
        probs[predicted_idx + 1] = 0.09
    probs += np.random.uniform(0, 0.01, len(stage_names))
    probs = probs / probs.sum()

    return {stage_names[i]: round(float(probs[i]), 4) for i in range(len(stage_names))}


def get_technique_details_for_stage(stage_name: str) -> pd.DataFrame:
    """Return technique details for a given MITRE stage as a DataFrame."""
    stage = get_stage_by_name(stage_name)
    rows = []
    for tech in stage["techniques"]:
        rows.append({
            "Technique ID": tech["id"],
            "Technique Name": tech["name"],
            "Stage": stage_name,
            "Detection Confidence": round(np.random.uniform(0.65, 0.95), 2),
        })
    return pd.DataFrame(rows)


def get_mitre_heatmap_data() -> pd.DataFrame:
    """Generate heatmap data: stages x techniques with detection frequency."""
    np.random.seed(10)
    rows = []
    for stage in MITRE_STAGES:
        for tech in stage["techniques"]:
            rows.append({
                "Stage": stage["name"],
                "Technique": f"{tech['id']}: {tech['name']}",
                "Detection Count": int(np.random.randint(5, 80)),
            })
    return pd.DataFrame(rows)
