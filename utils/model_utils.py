"""
Model utilities for Network Attack Forecasting.
Provides mock LSTM/Transformer world model and logistic regression baseline.
"""

import numpy as np
import pandas as pd
from datetime import datetime


# ─────────────────────────────────────────────────────────────
#  Mock Model Architecture Description
# ─────────────────────────────────────────────────────────────

MODEL_ARCHITECTURES = {
    "World Model (LSTM)": {
        "type": "LSTM World Model",
        "layers": [
            {"name": "Input Layer", "units": 76, "activation": "—"},
            {"name": "LSTM Layer 1", "units": 256, "activation": "tanh"},
            {"name": "Dropout", "units": "—", "activation": "p=0.3"},
            {"name": "LSTM Layer 2", "units": 128, "activation": "tanh"},
            {"name": "Dropout", "units": "—", "activation": "p=0.3"},
            {"name": "Dense (Latent)", "units": 64, "activation": "ReLU"},
            {"name": "Output (Attack Prob)", "units": 1, "activation": "Sigmoid"},
        ],
        "params": "2,847,361",
        "optimizer": "Adam (lr=1e-3)",
        "loss": "Binary Cross-Entropy",
        "sequence_len": 20,
        "k_steps": 10,
    },
    "World Model (Transformer)": {
        "type": "Transformer World Model",
        "layers": [
            {"name": "Input Embedding", "units": 76, "activation": "—"},
            {"name": "Positional Encoding", "units": 76, "activation": "—"},
            {"name": "Multi-Head Attention", "units": "8 heads", "activation": "Softmax"},
            {"name": "Feed Forward 1", "units": 256, "activation": "GELU"},
            {"name": "Feed Forward 2", "units": 128, "activation": "GELU"},
            {"name": "Layer Norm", "units": "—", "activation": "—"},
            {"name": "Output (Attack Prob)", "units": 1, "activation": "Sigmoid"},
        ],
        "params": "3,124,224",
        "optimizer": "AdamW (lr=5e-4)",
        "loss": "Binary Cross-Entropy",
        "sequence_len": 20,
        "k_steps": 10,
    },
}


# ─────────────────────────────────────────────────────────────
#  Training Simulation
# ─────────────────────────────────────────────────────────────

def generate_training_curves(epochs: int = 50, model_type: str = "LSTM") -> pd.DataFrame:
    """Simulate training loss and validation loss curves."""
    np.random.seed(7)
    ep = np.arange(1, epochs + 1)

    if model_type == "LSTM":
        train_loss = 0.72 * np.exp(-0.065 * ep) + 0.08 + np.random.normal(0, 0.008, epochs)
        val_loss = 0.75 * np.exp(-0.058 * ep) + 0.10 + np.random.normal(0, 0.012, epochs)
        train_acc = 1 - (0.68 * np.exp(-0.07 * ep) + 0.06) + np.random.normal(0, 0.005, epochs)
        val_acc = 1 - (0.70 * np.exp(-0.063 * ep) + 0.075) + np.random.normal(0, 0.007, epochs)
        train_f1 = 1 - (0.65 * np.exp(-0.068 * ep) + 0.07) + np.random.normal(0, 0.006, epochs)
        val_f1 = 1 - (0.68 * np.exp(-0.062 * ep) + 0.09) + np.random.normal(0, 0.009, epochs)
    else:  # Transformer
        train_loss = 0.68 * np.exp(-0.072 * ep) + 0.07 + np.random.normal(0, 0.007, epochs)
        val_loss = 0.71 * np.exp(-0.065 * ep) + 0.09 + np.random.normal(0, 0.010, epochs)
        train_acc = 1 - (0.64 * np.exp(-0.075 * ep) + 0.055) + np.random.normal(0, 0.004, epochs)
        val_acc = 1 - (0.67 * np.exp(-0.068 * ep) + 0.07) + np.random.normal(0, 0.006, epochs)
        train_f1 = 1 - (0.61 * np.exp(-0.073 * ep) + 0.065) + np.random.normal(0, 0.005, epochs)
        val_f1 = 1 - (0.64 * np.exp(-0.066 * ep) + 0.085) + np.random.normal(0, 0.008, epochs)

    return pd.DataFrame({
        "Epoch": ep,
        "Train Loss": np.clip(train_loss, 0.05, 1.0).round(4),
        "Val Loss": np.clip(val_loss, 0.06, 1.0).round(4),
        "Train Accuracy": np.clip(train_acc, 0.5, 1.0).round(4),
        "Val Accuracy": np.clip(val_acc, 0.5, 1.0).round(4),
        "Train F1": np.clip(train_f1, 0.5, 1.0).round(4),
        "Val F1": np.clip(val_f1, 0.5, 1.0).round(4),
    })


def get_model_performance_metrics(model_type: str = "World Model (LSTM)") -> dict:
    """Return mock performance metrics for the selected model."""
    if "LSTM" in model_type:
        return {
            "accuracy": 0.9712,
            "precision": 0.9634,
            "recall": 0.9581,
            "f1_score": 0.9607,
            "fpr": 0.0231,
            "auc_roc": 0.9891,
            "training_time_s": 847,
            "inference_time_ms": 12.4,
        }
    elif "Transformer" in model_type:
        return {
            "accuracy": 0.9784,
            "precision": 0.9721,
            "recall": 0.9698,
            "f1_score": 0.9709,
            "fpr": 0.0187,
            "auc_roc": 0.9934,
            "training_time_s": 1243,
            "inference_time_ms": 18.7,
        }
    else:  # Logistic Regression baseline
        return {
            "accuracy": 0.8934,
            "precision": 0.8712,
            "recall": 0.8501,
            "f1_score": 0.8605,
            "fpr": 0.0821,
            "auc_roc": 0.9231,
            "training_time_s": 23,
            "inference_time_ms": 1.2,
        }


def get_confusion_matrix(model_type: str = "World Model (LSTM)") -> np.ndarray:
    """Return a mock confusion matrix."""
    if "LSTM" in model_type:
        # [[TN, FP], [FN, TP]]
        return np.array([[9821, 232], [178, 8769]])
    elif "Transformer" in model_type:
        return np.array([[9891, 162], [149, 8798]])
    else:
        return np.array([[9421, 632], [712, 8235]])


# ─────────────────────────────────────────────────────────────
#  SHAP / Explainability
# ─────────────────────────────────────────────────────────────

FEATURE_IMPORTANCE_DATA = {
    "Flow Byts/s": 0.187,
    "Flow Duration": 0.163,
    "Tot Fwd Pkts": 0.141,
    "Init Fwd Win Byts": 0.128,
    "Pkt Len Mean": 0.112,
    "SYN Flag Cnt": 0.098,
    "Flow IAT Mean": 0.087,
    "Fwd Pkt Len Mean": 0.076,
    "Tot Bwd Pkts": 0.064,
    "ACK Flag Cnt": 0.058,
    "Bwd Pkt Len Mean": 0.052,
    "Flow Pkts/s": 0.049,
    "Fwd IAT Mean": 0.043,
    "RST Flag Cnt": 0.039,
    "FIN Flag Cnt": 0.031,
    "PSH Flag Cnt": 0.028,
}


def get_shap_values(n_samples: int = 20) -> pd.DataFrame:
    """Generate mock SHAP values for a batch of samples."""
    np.random.seed(21)
    features = list(FEATURE_IMPORTANCE_DATA.keys())
    base_importances = np.array(list(FEATURE_IMPORTANCE_DATA.values()))

    shap_matrix = []
    for _ in range(n_samples):
        row = base_importances * np.random.uniform(0.5, 1.5, len(features))
        # Random sign: positive = pushes toward attack, negative = toward benign
        signs = np.random.choice([-1, 1], len(features), p=[0.35, 0.65])
        shap_matrix.append(row * signs)

    df = pd.DataFrame(shap_matrix, columns=features)
    return df


def get_attention_weights(seq_len: int = 20) -> np.ndarray:
    """Generate mock attention weight matrix (seq_len x seq_len)."""
    np.random.seed(33)
    raw = np.random.dirichlet(np.ones(seq_len) * 0.5, size=seq_len)
    # Make it slightly diagonal-dominant (attend to recent steps more)
    for i in range(seq_len):
        for j in range(seq_len):
            if abs(i - j) <= 2:
                raw[i][j] *= 2.5
    # Normalize rows
    raw = raw / raw.sum(axis=1, keepdims=True)
    return raw.round(4)


# ─────────────────────────────────────────────────────────────
#  Benchmark comparison
# ─────────────────────────────────────────────────────────────

BENCHMARK_MODELS = [
    "World Model (LSTM)",
    "World Model (Transformer)",
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "SVM",
]

BENCHMARK_METRICS = {
    "World Model (LSTM)": {
        "F1 Score": 0.9607, "Precision": 0.9634, "Recall": 0.9581,
        "Accuracy": 0.9712, "FPR": 0.0231, "AUC-ROC": 0.9891,
    },
    "World Model (Transformer)": {
        "F1 Score": 0.9709, "Precision": 0.9721, "Recall": 0.9698,
        "Accuracy": 0.9784, "FPR": 0.0187, "AUC-ROC": 0.9934,
    },
    "Logistic Regression": {
        "F1 Score": 0.8605, "Precision": 0.8712, "Recall": 0.8501,
        "Accuracy": 0.8934, "FPR": 0.0821, "AUC-ROC": 0.9231,
    },
    "Random Forest": {
        "F1 Score": 0.9312, "Precision": 0.9401, "Recall": 0.9228,
        "Accuracy": 0.9487, "FPR": 0.0412, "AUC-ROC": 0.9712,
    },
    "XGBoost": {
        "F1 Score": 0.9478, "Precision": 0.9512, "Recall": 0.9445,
        "Accuracy": 0.9601, "FPR": 0.0334, "AUC-ROC": 0.9801,
    },
    "SVM": {
        "F1 Score": 0.8901, "Precision": 0.9012, "Recall": 0.8794,
        "Accuracy": 0.9123, "FPR": 0.0645, "AUC-ROC": 0.9456,
    },
}


def get_benchmark_dataframe() -> pd.DataFrame:
    """Return benchmark comparison as a DataFrame."""
    rows = []
    for model, metrics in BENCHMARK_METRICS.items():
        row = {"Model": model}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def get_roc_curve_data() -> dict:
    """Generate mock ROC curve data for all benchmark models."""
    np.random.seed(55)
    curves = {}
    for model, metrics in BENCHMARK_METRICS.items():
        auc = metrics["AUC-ROC"]
        # Simulate ROC curve points
        fpr_pts = np.linspace(0, 1, 100)
        # Use a beta distribution shape to get realistic curve
        tpr_pts = np.power(fpr_pts, 1 / (auc * 3)) * (1 + np.random.normal(0, 0.01, 100))
        tpr_pts = np.clip(tpr_pts, 0, 1)
        tpr_pts[-1] = 1.0
        tpr_pts[0] = 0.0
        curves[model] = {"fpr": fpr_pts.tolist(), "tpr": np.sort(tpr_pts).tolist(), "auc": auc}
    return curves


# ─────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────

def get_model_status() -> dict:
    """Return mock model status info for the dashboard."""
    return {
        "model_name": "World Model (LSTM)",
        "version": "v2.1.0",
        "last_trained": "2024-03-12 14:32:00",
        "dataset": "CIC-IDS2018",
        "epochs_trained": 50,
        "status": "Ready",
        "device": "CPU",
        "total_params": "2,847,361",
    }
