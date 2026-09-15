# 🛡️ NetGuard AI — Network Attack Forecasting

**AI-based Network Attack Forecasting from Network Traffic Data**  
Smart India Hackathon (SIH) Project

---

## Overview

NetGuard AI is a professional Streamlit application that uses a LSTM/Transformer-based **World Model** to forecast network attacks from traffic flow data (CIC-IDS2018 feature set). It provides real-time threat visualization, K-step attack probability forecasting, MITRE ATT&CK stage mapping, and model explainability.

---

## Features

| Page | Description |
|------|-------------|
| 📊 Dashboard | Live metrics, attack probability gauge, alert timeline |
| 📂 Data Ingestion | Upload CSV/PCAP, feature extraction, 3D scatter |
| 🧠 Model Training | LSTM/Transformer config, live training curves |
| 🔮 Attack Prediction | K-step forward simulation, stage annotation |
| 🗺️ MITRE ATT&CK | Kill chain progression, technique mapping |
| 💡 Explainability | SHAP beeswarm, attention heatmap, feature importance |
| 📈 Benchmark | vs. LR/RF/XGBoost/SVM — ROC, confusion matrix, radar |
| ⚙️ Settings | Thresholds, model config, system info |

---

## Quick Start

### 1. Clone / copy the project

```bash
cd "SIH project"
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
SIH project/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
└── utils/
    ├── __init__.py
    ├── data_processor.py       # CIC-IDS2018 data generation & processing
    ├── model_utils.py          # Mock LSTM/Transformer model utilities
    ├── visualizations.py       # All Plotly chart functions
    └── mitre_attack.py         # MITRE ATT&CK stage mapping
```

---

## Technical Stack

- **Frontend**: Streamlit 1.32 + custom CSS/HTML/JS
- **Charts**: Plotly 5.20 (2D, 3D, gauge, heatmap, radar)
- **Data**: Pandas, NumPy (CIC-IDS2018 feature schema)
- **ML**: Scikit-learn (benchmark baselines), mock LSTM/Transformer
- **Animated Background**: HTML5 Canvas + vanilla JS (network graph particles)
- **Theme**: Light blues/purples, Inter font, glassmorphism cards

---

## Dataset

The application uses the **CIC-IDS2018** feature set (76 features) including:
- Flow-level statistics (duration, byte counts, packet counts)
- TCP flag counters (SYN, ACK, FIN, RST, PSH, URG)
- Inter-arrival time (IAT) features
- Window size and subflow features
- Active/Idle time features

Attack categories supported:
- DDoS (LOIC-HTTP, LOIC-UDP)
- DoS (Hulk, SlowHTTPTest, GoldenEye)
- Brute Force (FTP, SSH, Web, XSS)
- Infiltration / Lateral Movement
- Bot / C&C traffic
- SQL Injection

---

## MITRE ATT&CK Stages

The app maps network attack patterns to 6 MITRE ATT&CK tactics:

1. 🔍 **Reconnaissance** (TA0043)
2. 🚪 **Initial Access** (TA0001)
3. ⚙️ **Execution** (TA0002)
4. ↔️ **Lateral Movement** (TA0008)
5. 📡 **Command & Control** (TA0011)
6. 📤 **Exfiltration** (TA0010)

---

## Model Architecture

### World Model (LSTM)
```
Input (76 features, seq_len=20)
  → LSTM Layer 1 (256 units, tanh)
  → Dropout (0.3)
  → LSTM Layer 2 (128 units, tanh)
  → Dropout (0.3)
  → Dense Latent (64, ReLU)
  → Output (1, Sigmoid) — Attack Probability
```

### World Model (Transformer)
```
Input Embedding (76 → 76)
  → Positional Encoding
  → Multi-Head Self-Attention (8 heads)
  → Feed Forward (256 → 128, GELU)
  → Layer Normalization
  → Output (1, Sigmoid) — Attack Probability
```

---

## Deployment

### Streamlit Cloud

1. Push the project to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo, set `app.py` as the main file
4. Deploy — no extra configuration needed

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t netguard-ai .
docker run -p 8501:8501 netguard-ai
```

---

## Performance (Mock Benchmarks)

| Model | F1 Score | Precision | Recall | FPR | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| World Model (Transformer) | **0.9709** | **0.9721** | **0.9698** | **0.0187** | **0.9934** |
| World Model (LSTM) | 0.9607 | 0.9634 | 0.9581 | 0.0231 | 0.9891 |
| XGBoost | 0.9478 | 0.9512 | 0.9445 | 0.0334 | 0.9801 |
| Random Forest | 0.9312 | 0.9401 | 0.9228 | 0.0412 | 0.9712 |
| SVM | 0.8901 | 0.9012 | 0.8794 | 0.0645 | 0.9456 |
| Logistic Regression | 0.8605 | 0.8712 | 0.8501 | 0.0821 | 0.9231 |

---

## License

MIT License — SIH Project 2024
