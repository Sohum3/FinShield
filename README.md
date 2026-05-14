# FinShield
End-to-end ad click fraud detection system for fintech — two-stage ML pipeline (rule engine + XGBoost), SHAP explainability, PostgreSQL, and a Streamlit dashboard with real-time fraud simulation and cost impact analysis in ₹

# 🛡️ FinShield — Ad Click Fraud Detection for Financial Services

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-blue)](https://postgresql.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)](https://shap.readthedocs.io)

---

## 📌 Business Problem

Ad fraud costs the Indian digital advertising industry an estimated **₹3,000+ crore annually**. The problem is especially severe in fintech — credit card, lending, UPI, insurance, and trading app ads command some of the highest cost-per-click (CPC) rates in the country (₹90–₹200 per click), making them prime targets for bot operators and click farms.

FinShield is an end-to-end fraud detection system built specifically for financial services ad campaigns. It detects fraudulent clicks in real time, quantifies the financial damage in rupees, and explains every fraud flag transparently — making it useful for both data teams and business stakeholders.

**In this project, FinShield detected ₹2.76 crore in fraudulent ad spend across a 4-day dataset of 1 million clicks.**

---

## 🏗️ Architecture

```
Raw Click Data (380M rows)
        │
        ▼
  Stratified Sample (1M rows)
        │
        ▼
┌─────────────────────────────┐
│     EDA & Feature           │
│     Engineering             │
│  (19 features built)        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   STAGE 1 — Rule Engine     │
│   Hard thresholds on        │
│   velocity, diversity,      │
│   suspicious hours          │
│   Precision: 99.9%          │
│   Fraud caught: 13.5%       │
└─────────────┬───────────────┘
              │ Passes through
              ▼
┌─────────────────────────────┐
│   STAGE 2 — XGBoost         │
│   Catches sophisticated     │
│   fraud that rules miss     │
│   PR-AUC: 0.5544            │
│   Recall: 84.5%             │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   SHAP Explainability       │
│   Local + Global            │
│   feature importance        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   PostgreSQL Database       │
│   + SQL EDA Queries         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Streamlit Dashboard       │
│   4 tabs + Company Upload   │
└─────────────────────────────┘
```

---

## 📊 Key EDA Findings

**1. Trading apps attract 56x more genuine users than UPI apps**
Trading app ads show a 14.12% conversion rate vs 0.02% for UPI apps. Fraudsters disproportionately target low-conversion verticals where bot traffic is harder to detect against the baseline noise.

**2. The top 1% of IPs by volume account for 21.5% of all fraud**
Just 814 IP addresses out of 81,453 drive over a fifth of all fraudulent clicks. The top bot IP alone generated 6,818 clicks across 69 unique apps in 3 days — a pattern impossible for any human user.

**3. Legitimate IPs click 1 unique app on average. Fraud IPs click 5.**
IP app diversity is the single strongest behavioural signal separating bots from genuine users. Bots are paid per click and spread across as many apps as possible to maximise payout.

**4. 56.7% of genuine downloads happen within 10 minutes of the click**
Legitimate users who intend to download act quickly. This makes time-to-download a highly discriminative signal — though we excluded it from the ML model to avoid data leakage (it is only known after the fact).

**5. Fraud rate is statistically significantly different across channels, devices, OS, and verticals**
Chi-square tests confirm p≈0 for all four features. However, raw linear correlations with the target are all under 0.06 — confirming that fraud patterns are non-linear and justifying the use of XGBoost over logistic regression.

---

## 🔬 Design Decisions

### Why a two-stage pipeline instead of a single model?
The rule engine catches 99.9% precision fraud (obvious bots) cheaply and fast, without needing a model at all. XGBoost then handles the harder cases that rules miss. This mirrors how production fraud systems work at companies like Razorpay and InMobi — rules for speed and precision, ML for coverage.

### Why XGBoost over AdaBoost?
XGBoost handles the extreme class imbalance (99.75% non-fraud) natively via `scale_pos_weight`. It is also parallelised and significantly faster on large datasets. AdaBoost is sequential by design and cannot scale as efficiently.

### Why Precision-Recall AUC over ROC-AUC?
With only 0.25% positive class (genuine downloads), a model that predicts every click as fraudulent achieves 99.75% accuracy and a misleadingly high ROC-AUC. PR-AUC directly measures performance on the minority class and is the correct metric for this problem. Our random baseline PR-AUC is 0.0027 — our model achieves 0.5544, which is 200x better than random.

### Why SHAP over LIME?
SHAP gives mathematically consistent explanations every time (grounded in Shapley values from game theory). LIME produces different explanations on repeated runs due to random perturbation. SHAP also supports both local explanations (why was this specific click flagged) and global feature importance from the same tool — LIME only does local.

---

## 📈 Results

| Metric | Value |
|---|---|
| Dataset size | 1,000,000 clicks |
| Stage 1 — Rule Engine fraud caught | 13.5% of all fraud |
| Stage 1 — Rule precision | 99.9% |
| Stage 2 — XGBoost PR-AUC | 0.5544 |
| Stage 2 — Random baseline PR-AUC | 0.0027 |
| Stage 2 — Recall (legitimate clicks) | 84.5% |
| Combined pipeline fraud caught | 30.6% of all fraud |
| Total ad spend in dataset | ₹9.07 crore |
| Fraudulent ad spend | ₹9.04 crore (99.6%) |
| Recovered by FinShield | ₹2.76 crore |

> **Note on 99.6% fraudulent spend:** This reflects the nature of the TalkingData dataset where `is_attributed=0` includes all non-converting clicks, not exclusively malicious bot traffic. Real-world fraud rates are lower. The cost impact figures are illustrative of the methodology.

---

## 💰 Cost Impact by Fintech Vertical

| Vertical | Total Spend | Fraudulent Spend | Fraud Rate | Conversion Rate |
|---|---|---|---|---|
| Other | ₹6.51 crore | ₹6.50 crore | 99.86% | 0.14% |
| Lending | ₹1.57 crore | ₹1.57 crore | 99.77% | 0.23% |
| UPI | ₹0.78 crore | ₹0.78 crore | 99.98% | 0.02% |
| Insurance | ₹0.11 crore | ₹0.10 crore | 94.41% | 5.59% |
| Trading | ₹0.10 crore | ₹0.09 crore | 85.88% | 14.12% |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| Data & EDA | Pandas, NumPy, Matplotlib, Seaborn, Plotly |
| Database | PostgreSQL 18, SQLAlchemy, psycopg2 |
| Machine Learning | XGBoost, scikit-learn, imbalanced-learn |
| Explainability | SHAP |
| Experiment Tracking | Weights & Biases |
| Dashboard | Streamlit |
| Dataset | TalkingData Ad Tracking Fraud Detection (Kaggle) |

---

## 📁 Project Structure

```
finshield/
├── data/
│   ├── processed/          ← cleaned + engineered CSV
│   └── samples/            ← 1M row stratified sample
├── notebooks/
│   ├── 01_eda.ipynb        ← full EDA with business narratives
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb
├── src/
│   ├── rules.py            ← Stage 1 rule engine
│   ├── model.py            ← Stage 2 XGBoost pipeline
│   ├── explainer.py        ← SHAP logic
│   ├── cost_impact.py      ← rupee calculations
│   └── sql_eda.py          ← SQL EDA queries
├── app/
│   └── streamlit_app.py    ← four-tab Streamlit dashboard
├── models/
│   ├── xgb_model.pkl
│   ├── feature_cols.pkl
│   ├── shap_explainer.pkl
│   ├── shap_values_sample.csv
│   └── X_shap_sample.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

### Prerequisites
- Python 3.10+
- PostgreSQL 18 installed and running
- Git

### 1. Clone the repository
```bash
git clone https://github.com/Sohum3/finshield.git
cd finshield
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up PostgreSQL
```bash
psql -U postgres
```
```sql
CREATE DATABASE finshield;
\q
```

### 5. Download the dataset
Download `train.csv` from the [TalkingData Kaggle competition](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data) and place it in `data/raw/`.

### 6. Run the sampling and feature engineering notebook
Open and run `notebooks/01_eda.ipynb` and `notebooks/02_feature_engineering.ipynb` in order. This generates `data/processed/train_engineered.csv`.

### 7. Load data into PostgreSQL
```bash
python src/load_data.py
```

### 8. Launch the dashboard
```bash
streamlit run app/streamlit_app.py
```

Open your browser at `http://localhost:8501`

---

## 📦 requirements.txt

```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
plotly
streamlit
sqlalchemy
psycopg2-binary
joblib
scipy
wandb
kaggle
```

---

## 🙋 About

Built by **Sohum Sharma** — Data Science student at PES University, Bengaluru (graduating 2026).

- 📧 sohumwork2001@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/sohum2001/)
- 🐙 [GitHub](https://github.com/Sohum3)
- 🌐 [Portfolio](https://sohums-portfolio-website.vercel.app/)
