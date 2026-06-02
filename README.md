# Credit Risk Intelligence Platform (AI-Powered)

An end-to-end **credit risk intelligence** platform built on the **Home Credit Default Risk** dataset. It includes:

- **Streamlit** dashboard for executives and analysts
- **XGBoost** model training and scoring for default risk prediction
- **SHAP** explainability for model transparency
- **SQLite** analytics layer for fast, local portfolio queries
- **Local-first “Talk to Data” chatbot** (keyword → predefined SQL → answers), with optional **OpenAI/Gemini** fallback for open-ended questions

---

## Project overview

This project is designed to be a practical portfolio-grade implementation of a modern risk analytics system:

- **Business dashboards**: portfolio composition, risk distributions, KPIs
- **Model scoring**: predict default risk for individual applications
- **Explainability**: understand which features drive model outputs
- **Data intelligence**: ask common business questions in plain English, safely executed as SQL

The platform runs fully locally with SQLite and predefined queries; LLM calls are only made when the question does not match a known template.

---

## Architecture

### High-level components

```text
┌────────────────────────────┐
│        Streamlit UI         │  :8501
│  - Dashboard / scoring      │
│  - SHAP explainability      │
│  - Data chatbot             │
└──────────────┬─────────────┘
               │
               │ local Python calls (same codebase)
               ▼
┌────────────────────────────┐
│      Analytics + ML Core    │
│  src/data   (SQLite loader) │
│  src/ml     (train/predict) │
│  src/talk_to_data (NL→SQL)  │
└──────────────┬─────────────┘
               │
               │ SQL (read-only)
               ▼
┌────────────────────────────┐
│         SQLite DB           │
│  data/credit_risk.db        │
│  table: customers           │
└────────────────────────────┘

┌────────────────────────────┐
│         FastAPI API          │  :8000
│  POST /predict  /chat        │
│  GET  /metrics               │
└────────────────────────────┘
```

### Repository structure

```text
credit_risk_platform/
├── app.py                        # Streamlit app entrypoint
├── src/
│   ├── api/                      # FastAPI app + services
│   ├── data/                     # dataset loader + SQLite database tools
│   ├── ml/                       # training, evaluation, SHAP explainability
│   ├── talk_to_data/             # chatbot (local templates + LLM fallback) + security layers
│   ├── ui/                       # Streamlit views + styles
│   └── utils/                    # config, logging, helpers
├── data/                         # dataset + SQLite DB (mounted in Docker, not committed)
├── models/                       # trained model artifacts (mounted in Docker, not committed)
├── documents/                    # reports and exported figures
├── notebooks/                    # EDA notebooks/scripts
├── sql/                          # schema helpers / notes
├── Dockerfile                    # Python 3.11 image
├── docker-compose.yml            # Streamlit + FastAPI services
├── requirements.txt              # dev + full deps
├── requirements-docker.txt       # runtime deps for Docker images
└── .env.example                  # environment variables template
```

---

## Setup (local)

### 1) Install dependencies

```bash
cd credit_risk_platform
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2) Add dataset

Download **Home Credit Default Risk** from Kaggle and place this file at:

- `data/application_train.csv`

Dataset link: [Home Credit Default Risk on Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)

### 3) Load SQLite database

Create the SQLite database used by the chatbot and portfolio analytics.

```bash
python -m src.data.database --load
```

Notes:
- Default load is **10,000 rows** for fast iteration.
- Use the full dataset:

```bash
python -m src.data.database --load --max-rows 0
```

### 4) Train the model

```bash
python -m src.ml.train
```

This writes the trained artifact to `models/model.pkl` by default (configurable via `MODEL_PATH`).

### 5) Run the Streamlit dashboard

```bash
streamlit run app.py
```

Or use the helper:

```bash
./run_app.sh
```

Open the UI at `http://localhost:8501`.

### 6) Run the FastAPI backend (optional)

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Or:

```bash
./run_api.sh
```

API docs: `http://localhost:8000/docs`

---

## Docker instructions (Streamlit + FastAPI)

### Prerequisites

- Docker Engine + Docker Compose v2

### Start services

```bash
docker compose up --build
```

URLs:
- **Streamlit**: `http://localhost:8501`
- **FastAPI docs**: `http://localhost:8000/docs`

### Persistent data/model volumes

Compose mounts these directories into containers:

- `./data:/app/data`
- `./models:/app/models`

So you can train/load on the host and reuse artifacts in Docker.

### Environment variables

- `.env` is optional in Docker Compose (keys and settings are loaded if present).
- See `.env.example` for all supported variables.

---

## ML model explanation (XGBoost)

The platform trains an **XGBoost binary classifier** to predict whether an application will default (`TARGET = 1`).

Key design choices:
- **Preprocessing**: missing value handling + encoding + scaling (`src/data/preprocessor.py`)
- **Imbalance handling**: configuration supports class-weighting/threshold selection
- **Evaluation**: ROC-AUC and classification metrics (`src/ml/evaluate.py`)
- **Decision threshold tuning**: tuned for business-relevant trade-offs (precision/recall)

Primary training entrypoint:

```bash
python -m src.ml.train
```

---

## SHAP explainability

SHAP helps explain which features are driving a prediction and which features matter globally.

Generate SHAP reports/plots:

```bash
python -m src.ml.shap_explainer
```

Outputs are written under `documents/shap/` (depending on configuration).

In the Streamlit app, use the **SHAP explainability** page to view the explainability artifacts.

---

## Chatbot workflow (Talk to Data)

The chatbot is designed to be **safe, useful, and resilient**:

### 1) Local-first matching (no API call)

Common business questions are detected using keyword matching in:
- `src/talk_to_data/local_chatbot.py`

Matched questions map to **predefined SQL** queries executed directly on SQLite, returning a **business-readable** answer.

Supported examples:
- “How many high-risk customers exist?”
- “What is the average income of defaulters?”
- “Which age group has highest default rate?”
- “Which loan type is riskiest?”
- “Give me a portfolio risk summary.”

### 2) LLM fallback (OpenAI/Gemini) for open-ended questions

If no local template matches, the system can call OpenAI/Gemini to generate SQL — then it is validated and safely executed.

Configuration (in `.env`):
- `LLM_PROVIDER=openai` and `OPENAI_API_KEY=...`
- or `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...`

### 3) Security & safety layers

The NL→SQL path applies multiple guardrails (see `src/talk_to_data/security.py`), including:
- input validation
- prompt-injection defenses
- strict SQL allowlisting (SELECT-only, table allowlist)
- enforced `LIMIT`
- safe execution + output sanitization

### 4) Quota/error handling (never crash UI)

If the AI provider fails (quota, rate limiting, service errors), the chatbot:
- retries local templates when possible
- returns a friendly fallback message
- **never crashes the Streamlit app**

CLI examples:

```bash
python -m src.talk_to_data.nl_engine --examples --no-llm-answer
```

---

## Screenshots (placeholders)

Add screenshots under `documents/screenshots/` and update the links below.

- **Dashboard**
  - `documents/screenshots/dashboard.png`
- **Risk prediction**
  - `documents/screenshots/prediction.png`
- **Model metrics**
  - `documents/screenshots/metrics.png`
- **SHAP explainability**
  - `documents/screenshots/shap.png`
- **Data chatbot**
  - `documents/screenshots/chatbot.png`

Example markdown (replace once you add files):

```md
![Dashboard](documents/screenshots/dashboard.png)
```

---

## Future improvements

- **Better local knowledge base**: expand templates, add synonym dictionaries, add parameter extraction (e.g., “top 10 occupations”)
- **Caching**: cache frequent SQLite queries and common aggregations
- **Model monitoring**: drift detection, data quality checks, performance over time
- **MLOps**: CI for training reproducibility, artifact versioning, model registry
- **Role-based access**: restrict sensitive analytics queries by user role
- **Improved explainability UX**: per-customer explanations in the UI, exportable reports
- **Async/background jobs**: dataset loading and SHAP generation as background tasks

---

## License / data terms

For educational/portfolio use. The Home Credit dataset is subject to Kaggle competition terms.
