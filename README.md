# Retail Sales Forecasting MLOps Platform

A production-quality, end-to-end Machine Learning system for Retail Sales Forecasting based on the Kaggle *Store Sales - Time Series Forecasting* dataset. Built following **Clean Architecture**, **SOLID principles**, and **MLOps best practices**.

---

## 🏗️ Architecture & Project Structure

The project is organized into decoupled, single-responsibility modules:

```text
forecasting_retail/
├── config/             # YAML configurations for hyperparameters & features
├── data/               # Raw, processed (Parquet), and external datasets
├── src/
│   ├── config/         # Pydantic Settings & YAML loader
│   ├── domain/         # Core business entities, protocols, and validation schemas
│   ├── application/    # Use cases (ingestion, validation, cleaning, merging)
│   ├── infrastructure/ # External services (KaggleHub repository, versioning)
│   ├── features/       # Modular feature transformers (Calendar, Lag, Rolling)
│   ├── training/       # Model trainers (LightGBM, XGBoost, Random Forest) & MLflow registry
│   ├── evaluation/     # Metrics evaluator (MAE, RMSE, MAPE, R²)
│   ├── inference/      # Predictor service
│   ├── reports/        # Business report generators (weekly/monthly, JSON/PDF)
│   ├── api/            # FastAPI REST endpoints
│   ├── streamlit/      # Multi-page interactive Streamlit dashboard
│   └── utils/          # Structured logger
├── tests/              # Unit and integration tests (pytest)
├── Dockerfile          # Container image definition
└── docker-compose.yml  # Multi-service container orchestration
```

---

## 🚀 Quick Start & Installation

### 1. Virtual Environment & Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Running Tests

```bash
pytest
```

### 3. Starting Services

* **FastAPI REST API:**

  ```bash
  uvicorn src.api.main:app --reload --port 8000
  ```
  * Interactive API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Streamlit Dashboard:**

  ```bash
  streamlit run src/streamlit/app.py
  ```
* **Docker Compose:**

  ```bash
  docker-compose up --build
  ```
