# MLOps Loan Default Prediction – Architectural Design & Decision Log

This document provides a detailed overview of the system architecture, core design patterns, and engineering trade-offs implemented in this project. These decisions are tailored to ensure production-grade reliability, compliance with banking standards, and high performance while operating entirely within standard local development and production environments.

---

## 🗺️ Architectural Diagram

For a high-level visual understanding of the data loops, validation checks, training cycles, and API boundaries, refer to:
*   Vector Diagram: [architecture.svg](./architecture.svg)
*   Interactive Presentation: [architecture_viewer.html](./architecture_viewer.html) (open in your browser to step through the lifecycle interactively).

---

## ⚡ Core Architectural Components

The system is split into four distinct layers, each designed to run asynchronously, securely, and with minimal memory footprint:

### 1. The Serving & Orchestration Layer (FastAPI)
*   **Role**: Handles real-time scoring (`/predict`), metadata status reporting (`/db-status`, `/pipeline-status`), and starts retraining processes.
*   **Implementation**: Enforces asynchronous routes where possible and handles long-running jobs (like model retraining) out-of-band using python thread-executors.

### 2. The Data Layer (Supabase PostgreSQL)
*   **Role**: Stores historical application training data and captures real-time application records (simulating production data feeds).
*   **Implementation**: Interfaced via a connection pool to minimize host connection overhead and prevent transaction latency over the cloud.

### 3. The Validation & Drift Audit Layer (Great Expectations & Evidently AI)
*   **Role**: Acts as a production guardrail.
    *   **Great Expectations** ensures that no corrupt data structure or invalid feature ranges (e.g., positive `days_birth` or negative credit values) can enter the training set.
    *   **Evidently AI** evaluates features for covariate drift (Kolmogorov-Smirnov test) and target drift, triggering model updates only when drift exceeds the `35%` threshold.

### 4. The Experiment Tracking & Lifecycle Layer (MLflow & XGBoost)
*   **Role**: Versions trained models, logs experiment parameters, audits metrics (Accuracy, ROC-AUC, Recall, Precision, F1-score), and stores evaluation curve data.
*   **Implementation**: Uses DagsHub's free hosted MLflow server as a remote registry, keeping a local serialized XGBoost JSON copy on the disk as a cold-standby cache.

---

## 🛠️ Design Patterns Implemented

### 1. Lazy-Initialized Global Connection Pool Pattern
*   **File**: [src/database.py](./src/database.py)
*   **Rationale**: Opening and closing a separate TLS/SSL database connection on every API request or batch transaction is slow and exhausts the connection limits of database hosts.
*   **Details**: We initialize a global `SimpleConnectionPool` upon the first database request. Database wrapper functions borrow a connection from the pool and return it in a `try...finally` block, ensuring that connections are never leaked even if query execution fails.

### 2. Asynchronous Background Task Worker Pattern
*   **File**: [app/main.py](./app/main.py)
*   **Rationale**: ML retraining runs can take minutes. Running them on the main web request thread would cause a HTTP timeout (504 Gateway Timeout) and freeze the server.
*   **Details**: The `/trigger-retrain` endpoint enqueues `run_retraining_pipeline_task` into FastAPI's `BackgroundTasks` queue. It immediately returns a JSON response containing `{"success": True}` to the caller. The frontend UI polls a lightweight GET `/pipeline-status` endpoint to fetch in-memory logs and progress states, giving users a smooth, interactive experience.

### 3. Local Offline Caching & Fallback Pattern
*   **File**: [app/main.py](./app/main.py)
*   **Rationale**: Remote model registries (like MLflow) are accessed over the public internet. If the registry server is down, rate-limited, or the local environment loses connection, the API server must not crash or fail to start up.
*   **Details**: During startup, `load_registered_model` attempts to pull the active model from MLflow. If successful, it caches a copy locally at `models/fallback_model.json`. If MLflow is unreachable, the system catches the exception, logs a warning, loads `models/fallback_model.json` using raw XGBoost serialization, wraps it in a fallback adapter, and remains online.

### 4. Monotonicity-Constrained Machine Learning Pattern
*   **File**: [src/train.py](./src/train.py)
*   **Rationale**: Tree-based models (like XGBoost) can learn erratic decision boundaries due to localized noise in the training set. For instance, a model might predict that an applicant with an external credit score of `0.2` has *less* risk of default than an applicant with a score of `0.6` due to sparse data. In banking, this is an illegal default prediction abnormality.
*   **Details**: We enforce strict **monotonic constraints** during training.
    *   Worsening credit scores (`ext_source_2`, `ext_source_3`, `ext_source_mean`, `ext_source_prod`) are constrained to have a non-increasing relationship with target risk (`-1`).
    *   Rising debt ratios (`annuity_income_ratio`, `credit_income_ratio`) are constrained to have a non-decreasing relationship with target risk (`+1`).
    This ensures that poorer credit profiles and higher debt levels *always* translate to equal or higher default probabilities.

---

## ⚖️ Architectural Trade-offs & Stack Selection

When building a production MLOps pipeline, engineers often resort to heavy, complex enterprise tools. Below is a justification of the trade-offs made in this project to optimize the architecture for low overhead, decoupled scale, and local developer environments:

| Enterprise Standard Stack | Our Chosen Stack | Resource & Performance Justification |
| :--- | :--- | :--- |
| **Apache Airflow / Prefect** (Full daemon suite with web server, scheduler, database, and workers) | **FastAPI BackgroundTasks** | Airflow requires a local Postgres instance, worker daemons, and metadata schedulers running constantly, consuming >1.5GB RAM and significant CPU. FastAPI BackgroundTasks runs within the existing server process, consuming **0MB additional RAM** and executing pipeline code asynchronously. |
| **Celery + RabbitMQ / Redis** | **FastAPI In-Memory Task Queue** | Celery requires installing a separate broker (Redis or RabbitMQ) and running a background worker process. This requires extra configuration, consumes ~300MB RAM, and complicates deployment on single-container cloud hosting environments. |
| **Self-Hosted MLflow Server + S3 Bucket** | **DagsHub Hosted MLflow Registry** | Hosting MLflow locally with an S3-compatible backend requires cloud storage costs, setup time, and continuous background RAM. DagsHub provides a hosted MLflow tracking and model registry server with zero local resource overhead. |
| **Kubeflow / SageMaker Pipelines** | **Evidently AI & Great Expectations Scripts** | Kubeflow requires a full Kubernetes cluster, costing hundreds of dollars per month and requiring massive computing power. Our validation scripts run locally as fast Python tasks during retraining, taking seconds and consuming negligible memory. |
| **Apache Kafka / Spark Streaming** | **Supabase Postgres Event Log** | Real-time feature logging can be done in Postgres. Ingesting new inference transactions into a dedicated schema table acting as an event log is simple, lightweight, and keeps database structures clean. |

---

## 🚀 Operational Readiness & Testing Strategy

To ensure code quality and operational resilience, the repository contains a test suite (`tests/test_pipeline.py`) that can be executed with a single command:

```bash
$env:PYTHONPATH="."; pytest -v
```

The test suite covers:
1.  **API Health Checks**: Confirms server status and validates that the model is loaded.
2.  **Preprocessing Integrity**: Checks that categorical labels are correctly mapped to integer encodings.
3.  **Predictive Routing**: Assures `/predict` behaves gracefully, returning a `503 Service Unavailable` if the model is not initialized.
4.  **Feature Engineering Correctness**: Mathematically validates the calculation of engineered risk ratios (e.g., DTI, Credit-to-Income, age conversion, and external score aggregates) against expected inputs.
5.  **Monotonicity Validation**: Generates synthetic user profiles with deteriorating credit scores (`ext_source_2` ranging from `0.9` down to `0.1`). It asserts that default probability strictly increases or stays flat as credit score deteriorates, validating that the mathematical monotonic constraint holds true in serving.

---

## ⚖️ Observability & Log Auditing

Observability is maintained through:
*   **Console Logging**: Beautifully formatted logs detailing model loading stages, connection pooling initialization, and endpoint routing.
*   **Audit File Logging**: Every step of the MLOps pipeline (validation metrics, drift detection results, and training progress) is appended directly to [reports/pipeline_execution.log](./reports/pipeline_execution.log), creating an offline audit trail.
*   **Data Drift Report**: Saved to [reports/data_drift_report.html](./reports/data_drift_report.html) for detailed visual monitoring of Kolmogorov-Smirnov statistical tests.
