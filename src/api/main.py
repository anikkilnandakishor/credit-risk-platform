"""
Credit Risk Platform — FastAPI backend.

Endpoints:
  POST /predict  — loan default risk scoring
  GET  /metrics  — model evaluation metrics
  POST /chat     — natural-language analytics (secured NL → SQL)

Run:
  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api import services
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    MetricsResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)
from src.talk_to_data.security import SecurityError
from src.utils.config import MODEL_PATH

MODEL_LOAD_ERROR = f"Trained model not found. Run: python -m src.ml.train ({MODEL_PATH})"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        services.load_model()
    except services.ModelNotLoadedError:
        pass
    yield
    services.unload_model()


app = FastAPI(
    title="Credit Risk Platform API",
    description="Prediction, metrics, and AI chatbot for Home Credit risk analytics.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "service": "Credit Risk Platform API",
        "endpoints": ["/predict", "/metrics", "/chat"],
        "docs": "/docs",
        "model_loaded": services.is_model_loaded(),
        "database_ready": services.is_database_ready(),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(body: PredictRequest) -> PredictResponse:
    """
    Score a single loan application using the saved XGBoost model.

    Returns JSON with default probability, binary prediction, and risk label.
    """
    if not services.is_model_loaded():
        raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR)
    try:
        return services.predict_application(body)
    except services.ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(
    "/predict/batch",
    response_model=PredictBatchResponse,
    tags=["Prediction"],
    include_in_schema=True,
)
def predict_batch_endpoint(body: PredictBatchRequest) -> PredictBatchResponse:
    """Score up to 100 applications in one request."""
    if not services.is_model_loaded():
        raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR)
    return services.predict_applications(body.applications)


@app.get("/metrics", response_model=MetricsResponse, tags=["Model"])
def metrics() -> MetricsResponse:
    """Return evaluation metrics stored with the trained model artifact."""
    if not services.is_model_loaded():
        raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR)
    return services.get_model_metrics()


@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
def chat(body: ChatRequest) -> ChatResponse:
    """
    Ask a business question about the portfolio.

    Uses the secured NL→SQL pipeline and returns answer, SQL, and result rows as JSON.
    """
    try:
        return services.chat_with_data(body.question)
    except SecurityError as exc:
        raise HTTPException(status_code=400, detail={"error": "security", "message": str(exc)})
    except services.DatabaseNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.exception_handler(services.ModelNotLoadedError)
def model_not_loaded_handler(_request, exc: services.ModelNotLoadedError):
    return JSONResponse(status_code=503, content={"error": "model_not_loaded", "message": str(exc)})


@app.exception_handler(services.DatabaseNotReadyError)
def database_not_ready_handler(_request, exc: services.DatabaseNotReadyError):
    return JSONResponse(status_code=503, content={"error": "database_not_ready", "message": str(exc)})
