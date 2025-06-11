# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import analysis, health
from core.config import settings
from core.models import sentiment_model  # this loads the model on import

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION
)


# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # ← Use the defined list here!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])

@app.on_event("startup")
async def startup_event():
    """Confirm model loaded at startup"""
    try:
        _ = sentiment_model.model  # access to confirm loading
        print("Startup: Sentiment model ready.")
    except Exception as e:
        print(f"Error during model load: {e}")
        raise
