from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from api.routes import analysis, health
from core.config import settings

app = FastAPI(title=settings.PROJECT_NAME, 
              version=settings.VERSION,
              description=settings.DESCRIPTION)

# CORS
# CORS Configuration
origins = [
    "http://localhost:3000",  # React dev server
    "http://127.0.0.1:3000",  # Alternative localhost
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    from core.models import load_models
    load_models()  # Load ML models when app starts