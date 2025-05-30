from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import analysis, health
from core.config import settings
from contextlib import asynccontextmanager
from core.models import SentimentModel, get_model

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
    allow_origins=["http://localhost:3000"],  # Your React frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]  # Important for some responses
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # This will raise an exception if model files are missing
        get_model()
        yield
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model loading failed: {str(e)}. "
                   "Please ensure you have placed bigru_model.keras and tokenizer.pkl in the models/ directory."
        )

app = FastAPI(lifespan=lifespan)