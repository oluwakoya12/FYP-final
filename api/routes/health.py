from fastapi import APIRouter
from core.models import SentimentModel
from api.schemas import HealthCheck

router = APIRouter()

@router.get("/", response_model=HealthCheck)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": SentimentModel.model is not None
    }