from pydantic import BaseModel
from typing import List, Optional, Dict, Union

class AnalysisResult(BaseModel):
    sentiment_distribution: Dict[str, float]
    feature_impacts: List[Dict[str, Union[str, float]]]
    sample_predictions: List[Dict[str, str]]
    raw_data: Optional[List[Dict]] = None

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool