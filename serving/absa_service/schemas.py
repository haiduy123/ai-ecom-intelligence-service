from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class Aspect(BaseModel):
    aspect: str = Field(min_length=1, max_length=200)
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(..., description="The sentiment associated with the aspect.")
    evidence: str = Field(default="", max_length=2000)

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_id: str
    comment: str = Field(min_length=1, max_length=20000)

class PredictResponse(BaseModel):
    model_name: str
    review_id: str | None = None
    aspects: list[Aspect]
    parse_success: bool
    raw_output: str | None = None
    error: str | None = None