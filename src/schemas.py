from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRecommendRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class RecommendationItem(BaseModel):
    shop_id: str
    shop_name: str
    address: str | None = None
    categories: str | None = None
    menus: str | None = None
    awards: str | None = None
    reason: str
    score: float
    semantic_score: float
    behavior_score: float


class ChatRecommendResponse(BaseModel):
    conversation_id: str
    answer: str
    recommendations: list[RecommendationItem]


class SessionResponse(BaseModel):
    conversation_id: str


class HealthResponse(BaseModel):
    status: str
    recommender_loaded: bool
