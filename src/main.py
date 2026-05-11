from __future__ import annotations

import logging
from functools import lru_cache
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from src.chat.memory import ConversationMemory
from src.chat.service import ChatService
from src.config import settings
from src.recommender.engine import RecommendationEngine
from src.schemas import (
    ChatRecommendRequest,
    ChatRecommendResponse,
    HealthResponse,
    SessionResponse,
)


logger = logging.getLogger(__name__)

app = FastAPI(title="2-hong-community-chatbot")
memory = ConversationMemory(
    max_turns=settings.max_history_turns,
    ttl_seconds=settings.conversation_ttl_seconds,
)
chat_service = ChatService(api_key=settings.gemini_api_key, model=settings.gemini_model)


@lru_cache(maxsize=1)
def get_recommender() -> RecommendationEngine:
    return RecommendationEngine(index_dir=settings.processed_data_dir)


def recommender_available() -> bool:
    try:
        get_recommender()
        return True
    except Exception:
        return False


@app.post("/chat/session", response_model=SessionResponse)
def create_session() -> SessionResponse:
    return SessionResponse(conversation_id=str(uuid4()))


@app.delete("/chat/session/{conversation_id}")
def delete_session(conversation_id: str) -> dict[str, str]:
    memory.delete(conversation_id)
    return {"status": "deleted"}


@app.post("/chat/recommend", response_model=ChatRecommendResponse)
def recommend_chat(request: ChatRecommendRequest) -> ChatRecommendResponse:
    conversation_id = request.conversation_id or str(uuid4())
    history = memory.get(conversation_id)

    parsed = chat_service.preprocess(
        message=request.message,
        conversation_history=history,
    )
    logger.info(
        "preprocess conversation_id=%s message=%s parsed=%s",
        conversation_id,
        request.message,
        parsed,
    )

    if not parsed.get("is_recommendation_request"):
        answer = "죄송하지만 저는 식당 추천에 대해서만 도와드릴 수 있습니다."
        memory.append(conversation_id, request.message, answer, [], parsed)
        return ChatRecommendResponse(
            conversation_id=conversation_id,
            answer=answer,
            recommendations=[],
        )

    try:
        recommender = get_recommender()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Recommendation index is not ready. Run "
                "`python -m src.recommender.indexer` first."
            ),
        ) from exc

    query = str(parsed.get("normalized_query") or request.message)
    recommendations = recommender.recommend(query=query, top_k=request.top_k)
    logger.info(
        "recommend conversation_id=%s query=%s result_count=%s",
        conversation_id,
        query,
        len(recommendations),
    )

    if not recommendations:
        answer = "조건에 맞는 식당을 찾지 못했습니다. 지역이나 음식 종류를 조금 넓혀서 다시 입력해 주세요."
        memory.append(conversation_id, request.message, answer, [], parsed)
        return ChatRecommendResponse(
            conversation_id=conversation_id,
            answer=answer,
            recommendations=[],
        )

    answer = chat_service.generate_answer(request.message, recommendations)
    memory.append(conversation_id, request.message, answer, recommendations, parsed)

    return ChatRecommendResponse(
        conversation_id=conversation_id,
        answer=answer,
        recommendations=[
            {
                "shop_id": item["shop_id"],
                "shop_name": item["shop_name"],
                "address": item.get("address"),
                "categories": item.get("categories"),
                "menus": item.get("menus"),
                "awards": item.get("awards"),
                "reason": item.get("rank_reason", ""),
                "score": item["score"],
                "semantic_score": item["semantic_score"],
                "behavior_score": item["behavior_score"],
            }
            for item in recommendations
        ],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", recommender_loaded=recommender_available())
