from __future__ import annotations

import logging
import sys
from time import perf_counter
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
LOG_DIR = PROJECT_ROOT / "logs"


def configure_recommendation_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)03d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app_logger = logging.getLogger("src")
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False

    if not any(getattr(handler, "name", "") == "recommendation_console" for handler in app_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.name = "recommendation_console"
        console_handler.setFormatter(formatter)
        app_logger.addHandler(console_handler)

    if not any(getattr(handler, "name", "") == "recommendation_file" for handler in app_logger.handlers):
        file_handler = logging.FileHandler(LOG_DIR / "recommendation.log", encoding="utf-8")
        file_handler.name = "recommendation_file"
        file_handler.setFormatter(formatter)
        app_logger.addHandler(file_handler)


configure_recommendation_logging()

app = FastAPI(title="2-hong-community-chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
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
    started_at = perf_counter()
    conversation_id = request.conversation_id or str(uuid4())
    history = memory.get(conversation_id)
    logger.info("You: %s", request.message)

    parsed = chat_service.preprocess(
        message=request.message,
        conversation_history=history,
    )
    logger.info(
        "[Step 1] location=%s, categories=%s, situations=%s, normalized_query=%s",
        parsed.get("location"),
        _as_log_list(parsed.get("food_or_category")),
        _as_log_list(parsed.get("occasion")),
        parsed.get("normalized_query"),
    )

    if not parsed.get("is_recommendation_request"):
        answer = "죄송하지만 저는 식당 추천에 대해서만 도와드릴 수 있습니다."
        memory.append(conversation_id, request.message, answer, [], parsed)
        logger.info("[Step 2] 추천 요청 아님 - conversation_id=%s", conversation_id)
        logger.info("Bot: %s", answer)
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
    try:
        recommendations, diagnostics = recommender.recommend_with_diagnostics(
            query=query,
            top_k=request.top_k,
        )
    except ValueError as exc:
        logger.exception("[Error] recommendation failed conversation_id=%s query=%s", conversation_id, query)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    logger.info(
        "[Step 2] 후보 %s개 로드 (인덱스 backend=%s, embedding_dim=%s)",
        diagnostics.get("candidate_count"),
        diagnostics.get("embedding_backend"),
        diagnostics.get("embedding_dimension"),
    )
    logger.info(
        "[Step 3] 행동 점수 매칭 %s개, 전역 행동 점수 %s개, 가중치 semantic=%.2f behavior=%.2f",
        diagnostics.get("matched_behavior_count", 0),
        diagnostics.get("global_behavior_count", 0),
        diagnostics.get("alpha", 0.0),
        diagnostics.get("beta", 0.0),
    )
    logger.info(
        "[Step 4] 하이브리드 점수 계산 완료 - score min=%.4f, max=%.4f, mean=%.4f",
        diagnostics.get("score_min", 0.0),
        diagnostics.get("score_max", 0.0),
        diagnostics.get("score_mean", 0.0),
    )

    if not recommendations:
        answer = "조건에 맞는 식당을 찾지 못했습니다. 지역이나 음식 종류를 조금 넓혀서 다시 입력해 주세요."
        memory.append(conversation_id, request.message, answer, [], parsed)
        logger.info("[Step 5] 상위 0개 선정")
        logger.info("Bot: %s", answer)
        return ChatRecommendResponse(
            conversation_id=conversation_id,
            answer=answer,
            recommendations=[],
        )

    logger.info("[Step 5] 상위 %s개 선정:", len(recommendations))
    for idx, item in enumerate(recommendations, start=1):
        logger.info(
            "    %s. %s (score=%.4f, semantic=%.4f, behavior=%.4f)",
            idx,
            item.get("shop_name"),
            item.get("score", 0.0),
            item.get("semantic_score", 0.0),
            item.get("behavior_score", 0.0),
        )

    answer = chat_service.generate_answer(request.message, recommendations)
    memory.append(conversation_id, request.message, answer, recommendations, parsed)
    logger.info(
        "Bot: %s (elapsed=%.3fs, conversation_id=%s)",
        _compact_log_text(answer),
        perf_counter() - started_at,
        conversation_id,
    )

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


def _as_log_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _compact_log_text(value: str, limit: int = 220) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
