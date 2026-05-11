from __future__ import annotations

import json
import re
from typing import Any

from src.chat.prompts import ANSWER_PROMPT, PREPROCESS_PROMPT


RECOMMENDATION_HINTS = {
    "맛집",
    "식당",
    "밥",
    "음식",
    "추천",
    "파스타",
    "한식",
    "중식",
    "일식",
    "양식",
    "오마카세",
    "라면",
    "데이트",
    "회식",
    "카페",
    "술",
    "예약",
}

LOCATION_HINTS = [
    "강남",
    "성수",
    "압구정",
    "선릉",
    "양재",
    "매봉",
    "홍대",
    "잠실",
    "역삼",
    "신사",
]

FOOD_HINTS = [
    "파스타",
    "한식",
    "중식",
    "일식",
    "양식",
    "라면",
    "오마카세",
    "고기",
    "회",
    "초밥",
    "피자",
    "카페",
]

OCCASION_HINTS = ["데이트", "회식", "가족", "조용", "혼밥", "모임", "기념일"]


class ChatService:
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.client = None
        if api_key:
            try:
                from google import genai

                self.client = genai.Client(api_key=api_key)
            except Exception:
                self.client = None

    def preprocess(self, message: str, conversation_history: list[dict[str, Any]]) -> dict[str, Any]:
        if self.client is not None:
            try:
                return self._preprocess_with_gemini(message, conversation_history)
            except Exception:
                pass
        return heuristic_preprocess(message, conversation_history)

    def generate_answer(self, user_message: str, recommendations: list[dict[str, Any]]) -> str:
        if self.client is not None:
            try:
                return self._answer_with_gemini(user_message, recommendations)
            except Exception:
                pass
        return template_answer(user_message, recommendations)

    def _preprocess_with_gemini(
        self,
        message: str,
        conversation_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = PREPROCESS_PROMPT.format(
            conversation_history=json.dumps(conversation_history, ensure_ascii=False),
            message=message,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        return json.loads(response.text)

    def _answer_with_gemini(self, user_message: str, recommendations: list[dict[str, Any]]) -> str:
        prompt = ANSWER_PROMPT.format(
            user_message=user_message,
            recommendations_json=json.dumps(recommendations, ensure_ascii=False),
        )
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return response.text


def heuristic_preprocess(message: str, conversation_history: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", message).strip()
    lower = normalized.lower()
    is_follow_up = bool(conversation_history) and any(
        keyword in normalized for keyword in ["더", "두 번째", "첫 번째", "자세히", "조용", "가격"]
    )
    is_recommendation = is_follow_up or any(keyword in lower for keyword in RECOMMENDATION_HINTS)

    previous_query = _last_normalized_query(conversation_history)
    if is_follow_up and previous_query:
        normalized_query = f"{previous_query} {normalized}"
    else:
        normalized_query = normalized

    return {
        "is_recommendation_request": is_recommendation,
        "normalized_query": normalized_query,
        "location": _first_match(normalized_query, LOCATION_HINTS),
        "food_or_category": _first_match(normalized_query, FOOD_HINTS),
        "occasion": _first_match(normalized_query, OCCASION_HINTS),
        "constraints": {
            "quiet": "조용" in normalized_query,
            "price_level": "낮은" if "가격대 낮" in normalized_query else None,
            "party_size": None,
        },
        "follow_up_target_rank": _extract_rank(normalized),
    }


def template_answer(user_message: str, recommendations: list[dict[str, Any]]) -> str:
    if not recommendations:
        return "조건에 맞는 식당을 찾지 못했습니다. 지역이나 음식 종류를 조금 넓혀서 다시 입력해 주세요."

    lines = [f"'{user_message}' 조건으로 추천드릴게요."]
    for idx, item in enumerate(recommendations, start=1):
        parts = [f"{idx}. {item.get('shop_name')}"]
        if item.get("categories"):
            parts.append(f"분류: {item['categories']}")
        if item.get("address"):
            parts.append(f"주소: {item['address']}")
        reason = item.get("rank_reason") or item.get("reason")
        if reason:
            parts.append(f"추천 이유: {reason}")
        lines.append(" / ".join(parts))
    return "\n".join(lines)


def _first_match(text: str, candidates: list[str]) -> str | None:
    return next((candidate for candidate in candidates if candidate in text), None)


def _last_normalized_query(conversation_history: list[dict[str, Any]]) -> str | None:
    for turn in reversed(conversation_history):
        parsed = turn.get("parsed_query") or {}
        query = parsed.get("normalized_query")
        if query:
            return str(query)
    return None


def _extract_rank(message: str) -> int | None:
    mapping = {"첫": 1, "두": 2, "세": 3, "네": 4, "다섯": 5}
    for prefix, rank in mapping.items():
        if f"{prefix} 번째" in message or f"{prefix}번째" in message:
            return rank
    match = re.search(r"(\d+)\s*번째", message)
    return int(match.group(1)) if match else None
