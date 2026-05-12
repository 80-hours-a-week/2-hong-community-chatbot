# 2-hong-community-chatbot - 상세 기능 명세서

커뮤니티 서비스 확장 채팅 봇 및 백엔드 구현 상세 가이드

## 1. 프로젝트 개요
- **목표**: 유저의 식당 추천 요청에 대해 사용자 행동 로그 기반으로 식당을 추천하는 챗봇 구현
- **기술 스택**: pytorch, Langchain, MCP, Vector Database, FastAPI
- **주요 기능**:
    - 추천 시스템: 유저의 검색어(Search_query)와 행동 로그(User_behavior_log)를 기반으로 식당을 추천하는 챗봇 구현
    - 자연어 응답 LLM 챗봇: 추천된 식당에 대한 자연어 응답 생성
    - 프론트엔드 통합: 2-hong-community-fe 프론트엔드와의 API 연동 및 실시간 채팅 기능 구현
- **핵심 제약사항**:
    - 웹 검색을 통한 식당 추천은 사용하지 않는다.
    - LLM이 자체 지식만으로 식당을 임의 추천하지 않도록 한다.
    - 추천 결과는 `shops.csv`에 존재하는 식당으로 제한한다.
    - 추천 모델이 챗봇 응답의 병목이 되지 않도록 추론 속도를 고려한다.

## 2. 커뮤니티 채팅 봇 UI/UX 설계
- **채팅 진입점**: 기존 커뮤니티 프론트엔드 내에서 식당 추천 챗봇을 실행할 수 있는 페이지 제공
- **사용자 입력**:
    - 사용자는 자연어로 식당 추천 조건을 입력한다.
    - 예시: `압구정 파스타 데이트`, `성수 데이트`, `라면 맛집`, `강남 회식하기 좋은 한식`
- **챗봇 응답**:
    - 추천 식당 목록과 추천 이유를 함께 제공한다.
    - 응답에는 식당명, 주소, 카테고리, 주요 메뉴 또는 시설, 추천 근거를 포함한다.
    - 추천 근거는 사용자 검색 의도와 행동 로그 기반 랭킹 결과를 중심으로 작성한다.
- **다회성 연속 채팅 지원**:
    - 사용자가 이전 추천 결과를 바탕으로 `더 조용한 곳`, `가격대 낮은 곳`, `두 번째 식당 자세히 알려줘`처럼 후속 질문을 입력할 수 있도록 한다.
    - 백엔드는 동일 대화의 이전 사용자 메시지, 추천 조건, 추천 결과를 참고해 후속 응답을 생성한다.
    - 외부 DB 없이 구현하는 경우 서버 메모리 또는 프론트엔드에서 전달하는 `conversation_id` 기준으로 대화 이력을 임시 관리한다.
    - 채팅 종료 시 대화 이력을 삭제하거나 일정 시간이 지나면 자동으로 삭제한다.
- **상호작용 흐름**:
    - 사용자가 검색어를 입력하면 프론트엔드가 백엔드 추천 챗봇 API를 호출한다.
    - 백엔드는 추천 모델 결과와 LLM 자연어 응답을 조합해 반환한다.
    - 프론트엔드는 추천 결과를 채팅 메시지 형태로 표시한다.

## 3. 추천 시스템 구현
- **추천 시스템 구현**: 유저의 검색어와 전체 사용자 행동 로그 집계 통계를 함께 활용하는 하이브리드 추천 모델 개발
- **입력 데이터**:
    - `logs.csv`: `event_type`, `event_timestamp`, `user_id`, `session_id`, `shop_id`, `search_query`, `position`
    - `shops.csv`: `shop_id`, `shop_name`, `address`, `menus`, `categories`, `facilities`, `awards`
- **권장 구현 파일 구조**:

    ```text
    src/
      main.py                         # FastAPI app 진입점
      schemas.py                      # API request/response Pydantic schema
      chat/
        service.py                    # LLM 1차/2차 호출과 추천 모델 orchestration
        memory.py                     # conversation_id 기반 임시 대화 이력 관리
        prompts.py                    # Gemini 프롬프트 템플릿
      recommender/
        session.py                    # sub-session 분리 및 search_query 전파
        query_parser.py               # 식당 문서 생성, 임베딩 모델 로딩
        indexer.py                    # 오프라인 인덱스 생성 스크립트
        ranker.py                     # 행동 점수, semantic 점수, hybrid 점수 계산
        engine.py                     # 온라인 추천 엔진
      evaluation/
        metrics.py                    # Hit@K, MRR, NDCG, Coverage 계산
        run_eval.py                   # hold-out/baseline 비교 실행
    data/
      raw/
        logs.csv
        shops.csv
      processed/
        behavior_scores.pkl
        shop_embeddings.npy
        shops_indexed.pkl
    ```
- **추천 알고리즘 개요**:
    - 최종 추천은 `검색어-식당 의미 유사도`와 `(search_query, shop_id) 단위 행동 로그 점수`를 결합해 산출한다.
    - 검색어가 기존 행동 로그에 충분히 존재하는 경우 행동 로그의 실제 선택 신호를 더 크게 반영한다.
    - 행동 로그가 희소하거나 처음 들어온 검색어는 임베딩 기반 의미 유사도를 중심으로 추천한다.
    - 현재 접속 사용자의 개인 이력은 반영하지 않고, 전체 사용자의 집계 행동 통계를 사용한다. 개인화 추천은 향후 확장 과제로 둔다.
- **세션화 및 검색어 전파 알고리즘**:
    - `bookmark`, `reservation` 등 일부 이벤트에는 `search_query`가 없으므로, 행동 이벤트를 어떤 검색 의도와 연결할지 먼저 보강해야 한다.
    - 단순히 같은 `session_id` 안에서 직전 검색어를 forward-fill하면 하나의 세션 안에 여러 검색 의도가 섞인 경우 오탐이 발생할 수 있다.
    - 이를 방지하기 위해 `session_id`를 다시 `sub_session`으로 분리한 뒤, 각 `sub_session` 내부에서만 검색어를 전파한다.
    - `sub_session` 경계 기준:
        - 동일 `session_id` 안에서 non-null `search_query`가 직전 non-null `search_query`와 달라지는 경우
        - 연속 이벤트 간 시간 차이가 30분을 초과하는 경우
    - 예시:
        - `압구정 파스타` 검색 후 발생한 bookmark는 `압구정 파스타` 의도로 연결한다.
        - 같은 `session_id` 안에서 이후 `강남 한식` 검색이 발생하면 새 `sub_session`으로 분리하고, 그 뒤 reservation은 `강남 한식` 의도로 연결한다.
    - 구현 대상 파일: `src/recommender/session.py`
    - 예시 구현:

    ```python
    import pandas as pd


    def split_sub_sessions(logs: pd.DataFrame, time_gap_minutes: int = 30) -> pd.DataFrame:
        required = {"session_id", "event_timestamp", "search_query"}
        missing = required - set(logs.columns)
        if missing:
            raise ValueError(f"logs.csv missing columns: {sorted(missing)}")

        df = logs.copy()
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
        df = df.sort_values(["session_id", "event_timestamp"]).reset_index(drop=True)

        df["non_null_query"] = df["search_query"].where(df["search_query"].notna())
        filled_query = df.groupby("session_id")["non_null_query"].ffill()
        df["prev_non_null_query"] = filled_query.groupby(df["session_id"]).shift(1)
        df["time_diff"] = df.groupby("session_id")["event_timestamp"].diff()

        query_changed = (
            df["search_query"].notna()
            & df["prev_non_null_query"].notna()
            & (df["search_query"] != df["prev_non_null_query"])
        )
        time_gap = df["time_diff"] > pd.Timedelta(minutes=time_gap_minutes)
        first_in_session = df.groupby("session_id").cumcount() == 0

        df["is_boundary"] = first_in_session | query_changed | time_gap
        df["sub_session_no"] = df.groupby("session_id")["is_boundary"].cumsum() - 1
        df["sub_session_id"] = (
            df["session_id"].astype(str) + "_" + df["sub_session_no"].astype(str)
        )

        df["search_query"] = df.groupby("sub_session_id")["search_query"].ffill()
        return df.drop(columns=["non_null_query", "prev_non_null_query", "time_diff"])
    ```
- **쿼리 임베딩 기반 의미 유사도 계산**:
    - 한국어 문장 임베딩 모델 `jhgan/ko-sroberta-multitask`를 사용해 사용자 검색어와 식당 문서 벡터 간 코사인 유사도를 계산한다.
    - 이 단계는 지역/음식/상황을 규칙 기반 슬롯으로만 분해하는 방식이 아니라, 검색어 전체 의미를 벡터 공간에서 비교하는 end-to-end 임베딩 유사도 방식이다.
    - 식당 임베딩 문서는 `shop_name + categories + awards`를 기준으로 구성한다.
    - `menus`는 기본 임베딩 문서에서 제외한다.
        - 이유: `라면 맛집` 검색 시 오마카세 식당의 후식 라면 메뉴가 과도하게 매칭되는 문제를 줄이기 위함이다.
        - 메뉴 정보는 자연어 응답 생성 시 설명 근거로 사용할 수 있지만, 핵심 후보 생성 임베딩에는 직접 넣지 않는다.
    - `awards`는 `미슐랭`, `수상 경력`, `인증된 맛집`처럼 품질 지향 검색어에 대응하기 위해 포함한다.
    - 구현 대상 파일: `src/recommender/query_parser.py`
    - 예시 구현:

    ```python
    import pandas as pd
    from sentence_transformers import SentenceTransformer


    MODEL_NAME = "jhgan/ko-sroberta-multitask"


    def build_shop_document(row: pd.Series) -> str:
        fields = [
            str(row.get("shop_name", "")),
            str(row.get("categories", "")),
            str(row.get("awards", "")),
        ]
        return " ".join(value for value in fields if value and value != "nan").strip()


    def load_embedding_model() -> SentenceTransformer:
        return SentenceTransformer(MODEL_NAME)


    def encode_shops(shops: pd.DataFrame, model: SentenceTransformer):
        docs = shops.apply(build_shop_document, axis=1).tolist()
        return model.encode(docs, normalize_embeddings=True, batch_size=64)


    def encode_query(query: str, model: SentenceTransformer):
        return model.encode([query], normalize_embeddings=True)[0]
    ```
- **행동 로그 기반 집계 가중 랭킹**:
    - 세션화와 검색어 전파가 완료된 로그를 `(search_query, shop_id)` 단위로 집계한다.
    - 이벤트별 가중치는 사용자의 전환 의도 강도에 따라 차등 부여한다.

    | 이벤트 | 가중치 | 의미 |
    | --- | ---: | --- |
    | `impression` | 0.1 | 단순 노출, 가장 약한 신호 |
    | `click` | 1.0 | 목록에서 관심을 보인 신호 |
    | `view` | 2.0 | 상세 정보를 확인한 신호 |
    | `bookmark` | 5.0 | 저장할 만큼 강한 선호 신호 |
    | `reservation` | 10.0 | 실제 전환에 해당하는 최강 신호 |

    - 행동 점수는 쿼리별로 min-max 정규화해 임베딩 유사도와 결합 가능한 범위로 맞춘다.
    - `position` 컬럼은 노출 순위 편향 보정에 사용할 수 있으나, 1차 구현 범위에서는 미적용한다.
        - 이유: 식당 500개 규모와 6일 구현 일정에서 position bias 보정보다 세션화, 임베딩, 행동 점수 결합의 완성도가 더 중요하다.
        - 향후 확장 시 `click_weight * 1 / log(1 + position)` 형태로 상위 노출 편향을 완화할 수 있다.
    - 구현 대상 파일: `src/recommender/ranker.py`
    - 예시 구현:

    ```python
    import pandas as pd


    EVENT_WEIGHTS = {
        "impression": 0.1,
        "click": 1.0,
        "view": 2.0,
        "bookmark": 5.0,
        "reservation": 10.0,
    }


    def build_behavior_scores(logs: pd.DataFrame) -> dict[str, dict[str, float]]:
        df = logs.dropna(subset=["search_query", "shop_id"]).copy()
        df["event_weight"] = df["event_type"].map(EVENT_WEIGHTS).fillna(0.0)

        grouped = (
            df.groupby(["search_query", "shop_id"], as_index=False)["event_weight"]
            .sum()
            .rename(columns={"event_weight": "raw_behavior_score"})
        )

        def normalize(group: pd.DataFrame) -> pd.DataFrame:
            min_v = group["raw_behavior_score"].min()
            max_v = group["raw_behavior_score"].max()
            if max_v == min_v:
                group["behavior_score"] = 1.0
            else:
                group["behavior_score"] = (
                    group["raw_behavior_score"] - min_v
                ) / (max_v - min_v)
            return group

        normalized = grouped.groupby("search_query", group_keys=False).apply(normalize)
        return {
            query: dict(zip(items["shop_id"].astype(str), items["behavior_score"]))
            for query, items in normalized.groupby("search_query")
        }
    ```
- **최종 하이브리드 스코어 계산**:
    - 기본 스코어 식:

    ```text
    score(query, shop) = alpha * semantic_sim(query, shop) + beta * behavior_score_norm(query, shop)
    ```

    - 행동 데이터가 충분한 쿼리:
        - `alpha = 0.4`, `beta = 0.6`
        - 실제 사용자 선택 패턴을 더 크게 반영한다.
    - 행동 데이터가 희소한 쿼리:
        - `alpha = 0.6`, `beta = 0.4`
        - 행동 로그 과적합을 줄이고 검색어 의미를 더 크게 반영한다.
    - 새로운 쿼리 또는 로그가 없는 쿼리:
        - `alpha = 1.0`, `beta = 0.0`
        - 의미 유사도만으로 cold-start 추천을 수행한다.
    - 행동 데이터 희소 기준:
        - 해당 검색어와 매칭되는 행동 로그 기반 후보 식당 수가 5개 미만인 경우
    - 동일 점수 또는 근소한 점수 차이가 있는 경우에는 `reservation`, `bookmark`, `view` 등 강한 이벤트 비중이 높은 식당을 우선한다.
    - 예시 구현:

    ```python
    import numpy as np


    def select_weights(matched_behavior_count: int) -> tuple[float, float]:
        if matched_behavior_count == 0:
            return 1.0, 0.0
        if matched_behavior_count < 5:
            return 0.6, 0.4
        return 0.4, 0.6


    def rank_candidates(
        query_embedding: np.ndarray,
        shop_embeddings: np.ndarray,
        shop_ids: list[str],
        behavior_by_shop: dict[str, float],
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[dict]:
        semantic_scores = shop_embeddings @ query_embedding
        alpha, beta = select_weights(len(behavior_by_shop))

        rows = []
        for idx, shop_id in enumerate(shop_ids):
            semantic = float(semantic_scores[idx])
            behavior = float(behavior_by_shop.get(str(shop_id), 0.0))
            if mode == "semantic_only":
                score = semantic
            elif mode == "behavior_only":
                score = behavior
            else:
                score = alpha * semantic + beta * behavior
            rows.append(
                {
                    "shop_id": str(shop_id),
                    "score": score,
                    "semantic_score": semantic,
                    "behavior_score": behavior,
                }
            )

        return sorted(rows, key=lambda row: row["score"], reverse=True)[:top_k]
    ```
- **오프라인 인덱스화 및 온라인 추론 구조**:
    - 추천 응답 속도를 위해 무거운 계산은 오프라인 인덱싱 단계에서 미리 수행한다.
    - 오프라인 인덱싱 단계:
        - 전체 로그를 세션화하고 `sub_session` 단위로 검색어를 전파한다.
        - `(search_query, shop_id)`별 행동 가중 점수를 집계해 `data/behavior_scores.pkl`로 저장한다.
        - 전체 식당 문서 임베딩을 계산해 `data/shop_embeddings.npy`로 저장한다.
        - 식당 메타데이터와 인덱스 매핑을 `data/shops_indexed.pkl`로 저장한다.
    - 온라인 추론 단계:
        - 사용자 입력 검색어를 1회 임베딩한다.
        - 전체 식당 임베딩과 numpy 기반 cosine similarity를 계산한다.
        - 행동 점수 딕셔너리에서 해당 검색어의 식당별 점수를 조회한다.
        - 하이브리드 스코어로 정렬해 상위 3~5개 식당을 반환한다.
    - 구현 대상 파일: `src/recommender/indexer.py`
    - 오프라인 인덱서 예시:

    ```python
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from recommender.query_parser import encode_shops, load_embedding_model
    from recommender.ranker import build_behavior_scores
    from recommender.session import split_sub_sessions


    def build_index(raw_dir: str = "data/raw", out_dir: str = "data/processed") -> None:
        raw_path = Path(raw_dir)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        logs = pd.read_csv(raw_path / "logs.csv")
        shops = pd.read_csv(raw_path / "shops.csv")

        logs = split_sub_sessions(logs)
        behavior_scores = build_behavior_scores(logs)

        model = load_embedding_model()
        shop_embeddings = encode_shops(shops, model)

        np.save(out_path / "shop_embeddings.npy", shop_embeddings)
        shops.to_pickle(out_path / "shops_indexed.pkl")
        with open(out_path / "behavior_scores.pkl", "wb") as f:
            pickle.dump(behavior_scores, f)
    ```
    - 온라인 추천 엔진 예시:

    ```python
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from recommender.query_parser import encode_query, load_embedding_model
    from recommender.ranker import rank_candidates


    class RecommendationEngine:
        def __init__(self, index_dir: str = "data/processed"):
            index_path = Path(index_dir)
            self.model = load_embedding_model()
            self.shop_embeddings = np.load(index_path / "shop_embeddings.npy")
            self.shops = pd.read_pickle(index_path / "shops_indexed.pkl")
            with open(index_path / "behavior_scores.pkl", "rb") as f:
                self.behavior_scores = pickle.load(f)
            self.shop_ids = self.shops["shop_id"].astype(str).tolist()

        def recommend(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[dict]:
            query_embedding = encode_query(query, self.model)
            behavior_by_shop = self.behavior_scores.get(query, {})
            ranked = rank_candidates(
                query_embedding=query_embedding,
                shop_embeddings=self.shop_embeddings,
                shop_ids=self.shop_ids,
                behavior_by_shop=behavior_by_shop,
                top_k=top_k,
                mode=mode,
            )
            shops_by_id = self.shops.assign(shop_id=self.shops["shop_id"].astype(str))
            shops_by_id = shops_by_id.set_index("shop_id")

            results = []
            for row in ranked:
                meta = shops_by_id.loc[row["shop_id"]].to_dict()
                results.append(
                    {
                        **row,
                        "shop_name": meta.get("shop_name"),
                        "address": meta.get("address"),
                        "categories": meta.get("categories"),
                        "menus": meta.get("menus"),
                        "awards": meta.get("awards"),
                        "rank_reason": make_rank_reason(row, meta),
                    }
                )
            return results


    def make_rank_reason(score_row: dict, meta: dict) -> str:
        if score_row["behavior_score"] >= 0.7:
            return "동일 검색어에서 예약, 북마크, 상세 조회 행동 점수가 높습니다."
        if score_row["semantic_score"] >= 0.7:
            return "검색어와 식당 카테고리 및 수상 정보의 의미 유사도가 높습니다."
        return "검색어 의미와 전체 행동 로그 점수를 종합해 상위로 선정되었습니다."
    ```
- **오탐 방지 로직**:
    - `라면 맛집`처럼 의도가 명확한 검색어는 후식 메뉴로 라면이 포함된 고가 오마카세 식당이 과도하게 상위 노출되지 않도록 한다.
    - 이를 위해 메뉴 정보는 기본 임베딩 문서에서 제외하고, 카테고리와 수상 정보 중심으로 의미 유사도를 계산한다.
    - 특정 검색어에서 행동 로그가 충분히 누적된 경우에는 실제 예약, 북마크, 상세 조회 이력이 있는 식당을 상위에 배치한다.
    - 의미 유사도는 높지만 행동 로그가 거의 없고 카테고리도 맞지 않는 식당은 후순위로 밀리도록 한다.
- **추천 결과 출력**:
    - 추천 모델은 상위 N개 식당과 각 식당의 점수, 주요 추천 근거를 반환한다.
    - 반환 결과는 LLM이 자연어 응답을 만들 때 사용할 수 있는 구조화된 형태로 제공한다.
    - 추천 결과에는 최소한 `shop_id`, `shop_name`, `address`, `categories`, `score`, `semantic_score`, `behavior_score`, `rank_reason`을 포함한다.
    - `rank_reason`은 `검색어 의미 유사도 높음`, `동일 검색어에서 예약/북마크 비중 높음`, `수상 정보가 품질 지향 검색어와 일치`처럼 데이터에 근거한 설명으로 제한한다.
- **디버깅용 로그 기록**:
    - 추천 시스템 작동 시 개발 및 검증을 위한 디버깅 로그를 기록한다.
    - 로그에는 요청 식별자, `conversation_id`, 원본 사용자 메시지, Gemini 1차 호출의 전처리 결과, 추출된 지역/음식/상황 키워드를 포함한다.
    - 추천 후보 생성 단계에서는 후보 식당 수, 필터링된 식당 수, 필터링 사유, 랭킹에 사용된 주요 점수 요소를 기록한다.
    - 최종 추천 결과 단계에서는 상위 N개 식당의 `shop_id`, 점수, 순위, 추천 근거를 기록한다.
    - 로그는 추천 품질 분석과 오류 재현을 위한 용도로 사용하며, 운영 환경에서는 필요 시 로그 레벨을 조정해 과도한 로그 생성을 방지한다.

## 4. 자연어 응답 LLM 챗봇 구현
- **LLM 챗봇 구현**: Gemini API(Gemini 2.5 Flash)와 Langchain을 활용하여 추천된 식당에 대한 자연어 응답 생성 및 다회성 연속 대화 지원
    - Gemini API Key는 .env에 Github Secret로 입력
- **Gemini API 사용 지점**:
    - Gemini API는 한 번의 사용자 요청 처리 과정에서 2번 사용한다.
    - 1차 호출: 유저 요청이 식당 추천 관련 요청인지 필터링하고, 추천 시스템에 투입할 수 있도록 검색 의도와 조건을 전처리한다.
    - 2차 호출: 추천 시스템 출력 결과와 식당 메타데이터를 바탕으로 유저에게 보여줄 자연어 응답을 생성한다.
- **LLM 역할**:
    - 사용자 입력에서 추천 조건을 정리하고, 식당 추천과 무관한 요청을 사전에 분리한다.
    - 추천 시스템 입력에 필요한 지역, 음식/카테고리, 상황/목적, 후속 질문 맥락을 구조화한다.
    - 추천 모델이 반환한 식당 목록을 자연스러운 대화형 문장으로 설명한다.
    - 추천 이유는 추천 모델 결과와 `shops.csv`의 식당 정보만 사용한다.
    - 과거의 질문-응답을 참고하여 이후의 일관된 대화 흐름을 유지한다.
- **금지 사항**:
    - LLM이 웹 검색 결과를 사용해 새로운 식당을 추가하지 않는다.
    - 추천 모델 결과에 없는 식당을 임의로 생성하지 않는다.
    - 데이터에 없는 주소, 메뉴, 수상 정보 등을 추측해 작성하지 않는다.
    - 식당 추천 이외의 주제 요청은 "죄송하지만 저는 식당 추천에 대해서만 도와드릴 수 있습니다."와 같이 응답한다.
- **프롬프트 구성**:
    - 1차 호출 프롬프트에는 식당 추천 요청 여부 판단 기준과 추천 시스템 입력 스키마를 명시한다.
    - 2차 호출 프롬프트에는 데이터 기반 추천 제약사항을 명시한다.
    - 사용자 검색어, 추천 모델 결과, 식당 메타데이터를 컨텍스트로 전달한다.
    - 컨텍스트에 없는 정보는 모른다고 답하거나 생략하도록 지시한다.
- **1차 LLM 전처리 출력 스키마**:
    - Gemini 1차 호출은 자유 문장이 아니라 아래 JSON 구조로만 응답하도록 제한한다.

    ```json
    {
      "is_recommendation_request": true,
      "normalized_query": "압구정 파스타 데이트",
      "location": "압구정",
      "food_or_category": "파스타",
      "occasion": "데이트",
      "constraints": {
        "quiet": null,
        "price_level": null,
        "party_size": null
      },
      "follow_up_target_rank": null
    }
    ```

    - `is_recommendation_request=false`이면 추천 엔진을 호출하지 않고 제한 안내 답변을 반환한다.
    - `normalized_query`는 추천 엔진에 직접 전달하는 최종 검색어이다.
    - 후속 질문이면 이전 대화의 조건과 새 조건을 합쳐 `normalized_query`를 재구성한다.
- **프롬프트 템플릿 예시**:

    ```python
    PREPROCESS_PROMPT = """
    너는 식당 추천 챗봇의 입력 전처리기다.
    사용자의 메시지가 식당 추천과 관련 있는지 판단하고, 반드시 JSON만 출력한다.

    규칙:
    - 식당 추천, 식당 상세 질문, 이전 추천 결과에 대한 후속 질문이면 is_recommendation_request=true
    - 식당 추천과 무관한 요청이면 is_recommendation_request=false
    - normalized_query는 추천 엔진에 넣을 짧은 한국어 검색어로 만든다.
    - 모르는 값은 null로 둔다.

    이전 대화:
    {conversation_history}

    사용자 메시지:
    {message}
    """

    ANSWER_PROMPT = """
    너는 식당 추천 결과를 사용자에게 설명하는 챗봇이다.
    아래 recommendations 배열에 포함된 식당만 답변에 사용한다.
    웹 검색, 일반 지식, 추측으로 식당을 추가하지 않는다.
    주소, 메뉴, 수상 정보가 비어 있으면 언급하지 않는다.

    사용자 요청:
    {user_message}

    추천 결과 JSON:
    {recommendations_json}

    답변 형식:
    - 추천 식당 3~5개
    - 각 식당마다 추천 이유 1~2문장
    - 필요한 경우 조건을 좁히는 후속 질문 1문장
    """
    ```
- **LLM 호출 서비스 예시**:

    ```python
    import json
    from google import genai


    class GeminiChatService:
        def __init__(self, api_key: str):
            self.client = genai.Client(api_key=api_key)
            self.model = "gemini-2.5-flash"

        def preprocess(self, message: str, conversation_history: list[dict]) -> dict:
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

        def generate_answer(self, user_message: str, recommendations: list[dict]) -> str:
            prompt = ANSWER_PROMPT.format(
                user_message=user_message,
                recommendations_json=json.dumps(recommendations, ensure_ascii=False),
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return response.text
    ```
- **응답 형식**:
    - 추천 식당 3~5개를 기본으로 제공한다.
    - 각 식당별로 추천 이유를 1~2문장으로 제공한다.
    - 필요한 경우 사용자가 조건을 좁힐 수 있도록 후속 질문을 제공한다.

## 5. 채팅 봇 서비스 확장 및 백엔드 구현
- **FastAPI를 활용한 백엔드 구현**: FastAPI를 활용해 채팅 봇의 백엔드 API 구현
- **주요 API**:
    - `POST /chat/recommend`: 사용자 메시지를 받아 추천 식당과 챗봇 응답 반환
    - `POST /chat/session`: 신규 채팅 세션 또는 `conversation_id` 생성
    - `DELETE /chat/session/{conversation_id}`: 채팅 종료 시 임시 대화 이력 삭제
    - `GET /health`: 서버 상태 확인
- **요청 예시**:
```json
{
  "conversation_id": "optional_conversation_id",
  "message": "압구정 파스타 데이트",
  "top_k": 5
}
```
- **응답 예시**:
```json
{
  "conversation_id": "optional_conversation_id",
  "answer": "압구정에서 데이트하기 좋은 파스타 식당을 추천드릴게요...",
  "recommendations": [
    {
      "shop_id": "example_shop_id",
      "shop_name": "식당명",
      "address": "서울특별시 ...",
      "categories": "이탈리안,파스타",
      "reason": "검색어의 지역/음식/상황 조건과 행동 로그 점수가 높습니다.",
      "score": 0.87
    }
  ]
}
```
- **Pydantic 스키마 예시**:

```python
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
```
- **처리 흐름**:
    - 사용자 메시지 수신
    - `conversation_id` 기준으로 이전 대화 이력 조회
    - Gemini API 1차 호출로 식당 추천 관련 요청 여부를 필터링하고 추천 시스템 입력값을 전처리
    - 식당 추천과 무관한 요청이면 추천 모델을 호출하지 않고 제한 안내 응답 반환
    - 추천 시스템 호출 전 전처리 결과와 요청 식별자를 디버깅 로그로 기록
    - 전처리된 검색 의도와 조건을 기반으로 추천 모델 호출
    - 추천 후보 생성, 필터링, 랭킹 점수 계산 과정을 디버깅 로그로 기록
    - 추천 결과를 식당 메타데이터와 결합
    - Gemini API 2차 호출로 추천 결과를 자연어 응답으로 변환
    - 사용자 메시지, 추천 조건, 추천 결과, 챗봇 응답을 대화 이력에 저장
    - 프론트엔드에 구조화된 추천 결과와 답변 반환
- **FastAPI 엔드포인트 예시**:

```python
import os
from fastapi import FastAPI
from uuid import uuid4

from schemas import ChatRecommendRequest, ChatRecommendResponse
from chat.memory import ConversationMemory
from chat.service import GeminiChatService
from recommender.engine import RecommendationEngine


app = FastAPI(title="2-hong-community-chatbot")
memory = ConversationMemory(max_turns=5, ttl_seconds=3600)
llm_service = GeminiChatService(api_key=os.environ["GEMINI_API_KEY"])
recommender = RecommendationEngine(index_dir="data/processed")


@app.post("/chat/recommend", response_model=ChatRecommendResponse)
def recommend_chat(request: ChatRecommendRequest):
    conversation_id = request.conversation_id or str(uuid4())
    history = memory.get(conversation_id)

    parsed = llm_service.preprocess(
        message=request.message,
        conversation_history=history,
    )
    if not parsed.get("is_recommendation_request"):
        answer = "죄송하지만 저는 식당 추천에 대해서만 도와드릴 수 있습니다."
        memory.append(conversation_id, request.message, answer, [], parsed)
        return {
            "conversation_id": conversation_id,
            "answer": answer,
            "recommendations": [],
        }

    query = parsed["normalized_query"]
    recommendations = recommender.recommend(query=query, top_k=request.top_k)
    if not recommendations:
        answer = "조건에 맞는 식당을 찾지 못했습니다. 지역이나 음식 종류를 조금 넓혀서 다시 입력해 주세요."
        memory.append(conversation_id, request.message, answer, [], parsed)
        return {
            "conversation_id": conversation_id,
            "answer": answer,
            "recommendations": [],
        }

    answer = llm_service.generate_answer(request.message, recommendations)
    memory.append(conversation_id, request.message, answer, recommendations, parsed)

    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "recommendations": [
            {
                **item,
                "reason": item["rank_reason"],
            }
            for item in recommendations
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
```
- **대화 이력 관리**:
    - 하나의 API에서 한 명과 대화하는 상황을 기본으로 가정하되, 연속 채팅을 위해 `conversation_id`를 선택적으로 지원한다.
    - 대화 이력은 최근 N턴만 유지해 프롬프트 길이와 응답 속도를 관리한다.
    - 사용자가 채팅을 종료하거나 일정 시간이 지나면 임시 이력을 삭제한다.
    - 예시 구현:

```python
import time
from collections import defaultdict, deque


class ConversationMemory:
    def __init__(self, max_turns: int = 5, ttl_seconds: int = 3600):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self.store = defaultdict(lambda: deque(maxlen=max_turns))
        self.updated_at = {}

    def get(self, conversation_id: str) -> list[dict]:
        self._expire_old()
        return list(self.store.get(conversation_id, []))

    def append(
        self,
        conversation_id: str,
        user_message: str,
        assistant_answer: str,
        recommendations: list[dict],
        parsed_query: dict,
    ) -> None:
        self.store[conversation_id].append(
            {
                "user": user_message,
                "assistant": assistant_answer,
                "parsed_query": parsed_query,
                "recommendations": recommendations,
            }
        )
        self.updated_at[conversation_id] = time.time()

    def delete(self, conversation_id: str) -> None:
        self.store.pop(conversation_id, None)
        self.updated_at.pop(conversation_id, None)

    def _expire_old(self) -> None:
        now = time.time()
        expired = [
            key
            for key, updated in self.updated_at.items()
            if now - updated > self.ttl_seconds
        ]
        for key in expired:
            self.delete(key)
```
- **성능 요구사항**:
    - 추천 후보 생성과 랭킹은 사전 집계 또는 캐싱을 활용해 빠르게 처리한다.
    - LLM 호출 전 추천 후보 수를 제한해 전체 응답 시간을 줄인다.
    - 디버깅 로그 기록이 추천 응답 속도에 큰 영향을 주지 않도록 로그 레벨과 기록 범위를 설정한다.

## 6. 2-hong-community-fe 프론트엔드 통합
- **API 연동**: FastAPI로 구현된 백엔드 API와 2-hong-community-fe 프론트엔드 연동
- **실시간 채팅 기능 구현**: 프론트엔드에서 실시간으로 채팅 봇과 상호작용할 수 있도록 기능 구현
- **채팅 봇 페이지 추가**:
    - 2-hong-community-fe에 식당 추천 챗봇 전용 페이지를 추가한다.
    - 기존 커뮤니티 화면의 메뉴, 버튼, 라우팅을 통해 채팅 봇 페이지로 이동할 수 있도록 한다.
    - 채팅 봇 페이지는 대화 영역, 메시지 입력창, 추천 식당 결과 영역을 포함한다.
- **화면 표시 항목**:
    - 사용자 메시지
    - 챗봇 자연어 답변
    - 추천 식당 카드 또는 리스트
    - 식당명, 주소, 카테고리, 추천 이유
    - 현재 대화에서 이어진 이전 추천 조건 또는 후속 질문 맥락
- **상호작용 흐름**:
    - 채팅 시작 버튼 클릭
    - 프론트엔드가 신규 `conversation_id`를 생성하거나 백엔드 세션 생성 API를 호출
    - 사용자가 검색어 메시지 입력 후 전송
    - 프론트엔드가 백엔드 API 호출
    - 백엔드에서 추천 결과와 챗봇 응답 수신
    - 프론트엔드에서 챗봇 응답과 추천 식당 정보를 채팅 메시지 형태로 표시
    - 사용자가 후속 메시지 입력 시 동일 `conversation_id`로 API를 재호출해 다회성 연속 채팅을 유지
    - 채팅 종료 버튼 클릭 시 대화 종료
- **UI/UX 고려사항**:
    - 채팅 인터페이스는 직관적이고 반응성이 좋아야 한다.
    - 추천 식당 정보는 명확하게 구분된 카드 또는 리스트 형태로 제공되어야 한다.
- **오류 처리**:
    - 추천 가능한 식당이 없을 경우 조건 변경을 요청하는 메시지를 표시한다.
    - 백엔드 또는 LLM 호출 실패 시 재시도 안내 메시지를 표시한다.

## 7. CI/CD 및 AWS 배포
- **Github Actions CI/CD 설정**: 코드 변경 시 자동으로 테스트 및 배포가 이루어지도록 Github Actions 설정
- **AWS 배포**: AWS ECS 또는 AWS Lambda를 활용해 FastAPI 백엔드 배포
- **배포 자동화**: Github Actions에서 AWS CLI 또는 Terraform을 활용해 배포 자동화 스크립트 작성

## 8. 검증 및 보고서 포함 항목
- **추천 모델 검증**:
    - 추천 모델 검증은 정성 평가, 오프라인 정량 평가, baseline 비교, 추론 속도 측정을 함께 수행한다.
    - 검색어별 추천 결과가 사용자의 의도와 맞는지 정성 평가한다.
    - `성수 데이트`, `라면 맛집`, `압구정 파스타 데이트`, `강남 한식`, `미슐랭 맛집` 등 요구사항과 설계상 취약 케이스를 테스트한다.
    - 각 테스트 케이스에서 상위 5개 추천 결과의 `shop_name`, `categories`, `awards`, `semantic_score`, `behavior_score`, 최종 `score`, 추천 근거를 함께 확인한다.
    - 디버깅 로그를 활용해 검색 의도 추출, `sub_session` 분리, 검색어 전파, 후보 필터링, 랭킹 점수 계산이 의도대로 수행되었는지 확인한다.
- **오프라인 정량 평가 데이터 구성**:
    - 정답 레이블은 `reservation` 이벤트를 사용한다.
        - 예약은 사용자의 최종 전환 행동이므로 추천 관련성 판단에서 가장 신뢰도 높은 신호로 간주한다.
    - 평가 단위는 `sub_session`으로 설정한다.
        - 같은 원본 `session_id` 안에 여러 검색 의도가 섞일 수 있으므로, `session_id` 단위가 아니라 `sub_session` 단위로 train/test를 분리한다.
    - Hold-out 방식:
        - 전체 reservation 포함 `sub_session` 중 80%를 train, 20%를 test로 분리한다.
        - train 데이터로 행동 점수 인덱스를 생성하고, test 데이터의 `search_query`를 추천 엔진에 입력한다.
        - 추천 결과 Top-K 안에 test reservation의 `shop_id`가 포함되는지 측정한다.
    - 데이터 누수를 막기 위해 test `sub_session`의 행동 이벤트는 행동 점수 집계에 포함하지 않는다.
- **정량 평가 지표**:
    - 추천 시스템 표준 지표를 사용해 Top-K 추천 품질을 측정한다.
    - 기본 K는 챗봇 응답 개수와 맞춰 `K=5`로 둔다.

    | 지표 | 설명 | 사용 목적 | 목표 기준 |
    | --- | --- | --- | --- |
    | `Precision@K` | 추천 K개 중 정답 또는 관련 식당 비율 | 추천 목록의 정확도 측정 | 0.20 이상 |
    | `Recall@K` | 전체 정답 중 추천 K개에 포함된 비율 | 관련 식당 회수율 확인 | 참고 지표 |
    | `Hit Rate@K` | 정답 식당이 Top-K에 1개 이상 포함된 비율 | 단일 예약 정답에서 핵심 성공률 측정 | 0.30 이상 |
    | `MRR` | 정답 식당 순위의 역수 평균 | 정답이 상위에 배치되는지 측정 | 0.20 이상 |
    | `NDCG@K` | 순위를 고려한 누적 이득 | 상위 순위 품질 측정 | 0.25 이상 |
    | `Coverage` | 추천에 등장한 고유 식당 수 / 전체 500개 식당 수 | 인기 식당 쏠림과 다양성 확인 | 0.30 이상 |

    - 단일 reservation을 정답으로 사용하는 구조에서는 `Hit Rate@5`와 `MRR`을 핵심 지표로 본다.
    - `NDCG@5`는 정답을 단순 포함하는지뿐 아니라 얼마나 앞 순위에 배치하는지 확인하기 위해 사용한다.
    - `Coverage`는 특정 인기 식당만 반복 추천되는 문제를 탐지하기 위해 사용한다.
    - Metric 계산 예시:

    ```python
    import math


    def hit_rate_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
        return 1.0 if relevant.intersection(recommended[:k]) else 0.0


    def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
        if k == 0:
            return 0.0
        return len(relevant.intersection(recommended[:k])) / k


    def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
        if not relevant:
            return 0.0
        return len(relevant.intersection(recommended[:k])) / len(relevant)


    def mrr(recommended: list[str], relevant: set[str]) -> float:
        for rank, shop_id in enumerate(recommended, start=1):
            if shop_id in relevant:
                return 1.0 / rank
        return 0.0


    def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
        dcg = 0.0
        for idx, shop_id in enumerate(recommended[:k], start=1):
            if shop_id in relevant:
                dcg += 1.0 / math.log2(idx + 1)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
        return dcg / idcg if idcg > 0 else 0.0


    def coverage(all_recommendations: list[list[str]], total_shop_count: int) -> float:
        recommended_unique = {shop_id for recs in all_recommendations for shop_id in recs}
        return len(recommended_unique) / total_shop_count
    ```
- **교차 검증 및 안정성 검증**:
    - Hold-out 평가 외에 `sub_session` 단위 5-Fold Cross Validation을 수행한다.
    - 각 fold마다 `Hit Rate@5`, `MRR`, `NDCG@5`, `Coverage`를 계산한다.
    - 보고서에는 평균과 표준편차를 함께 기록한다.
    - fold 간 표준편차가 큰 경우 특정 검색어, 특정 기간, 특정 인기 식당에 과적합되었는지 분석한다.
- **쿼리 유형별 슬라이싱 분석**:
    - 전체 평균 성능만으로는 취약 구간을 확인하기 어렵기 때문에 검색어 유형별로 성능을 나누어 보고한다.
    - 행동 로그 풍부 쿼리와 희소 쿼리를 분리해 hybrid 가중치 전환 전략이 효과적인지 확인한다.
    - 지역 지정 쿼리와 카테고리 중심 쿼리를 분리한다.
        - 예시: `강남 한식`, `압구정 파스타` vs `라면 맛집`, `오마카세`
    - 상황/목적 쿼리를 별도 측정한다.
        - 예시: `성수 데이트`, `회식하기 좋은 한식`, `조용한 식당`
    - 품질 지향 쿼리를 별도 측정한다.
        - 예시: `미슐랭 맛집`, `수상 경력 있는 식당`
    - 취약 케이스는 상위 추천 결과와 로그를 함께 확인해 원인을 분류한다.
        - 임베딩 유사도 문제
        - 행동 로그 부족
        - 세션 검색어 전파 오류
        - 카테고리/수상 정보 부족
        - 인기 식당 편향
- **Baseline 비교 실험**:
    - 제안 방식인 Hybrid 추천의 효과를 검증하기 위해 아래 비교군을 동일 test set에서 평가한다.

    | 방식 | 설명 | 검증 목적 |
    | --- | --- | --- |
    | `Random` | 500개 식당 중 무작위 Top-K 반환 | 추천 모델이 랜덤보다 유의미한지 확인 |
    | `Popularity` | 검색어와 무관하게 전체 행동 가중합 상위 식당 반환 | 쿼리 맥락 반영 효과 확인 |
    | `Semantic-only` | 임베딩 의미 유사도만 사용 | 행동 로그 추가 효과 확인 |
    | `Behavior-only` | 행동 로그 점수만 사용하고 없으면 인기순 fallback | cold-start에서 의미 유사도의 필요성 확인 |
    | `Hybrid` | 의미 유사도와 행동 로그 점수를 동적으로 결합 | 최종 제안 방식의 종합 성능 확인 |

    - 예상 성능 순서는 `Random < Popularity < Behavior-only 또는 Semantic-only < Hybrid`로 설정한다.
    - `Random`의 `Hit@5` 이론 기준은 `5 / 500 = 0.010`이다.
    - 보고서에는 아래 형식으로 결과 표를 포함한다.

    ```text
    방식          Hit@5   MRR    NDCG@5   Coverage
    Random        0.010   0.004  0.005    1.000
    Popularity    0.XXX   0.XXX  0.XXX    0.XXX
    Semantic      0.XXX   0.XXX  0.XXX    0.XXX
    Behavior      0.XXX   0.XXX  0.XXX    0.XXX
    Hybrid        0.XXX   0.XXX  0.XXX    0.XXX
    ```

    - Baseline 평가 실행 예시:

    ```python
    import random
    import numpy as np
    import pandas as pd

    from evaluation.metrics import (
        coverage,
        hit_rate_at_k,
        mrr,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )


    def evaluate_ranker(test_cases: pd.DataFrame, ranker_fn, all_shop_ids: list[str], k: int = 5):
        rows = []
        all_recs = []

        for _, case in test_cases.iterrows():
            query = case["search_query"]
            relevant = {str(case["shop_id"])}
            recommended = [str(x) for x in ranker_fn(query, k)]
            all_recs.append(recommended)

            rows.append(
                {
                    "precision": precision_at_k(recommended, relevant, k),
                    "recall": recall_at_k(recommended, relevant, k),
                    "hit": hit_rate_at_k(recommended, relevant, k),
                    "mrr": mrr(recommended, relevant),
                    "ndcg": ndcg_at_k(recommended, relevant, k),
                }
            )

        metrics = pd.DataFrame(rows).mean().to_dict()
        metrics["coverage"] = coverage(all_recs, total_shop_count=len(all_shop_ids))
        return metrics


    def random_ranker_factory(all_shop_ids: list[str]):
        def ranker(query: str, k: int):
            return random.sample(all_shop_ids, k)
        return ranker


    def popularity_ranker_factory(global_popularity: pd.DataFrame):
        ranked_ids = global_popularity.sort_values("score", ascending=False)["shop_id"].astype(str)
        ranked_ids = ranked_ids.tolist()

        def ranker(query: str, k: int):
            return ranked_ids[:k]
        return ranker


    def engine_ranker_factory(engine, mode: str):
        def ranker(query: str, k: int):
            results = engine.recommend(query=query, top_k=k, mode=mode)
            return [item["shop_id"] for item in results]
        return ranker


    def run_baseline_report(test_cases, all_shop_ids, global_popularity, engine):
        rankers = {
            "Random": random_ranker_factory(all_shop_ids),
            "Popularity": popularity_ranker_factory(global_popularity),
            "Semantic": engine_ranker_factory(engine, mode="semantic_only"),
            "Behavior": engine_ranker_factory(engine, mode="behavior_only"),
            "Hybrid": engine_ranker_factory(engine, mode="hybrid"),
        }
        return pd.DataFrame(
            [
                {"method": name, **evaluate_ranker(test_cases, fn, all_shop_ids)}
                for name, fn in rankers.items()
            ]
        )
    ```

- **추론 속도 및 운영 검증**:
    - 추천 시스템이 챗봇 응답 병목이 되지 않도록 온라인 추천 단계의 실행 시간을 측정한다.
    - 측정 구간:
        - 사용자 검색어 임베딩 시간
        - cosine similarity 계산 시간
        - 행동 점수 조회 시간
        - 하이브리드 스코어 정렬 시간
        - Top-K 결과와 식당 메타데이터 결합 시간
    - 오프라인 인덱싱 결과(`shop_embeddings.npy`, `behavior_scores.pkl`, `shops_indexed.pkl`)를 사용했을 때와 사용하지 않았을 때를 비교해 속도 개선 효과를 보고한다.
    - 목표는 추천 모델 추론 시간이 LLM 호출 시간보다 충분히 작도록 유지하는 것이다.
    - 추론 시간 측정 예시:

    ```python
    import time


    def measure_latency(engine, queries: list[str], top_k: int = 5) -> dict:
        durations = []
        for query in queries:
            start = time.perf_counter()
            engine.recommend(query=query, top_k=top_k)
            durations.append((time.perf_counter() - start) * 1000)

        durations = sorted(durations)
        return {
            "count": len(durations),
            "avg_ms": sum(durations) / len(durations),
            "p50_ms": durations[int(len(durations) * 0.50)],
            "p95_ms": durations[int(len(durations) * 0.95)],
            "max_ms": max(durations),
        }
    ```
- **정성 평가 체크리스트**:
    - 추천 식당이 반드시 `shops.csv`에 존재하는지 확인한다.
    - LLM 응답이 추천 모델 결과에 없는 식당을 추가하지 않는지 확인한다.
    - 주소, 메뉴, 수상 정보 등 데이터에 없는 내용을 추측하지 않는지 확인한다.
    - `라면 맛집` 검색에서 후식 라면만 가진 고가 식당이 과도하게 상위 노출되지 않는지 확인한다.
    - `성수 데이트`, `압구정 파스타 데이트`처럼 지역과 상황이 결합된 쿼리에서 지역, 카테고리, 상황 의도가 모두 반영되는지 확인한다.
    - 후속 질문(`더 조용한 곳`, `가격대 낮은 곳`, `두 번째 식당 자세히 알려줘`)에서 이전 추천 결과와 대화 맥락이 유지되는지 확인한다.
- **보고서 포함 내용**:
    - LLM과 추천 모델링을 포함한 전체 구성
    - 추천 모델 세부 알고리즘
        - `sub_session` 분리 및 검색어 전파
        - `jhgan/ko-sroberta-multitask` 기반 쿼리-식당 임베딩 유사도
        - 이벤트 가중치 기반 행동 로그 점수 집계
        - 행동 데이터 희소도에 따른 hybrid alpha/beta 전환
        - 오프라인 인덱싱과 온라인 추론 구조
    - 사용 모델 및 선정 이유
        - 한국어 검색어 처리 적합성
        - sentence-transformers 기반의 구현 용이성
        - 500개 식당 규모에서 충분한 추론 속도
    - 평가 데이터 구성 방식
        - reservation 이벤트를 정답 레이블로 사용한 이유
        - `sub_session` 단위 train/test 분리 이유
        - 데이터 누수 방지 방식
    - 정량 평가 결과
        - `Hit Rate@5`, `MRR`, `NDCG@5`, `Coverage`
        - 5-Fold Cross Validation 평균 및 표준편차
        - 쿼리 유형별 슬라이싱 분석 결과
    - 비교군 비교
        - Random, Popularity, Semantic-only, Behavior-only, Hybrid 성능 비교
        - Hybrid 방식 채택 근거
        - 랜덤 추천 대비 개선 폭과 행동 로그 기반 랭킹의 효과 분석
    - 추천 성능 개선 및 추론 속도 개선 방법
        - 임베딩 사전 계산
        - 행동 점수 pickle 인덱스화
        - Top-K 후보 제한
        - 로그 희소 쿼리에 대한 semantic fallback
        - 향후 position bias 보정 적용 가능성
    - 실패 사례 및 한계
        - 행동 로그가 부족한 검색어의 성능 한계
        - 카테고리 또는 awards 정보가 부실한 식당의 노출 한계
        - 개인화 미적용 한계
        - LLM 응답이 추천 결과를 얼마나 충실히 따르는지에 대한 검증 필요
    - 대화 예시
        - 일반 추천 요청
        - 지역/카테고리/상황 복합 요청
        - 추천 불가 또는 후보 부족 요청
        - 후속 질문이 포함된 다회성 대화
