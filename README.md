# 2-hong-community-chatbot

식당 추천 챗봇 FastAPI 백엔드입니다. `shops.csv`에 존재하는 식당만 추천하며, `logs.csv` 행동 로그와 검색어-식당 의미 유사도를 결합해 추천합니다.

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

고품질 한국어 임베딩과 Gemini 응답 생성을 쓰려면 선택 의존성을 추가합니다.

```bash
pip install sentence-transformers google-genai
```

## 인덱스 생성

기본 데이터 위치는 저장소 루트의 `logs.csv`, `shops.csv`입니다.

```bash
python -m src.recommender.indexer --embedding-backend hashing
```

`sentence-transformers`가 설치되어 있고 모델 다운로드가 가능한 환경이면 아래처럼 실행할 수 있습니다.

```bash
python -m src.recommender.indexer --embedding-backend sentence-transformers
```

개발 중 빠른 확인용으로 일부 로그만 쓰려면:

```bash
python -m src.recommender.indexer --embedding-backend hashing --max-log-rows 10000
```

## 서버 실행

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

브라우저에서 `http://localhost:8000`을 열면 간단한 챗봇 프론트엔드를 사용할 수 있습니다.
`frontend/index.html`을 직접 열어도 기본 API 주소로 `http://localhost:8000`을 사용합니다.

주요 API:

- `POST /chat/session`
- `POST /chat/recommend`
- `DELETE /chat/session/{conversation_id}`
- `GET /health`

요청 예시:

```json
{
  "message": "압구정 파스타 데이트",
  "top_k": 5
}
```

## 평가

인덱스 생성 후 baseline 비교를 실행합니다.

```bash
python -m src.evaluation.run_eval --k 5 --max-log-rows 10000
```

## 환경 변수

- `GEMINI_API_KEY`: 설정 시 Gemini 2.5 Flash로 전처리/응답 생성
- `GEMINI_MODEL`: 기본값 `gemini-2.5-flash`
- `RAW_DATA_DIR`: 기본값 `.`
- `PROCESSED_DATA_DIR`: 기본값 `data/processed`
- `EMBEDDING_BACKEND`: `auto`, `hashing`, `sentence-transformers`

## GitHub Actions CI/CD 설정

CI/CD 워크플로는 `.github/workflows/ci.yml`에 있습니다. CI는 push/PR에서 실행되고, 배포는 CI 성공 후 `main` 브랜치 push 또는 수동 실행에서만 동작합니다.

GitHub Secrets:

- `SERVER_HOST`: 배포 서버 IP 또는 도메인
- `SERVER_USER`: SSH 접속 사용자
- `SSH_PRIVATE_KEY`: 서버 접속용 private key

GitHub Variables:

- `APP_DIR`: 서버의 repository 경로, 기본값 `~/2-hong-community-chatbot`
- `SERVICE_NAME`: 재시작할 systemd 서비스명, 기본값 `2-hong-community-chatbot`
- `EMBEDDING_BACKEND`: 배포 시 인덱스 생성에 사용할 임베딩 방식, 기본값 `hashing`

서버에는 최초 1회 repository clone과 systemd 서비스 등록이 필요합니다.
