# 2-hong-community-chatbot - 상세 기능 명세서

 커뮤니티 서비스 확장 채팅 봇 및 백엔드 구현 상세 가이드

## 1. 프로젝트 개요
- **목표**: 유저의 식당 추천 요청에 대해 사용자 행동 로그 기반으로 식당을 추천하는 챗봇 구현
- **기술 스택**: pytorch, Langchain, MCP, Vector Database, FastAPI
- **주요 기능**:
    - 추천 시스템: 유저의 검색어 (Search_query)와 행동 로그 (User_behavior_log)를 기반으로 식당을 추천하는 챗봇 구현
    - 자연어 응답 LLM 챗봇: 추천된 식당에 대한 자연어 응답 생성
    - 프론트엔드 통합: 2-hong-community-fe 프론트엔드와의 API 연동 및 실시간 채팅 기능 구현


## 2. 커뮤니티 채팅 봇 UI/UX 설계

## 3. 추천 시스템 구현
- **추천 시스템 구현**: 유저의 검색어와 행동 로그를 기반으로 식당을 추천하는 모델 개발

## 4. 자연어 응답 LLM 챗봇 구현
- **LLM 챗봇 구현**: Gemini API (Gemini 2.5 Flash)와 Langchain을 활용하여 추천된 식당에 대한 자연어 응답 생성

## 5. 채팅 봇 서비스 확장 및 백엔드 구현
- **FastAPI를 활용한 백엔드 구현**: FastAPI를 활용해 채팅 봇의 백엔드 API 구현

## 6. 2-hong-comnmunity-fe 프론트엔드 통합
- **API 연동**: FastAPI로 구현된 백엔드 API와 2-hong-community-fe 프론트엔드 연동
- **실시간 채팅 기능 구현**: 프론트엔드에서 실시간으로 채팅 봇과 상호작용할 수 있도록 기능 구현


