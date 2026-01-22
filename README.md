# KlosetLab AI Service

FastAPI 기반 의류 분석 및 코디 추천 AI 서비스

## 프로젝트 개요

사용자의 옷장을 AI로 관리하고 맞춤형 코디를 추천하는 서비스입니다.

## 프로젝트 구조

```
app/
├── main.py              # FastAPI 앱 진입점
├── config.py            # 환경변수 설정
├── core/                # 인프라 (Qdrant, S3, Models)
├── common/              # 공통 유틸리티
├── closet/              # 옷장 관리 (이미지 검증, 전처리)
├── embedding/           # 임베딩 생성 및 저장
└── outfit/              # 코디 추천
```

## 설치 및 실행

### 1. 환경 설정
```bash
cp .env.example .env
# .env 파일 수정
```

### 2. 의존성 설치
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 개발 환경
```

### 3. 모델 다운로드 (선택)
```bash
python scripts/download_models.py
```

### 4. 서버 실행
```bash
uvicorn app.main:app --reload --port 8000
```

API 문서: http://localhost:8000/docs

## 기술 스택

- Python 3.11
- FastAPI
- Qdrant (Vector DB)
- AWS S3 (Storage)
- HuggingFace Transformers
- Sentence Transformers
