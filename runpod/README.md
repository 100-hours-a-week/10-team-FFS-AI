# RunPod Serverless - FFS Validator

이미지 검증용 RunPod Serverless Endpoint 설정 가이드

## 파일 구조

```
runpod/
├── Dockerfile              # Docker 이미지 설정
├── handler.py              # RunPod 핸들러 (진입점)
├── validators.py           # AI 모델 래퍼
├── requirements-runpod.txt # 의존성
├── README.md               # 이 문서
└── DEPLOY.md               # 배포 가이드
```

---

## Docker 이미지 정보

| 항목 | 값 |
|------|-----|
| 베이스 이미지 | `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel` |
| CUDA | 11.8 (베이스 이미지 포함) |
| PyTorch | 2.1.0 (베이스 이미지 포함) |
| Python | 3.10 |

### AI 모델
| 모델 | HuggingFace ID | 용도 |
|------|---------------|------|
| NSFW | `Falconsai/nsfw_image_detection` | 부적절 콘텐츠 감지 |
| CLIP | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | 패션 도메인 분류 |
| SigLIP | `Marqo/marqo-fashionSigLIP` | 임베딩/유사도 검증 |

---

## API 사용법

### 요청 형식
```json
{
  "input": {
    "action": "validate",
    "images": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
  }
}
```

### 지원 Action
| Action | 설명 |
|--------|------|
| `validate` | 전체 검증 (NSFW + 패션 + 임베딩) |
| `nsfw` | NSFW 검증만 |
| `fashion` | 패션 분류만 |
| `embedding` | 임베딩 생성만 |

### 응답 형식
```json
{
  "results": [
    {
      "url": "https://example.com/image1.jpg",
      "nsfw": {"is_nsfw": false, "score": 0.05},
      "fashion": {"is_fashion": true, "score": 0.85},
      "embedding": [0.1, 0.2, ...]
    }
  ]
}
```

---

## RunPod 콘솔 탭 설명

### Overview (개요)
- **Ready**: Endpoint 준비 상태
- **$0.00000/s**: 현재 비용 (워커 미실행 시 0)
- **running workers**: 실행 중인 GPU 워커 수
- **jobs in queue**: 대기 중인 작업 수

### Metrics (메트릭)
- **Requests**: 요청 통계 (성공/실패)
- **Execution Time**: 실제 처리 시간
- **Delay Time**: 대기 시간 (Cold Start 포함)
- **Cold Start Count**: 워커 새로 시작한 횟수

### Logs (로그)
- 에러 디버깅용
- CUDA 에러, OOM 에러 확인

### Requests (테스트)
- API 직접 테스트 가능
- Run 버튼으로 즉시 테스트

### Workers (워커)
- 사용 가능한 GPU 목록
- GPU 타입, 위치, 상태 확인

---

## CUDA 버전 호환성

> ⚠️ CUDA 충돌 걱정 없음!

- 베이스 이미지에 CUDA 11.8이 미리 설치됨
- PyTorch도 CUDA 11.8용으로 빌드된 버전 포함
- 직접 CUDA 설치 안 해서 버전 충돌 가능성 없음

---

## 환경변수

| 변수 | 설명 | 예시 |
|------|------|------|
| `RUNPOD_API_KEY` | RunPod API 키 | `rpa_xxx...` |
| `RUNPOD_ENDPOINT_ID` | Endpoint ID | `oar9l8rt41v77` |

---

## 테스트

### RunPod 콘솔에서
Requests 탭 → 아래 입력 → Run 클릭
```json
{
  "input": {
    "action": "validate",
    "images": ["https://fastly.picsum.photos/id/1/200/200.jpg"]
  }
}
```

### curl로
```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer {API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "validate", "images": ["https://example.com/test.jpg"]}}'
```

---

## 트러블슈팅

| 문제 | 해결 |
|------|------|
| Cold Start 느림 | Active Workers 1개 이상 유지 |
| OOM 에러 | GPU VRAM 24GB 이상 선택 |
| 모델 로딩 실패 | Logs 탭에서 에러 확인 |
| CUDA 에러 | 베이스 이미지 버전 확인 |
