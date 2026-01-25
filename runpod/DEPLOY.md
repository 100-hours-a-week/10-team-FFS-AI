# RunPod 배포 가이드

## 1. 사전 준비

### RunPod API Key 발급
1. [RunPod Console](https://www.runpod.io/console/user/settings) 접속
2. `API Keys` 탭에서 `Create API Key` 클릭
3. 발급된 키 복사 (한 번만 보여짐!)

### DockerHub 계정
Docker 이미지를 업로드할 DockerHub 계정이 필요해요.

---

## 2. Docker 이미지 빌드 & 푸시

```bash
cd /Users/ijeonglim/Downloads/Final_Project/Project/runpod

# Docker 빌드 (약 10-15분 소요)
docker build -t YOUR_DOCKERHUB_USERNAME/ffs-validator:latest .

# DockerHub 로그인
docker login

# 이미지 푸시
docker push YOUR_DOCKERHUB_USERNAME/ffs-validator:latest
```

---

## 3. RunPod Serverless Endpoint 생성

1. [RunPod Serverless](https://www.runpod.io/console/serverless) 접속
2. `New Endpoint` 클릭
3. 설정:
   - **Name**: `ffs-validator`
   - **Container Image**: `YOUR_DOCKERHUB_USERNAME/ffs-validator:latest`
   - **GPU Type**: `RTX 3090` 또는 `RTX 4090` (16GB+ VRAM)
   - **Max Workers**: 1 (테스트용)
   - **Active Workers**: 0 (사용할 때만 실행)
4. `Deploy` 클릭
5. **Endpoint ID** 복사

---

## 4. 환경변수 설정

`.env` 파일에 추가:
```bash
RUNPOD_API_KEY=your_api_key_here
RUNPOD_ENDPOINT_ID=your_endpoint_id_here
USE_MOCK_VALIDATOR=false  # Mock 모드 끄기
```

---

## 5. 테스트

```bash
# RunPod Endpoint 직접 테스트
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer {API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "validate", "images": ["https://picsum.photos/200/200.jpg"]}}'
```
