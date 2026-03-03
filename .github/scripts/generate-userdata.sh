#!/bin/bash
set -e

# 인자 받기
ENVIRONMENT=$1
IMAGE_URI=$2
AWS_REGION=$3

# 현재 스크립트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="${SCRIPT_DIR}/../templates"

# 템플릿 파일 base64 인코딩
COMPOSE_B64=$(base64 -w 0 "${TEMPLATES_DIR}/docker-compose.yml")
ENV_B64=$(base64 -w 0 "${TEMPLATES_DIR}/.env.template")

# User Data 스크립트 생성
cat > user-data.sh <<'EOF'
#!/bin/bash
set -e

# 로깅 설정
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "=== User Data Script Started ==="

# 환경 변수
export AWS_REGION="{{AWS_REGION}}"
export IMAGE_URI="{{IMAGE_URI}}"
export ENVIRONMENT="{{ENVIRONMENT}}"

COMPOSE_B64="{{COMPOSE_B64}}"
ENV_B64="{{ENV_B64}}"

# 필수 도구 설치
echo "Installing required tools..."

apt update -y
apt install -y ca-certificates curl gnupg jq gettext-base awscli

install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt update -y
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

systemctl enable docker
systemctl start docker

usermod -a -G docker ubuntu

echo "Docker version:"
docker --version
docker compose version

# 작업 디렉토리
APP_DIR="/home/ubuntu/app"
mkdir -p ${APP_DIR}
cd ${APP_DIR}

# ECR 로그인
echo "Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin $(echo ${IMAGE_URI} | cut -d'/' -f1)

# SSM에서 환경 변수 가져오기
echo "Fetching configuration from Parameter Store..."

get_param() {
  aws ssm get-parameter --name "$1" --query "Parameter.Value" --output text --region ${AWS_REGION} 2>/dev/null || echo ""
}

get_secure_param() {
  aws ssm get-parameter --name "$1" --with-decryption --query "Parameter.Value" --output text --region ${AWS_REGION} 2>/dev/null || echo ""
}

# 환경 변수 설정
export APP_ENV=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/app-env")
export DEBUG=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/debug")
export HOST=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/host")
export PORT=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/port")

export QDRANT_HOST=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/qdrant/host")
export QDRANT_PORT=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/qdrant/port")
export QDRANT_COLLECTION_NAME=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/qdrant/collection-name")
export QDRANT_SHOP_COLLECTION_NAME=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/qdrant/shop-collection-name")
export QDRANT_USE_HTTPS=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/qdrant/use-https")
export QDRANT_PREFER_GRPC=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/qdrant/prefer-grpc")

export REDIS_HOST=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/redis/host")
export REDIS_PORT=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/redis/port")
export REDIS_DB=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/redis/db")
export REDIS_PASSWORD=$(get_secure_param "/klosetlab/${ENVIRONMENT}/fastapi/redis/password")
export REDIS_MAX_CONNECTIONS=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/redis/max-connections")

export HF_HOME=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/huggingface/hf-home")
export HF_TOKEN=$(get_secure_param "/klosetlab/${ENVIRONMENT}/fastapi/huggingface/hf-token")

export EMBEDDING_MODEL=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/model/embedding-model")
export UPSTAGE_API_KEY=$(get_secure_param "/klosetlab/${ENVIRONMENT}/fastapi/model/upstage-api-key")
export CAPTION_MODEL=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/model/caption-model")

export GEMINI_API_KEY=$(get_secure_param "/klosetlab/${ENVIRONMENT}/fastapi/model/gemini-api-key")
export GEMINI_MODEL=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/model/gemini-model")

export VTON_MODEL=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/model/vton-model")

export OPENAI_API_KEY=$(get_secure_param "/klosetlab/${ENVIRONMENT}/fastapi/model/openai-api-key")
export OPENAI_CHAT_MODEL=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/model/openai-chat-model")

export LLM_TIMEOUT=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/model/llm-timeout")
export LLM_MAX_RETRIES=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/model/llm-max-retries")

export USE_MOCK_ANALYZER=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/mock-analyzer/use-mock-analyzer")

export LANGFUSE_SECRET_KEY=$(get_secure_param "/klosetlab/${ENVIRONMENT}/fastapi/mock-analyzer/langfuse-secret-key")
export LANGFUSE_PUBLIC_KEY=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/mock-analyzer/langfuse-public-key")
export LANGFUSE_HOST=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/mock-analyzer/langfuse-host")
export LANGFUSE_DEBUG=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/mock-analyzer/langfuse-debug")

export NAVER_CLIENT_ID=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/mock-analyzer/naver-client-id")
export NAVER_CLIENT_SECRET=$(get_secure_param "/klosetlab/${ENVIRONMENT}/fastapi/mock-analyzer/naver-client-secret")

export BACKEND_INTERNAL_URL=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/spring/url")
export BACKEND_INTERNAL_API_KEY=$(get_secure_param "/klosetlab/${ENVIRONMENT}/fastapi/spring/api-key")

export KAFKA_BOOTSTRAP_SERVERS=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/kafka/server")
export KAFKA_ANALYZE_CONSUMER_GROUP=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/kafka/analyze-consumer-group")
export KAFKA_MAX_CONCURRENT_TASKS=$(get_param "/klosetlab/${ENVIRONMENT}/fastapi/kafka/max-concurrent-tasks")

echo "Restoring docker-compose.yml..."
echo "${COMPOSE_B64}" | base64 -d > docker-compose.yml

echo "Restoring .env.template..."
echo "${ENV_B64}" | base64 -d > .env.template

echo "Generating .env..."
envsubst < .env.template > .env

chmod 600 .env
chmod 644 docker-compose.yml

echo "✅ Configuration files created"
ls -lh docker-compose.yml .env

# 배포
echo "Deploying application..."

# 이미지 Pull
echo "Pulling Docker image..."
docker compose pull

# 컨테이너 시작
echo "Starting containers..."
docker compose up -d --remove-orphans

echo "Waiting for container..."

for i in {1..30}; do
  if docker compose ps | grep "Up"; then
    echo "Container running"
    exit 0
  fi
  sleep 10
done

echo "Deployment failed"
docker compose logs --tail=50
exit 1
EOF

sed -i "s|{{COMPOSE_B64}}|${COMPOSE_B64}|g" user-data.sh
sed -i "s|{{ENV_B64}}|${ENV_B64}|g" user-data.sh
sed -i "s|{{IMAGE_URI}}|${IMAGE_URI}|g" user-data.sh
sed -i "s|{{AWS_REGION}}|${AWS_REGION}|g" user-data.sh
sed -i "s|{{ENVIRONMENT}}|${ENVIRONMENT}|g" user-data.sh

# Base64 인코딩
USER_DATA_BASE64=$(base64 -w 0 user-data.sh)
echo "user_data_base64=${USER_DATA_BASE64}" >> $GITHUB_OUTPUT

echo "✅ User Data script created and encoded"
