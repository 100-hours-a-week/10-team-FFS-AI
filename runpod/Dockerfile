FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# 의존성 설치 (torch는 베이스 이미지에 포함됨)
COPY requirements-runpod.txt .
RUN pip install --no-cache-dir -r requirements-runpod.txt

# 모델 사전 다운로드 (빌드 시 - Cold Start 방지)
# NSFW 모델
RUN python -c "from transformers import pipeline; \
    pipeline('image-classification', 'Falconsai/nsfw_image_detection'); \
    print('NSFW model downloaded')"

# CLIP 모델
RUN python -c "import open_clip; \
    open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k'); \
    print('CLIP model downloaded')"

# FashionSigLIP 모델 - 모델만 다운로드 (Processor는 런타임에 로드)
RUN python -c "from transformers import AutoModel; \
    AutoModel.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True); \
    print('FashionSigLIP model downloaded')"

# 소스 코드 복사
COPY validators.py .
COPY handler.py .

# 실행
CMD ["python", "-u", "handler.py"]
