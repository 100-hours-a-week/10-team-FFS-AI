import logging
from typing import Any

from ray import serve
from starlette.requests import Request

from app.core.models import FashionClassifier, NSFWValidator, SegmentationModel
from app.main import app as fastapi_app

logger = logging.getLogger("ray.serve")


# 1. NSFW 모델 배포
@serve.deployment(
    num_replicas=1, ray_actor_options={"num_gpus": 0.3}, route_prefix="/nsfw"
)
class NSFWDeployment:
    def __init__(self) -> None:
        logger.info("Initializing NSFWDeployment...")
        self.model = NSFWValidator()
        self.model.load_model()
        logger.info("NSFWDeployment initialized.")

    async def __call__(self, request: Request) -> list[dict[str, Any]]:
        # 이미지 데이터(bytes)를 받아서 예측
        image_data = await request.body()
        from io import BytesIO

        from PIL import Image

        image = Image.open(BytesIO(image_data)).convert("RGB")
        return self.model.predict(image)


# 2. Fashion 분류 모델 배포
@serve.deployment(
    num_replicas=1, ray_actor_options={"num_gpus": 0.3}, route_prefix="/fashion"
)
class FashionDeployment:
    def __init__(self) -> None:
        logger.info("Initializing FashionDeployment...")
        self.model = FashionClassifier()
        self.model.load_model()
        logger.info("FashionDeployment initialized.")

    async def __call__(self, request: Request) -> dict[str, Any]:
        # Expect JSON: {"image_url": "...", "texts": [...]}
        data = await request.json()

        image_url = data.get("image_url")
        texts = data.get("texts", ["clothing"])

        import io

        import httpx
        from PIL import Image

        # Download Image
        if image_url:
            async with httpx.AsyncClient() as client:
                resp = await client.get(image_url)
                resp.raise_for_status()
                image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            # Fallback for testing: maybe raw bytes were expected?
            # But here we stick to URL protocol for consistency with Validator
            return {"error": "image_url is required"}

        # Call Model
        return self.model.get_features(image, texts)


# 3. 배경 제거 모델 배포
@serve.deployment(
    num_replicas=1, ray_actor_options={"num_gpus": 0.4}, route_prefix="/segmentation"
)
class SegmentationDeployment:
    def __init__(self) -> None:
        logger.info("Initializing SegmentationDeployment...")
        self.model = SegmentationModel()
        self.model.load_model()
        logger.info("SegmentationDeployment initialized.")

    async def __call__(self, request: Request) -> bytes:
        image_data = await request.body()
        from io import BytesIO

        from PIL import Image

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # 예측 (Tensor 반환됨)
        preds = self.model.predict(image)  # 여기 수정 필요할 수 있음 (전처리 로직)

        # 마스크를 이미지로 변환해서 반환
        from torchvision import transforms

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)

        buf = BytesIO()
        pred_pil.save(buf, format="PNG")
        return buf.getvalue()


# 4. FastAPI 앱 배포 (Ingress)
@serve.deployment(num_replicas=1, route_prefix="/")
@serve.ingress(fastapi_app)
class KlosetIngress:
    pass


# 배포 객체 생성
nsfw = NSFWDeployment.bind()
fashion = FashionDeployment.bind()
seg = SegmentationDeployment.bind()
ingress = KlosetIngress.bind()
