from __future__ import annotations

import logging
from typing import BinaryIO

import boto3
from botocore.exceptions import NoCredentialsError

from app.config import get_settings
from app.core.exceptions import StorageError

logger = logging.getLogger(__name__)
settings = get_settings()


class S3Storage:
    def __init__(self: S3Storage) -> None:
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        self.bucket_name = settings.s3_bucket_name

    def upload_file(
        self: S3Storage,
        file_obj: BinaryIO,
        object_key: str,
        content_type: str = "image/png",
    ) -> str | None:
        """
        파일을 S3에 업로드

        Args:
            file_obj: 파일 객체 (BytesIO 등)
            object_key: S3 저장 경로 (Key)
            content_type: MIME 타입

        Returns:
            업로드된 파일의 S3 URL (실패 시 None)
        """
        try:
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket_name,
                object_key,
                ExtraArgs={"ContentType": content_type},
            )
            # S3 URL 생성
            # (Note: 버킷 정책에 따라 public access 가능 여부 다름. 여기서는 단순히 URL 문자열 구성)
            url = f"https://{self.bucket_name}.s3.{settings.aws_region}.amazonaws.com/{object_key}"
            logger.info(f"S3 업로드 성공: {url}")
            return url
        except NoCredentialsError:
            logger.error("AWS 자격 증명 실패")
            raise StorageError("AWS Credentials not found") from None
        except Exception as e:
            logger.error(f"S3 업로드 실패: {e}")
            raise StorageError(f"S3 Upload failed: {str(e)}") from e


# 싱글톤
_storage_instance = None


def get_storage() -> S3Storage:
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = S3Storage()
    return _storage_instance
