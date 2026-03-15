"""커머스 배치 스케줄러.

KST 새벽 2시에 배치를 시작하고, 6시까지 연속 처리한다.
시간 내 완료되지 않으면 중단하고 다음 날 2시에 이어서 진행한다.
vLLM 서버가 꺼져 있으면 해당 날 배치를 스킵한다.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from app.commerce.service import CommerceBatchService
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# 한국 표준시 (UTC+9)
KST = timezone(timedelta(hours=9))


class CommerceBatchScheduler:
    """KST 새벽 시간대에 배치를 실행하는 스케줄러."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.service = CommerceBatchService(self.settings)

    def _now_kst(self) -> datetime:
        """현재 한국 시간을 반환한다."""
        return datetime.now(KST)

    def _next_batch_start(self, now: datetime) -> datetime:
        """다음 배치 시작 시각(KST batch_start_hour시)을 계산한다.

        - 현재 시각이 batch_start_hour 이전이면 오늘 batch_start_hour시
        - 현재 시각이 batch_start_hour 이후이면 내일 batch_start_hour시
        """
        today_start = now.replace(
            hour=self.settings.batch_start_hour,
            minute=0,
            second=0,
            microsecond=0,
        )

        if now < today_start:
            return today_start

        # 오늘 시작 시각이 이미 지남 → 내일
        return today_start + timedelta(days=1)

    async def run(self) -> None:
        """스케줄러 메인 루프.

        1. 다음 batch_start_hour(KST)까지 대기
        2. vLLM 헬스체크 → 실패 시 스킵
        3. 배치 실행 (service가 batch_end_hour까지 처리하고 알아서 중단)
        4. 다음 날 batch_start_hour까지 대기 → 반복
        """
        logger.info(
            "커머스 배치 스케줄러 시작 (KST %d~%d시)",
            self.settings.batch_start_hour,
            self.settings.batch_end_hour,
        )

        while True:
            now = self._now_kst()
            next_start = self._next_batch_start(now)
            wait_seconds = (next_start - now).total_seconds()

            if wait_seconds > 0:
                logger.info(
                    "다음 배치: %s KST (%.0f분 후)",
                    next_start.strftime("%Y-%m-%d %H:%M"),
                    wait_seconds / 60,
                )
                await asyncio.sleep(wait_seconds)

            # 배치 시작
            await self._try_run_batch()

    async def _try_run_batch(self) -> None:
        """배치 실행을 시도한다. vLLM 서버가 꺼져 있으면 스킵."""
        now = self._now_kst()
        logger.info("배치 실행 시도 (KST %s)", now.strftime("%Y-%m-%d %H:%M"))

        if not await self.service.is_vllm_available():
            logger.warning("vLLM 서버 미응답 -> 오늘 배치 스킵")
            return

        try:
            result = await self.service.run_batch()
            logger.info(
                "배치 결과: fetched=%d, skipped=%d, analyzed=%d, "
                "upserted=%d, failed=%d, duration=%.0fs",
                result.total_fetched,
                result.total_skipped,
                result.total_analyzed,
                result.total_upserted,
                result.total_failed,
                result.duration_sec,
            )
        except Exception as e:
            logger.error("배치 실행 실패: %s", e, exc_info=True)
