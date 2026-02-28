import asyncio
import logging
import uuid
import time
import httpx

import os

# API 설정 (EC2 환경 등 환경 변수로 오버라이드 가능, 기본값은 로컬 서버)
API_BASE_URL = os.environ.get("KLOSETLAB_API_URL", "http://localhost:8000/ai/v1/closet/outfit")
TIMEOUT = 60.0

logger = logging.getLogger(__name__)

# 테스트용 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 시나리오 설정
SCENARIOS = [
    {
        "name": "Scenario 1: 옷장 아이템이 적은 사용자",
        "user_id": 999,  # DB에 아이템이 거의 없는 가상의 유저 가정
        "query": "오늘 입을 옷 추천해줘",
        "iterations": 1,
    },
    {
        "name": "Scenario 2: 모호한 쿼리",
        "user_id": 102,
        "query": "아무거나 추천해줘 편하게 입고 싶어",
        "iterations": 1,
    },
    {
        "name": "Scenario 3: 계절 불일치 쿼리 (Phase 1 한계 체크)",
        "user_id": 102,
        "query": "한여름 해변에서 입을 두꺼운 겨울 패딩 코디",
        "iterations": 1,
    },
    {
        "name": "Scenario 4: 반복 쿼리 (응답 다양성 체크)",
        "user_id": 102,
        "query": "중요한 결혼식 하객룩 추천해줘",
        "iterations": 3,
    },
]

async def call_outfit_api(client: httpx.AsyncClient, user_id: int, query: str) -> None:
    session_id = f"test-session-{uuid.uuid4().hex[:8]}"
    payload = {
        "user_id": user_id,
        "query": query,
        "session_id": session_id,
        # 날씨나 urls 정보는 필요 시 추가 가능 (현재 Phase 1 검증에는 필수 아님)
    }
    
    logger.info(f"요청 전송: user_id={user_id}, query='{query}'")
    start_time = time.time()
    
    try:
        response = await client.post(API_BASE_URL, json=payload)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            outfit_count = len(data.get("outfits", []))
            query_summary = data.get("query_summary", "")
            logger.info(f"성공 ({elapsed:.2f}s) | 반환된 코디 수: {outfit_count}세트 | 요약: {query_summary}")
        else:
            logger.error(f"실패 ({elapsed:.2f}s) | Status: {response.status_code} | Body: {response.text}")
            
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"예외 발생 ({elapsed:.2f}s) | Error: {e}")

async def run_tests():
    logger.info("=== LangGraph Phase 1 평가 스크립트 시작 ===")
    logger.info(f"Target API Server: {API_BASE_URL}")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for scenario in SCENARIOS:
            logger.info(f"\n--- {scenario['name']} ---")
            for i in range(scenario['iterations']):
                if scenario['iterations'] > 1:
                    logger.info(f" > Iteration {i+1}/{scenario['iterations']}")
                await call_outfit_api(
                    client,
                    user_id=scenario["user_id"],
                    query=scenario["query"]
                )
                # 약간의 딜레이
                await asyncio.sleep(1)

    logger.info("\n=== LangGraph Phase 1 평가 스크립트 완료 ===")
    logger.info("이제 Langfuse 대시보드에 접속하여 다음 사항을 확인하세요:")
    logger.info("1. vector_search 노드의 category_coverage (0건인 카테고리 확인)")
    logger.info("2. outfit_recommend 트레이스의 outfit_count (3세트 미만 반환 비율)")
    logger.info("3. tpo_extract, vector_search, outfit_compose 노드의 평균 실행 시간 (Gantt)")

if __name__ == "__main__":
    asyncio.run(run_tests())
