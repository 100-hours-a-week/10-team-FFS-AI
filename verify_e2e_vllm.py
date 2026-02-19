"""E2E 테스트: ModelServerAnalyzer → GCP vLLM 서버 (다중 이미지)

로컬 이미지 10개를 GCP L4의 vLLM 서버(Qwen2.5-VL-7B)로 보내
카테고리 분류가 잘 되는지 검증합니다.

사용법:
  cd Project
  venv/bin/python verify_e2e_vllm.py
"""

import asyncio
import base64
import json
import re
import time
from pathlib import Path

import httpx

BASE_URL = "http://35.193.147.17:8001"
IMAGE_DIR = Path("/Users/ijeonglim/Downloads/test_runpod/image_data")

# 테스트할 이미지 10장 선택 (다양한 종류)
TEST_IMAGES = [
    "image1.jpg",
    "image2.png",
    "image3.jpg",
    "image4.jpg",
    "image5.png",
    "image6.png",
    "image7.png",
    "image8.png",
    "image9.png",
    "image10.png",
]

PROMPT = """너는 글로벌 패션 매거진의 에디터이자 15년 경력의 베테랑 패션 MD야.
제공된 이미지를 전문가의 시각으로 정밀 분석하되, 반드시 약속된 JSON 구조 내에서만 응답해.

반드시 아래 JSON 구조를 유지하고, 다른 설명 없이 JSON 데이터만 출력해:
{
  "major": {
    "category": "TOP, BOTTOM, DRESS, SHOES, ACCESSORY, ETC 중 택1",
    "color": ["구체적 색상명"],
    "material": ["소재"],
    "style_tags": ["스타일 키워드"]
  },
  "extra": {
    "meta_data": {
        "gender": "남성, 여성, 유니섹스 중 택1",
        "season": ["계절"],
        "formality": "격식 수준",
        "fit": "핏",
        "occasion": ["적합한 상황"]
    },
    "caption": "이미지 설명 2문장"
  }
}"""


async def analyze_image(image_bytes: bytes) -> tuple[dict, float]:
    """vLLM 서버에 이미지 분석을 요청하고 결과와 소요시간을 반환."""
    image_b64 = base64.b64encode(image_bytes).decode()
    content_type = "image/png" if image_bytes[:4] == b"\x89PNG" else "image/jpeg"

    payload = {
        "model": "skt/A.X-4.0-VL-Light",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{content_type};base64,{image_b64}"},
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    start = time.time()
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{BASE_URL}/v1/chat/completions", json=payload)
        resp.raise_for_status()
    elapsed = time.time() - start

    content = resp.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        return json.loads(match.group()), elapsed
    return {"error": "JSON not found", "raw": content[:200]}, elapsed


async def main() -> None:
    print("=" * 70)
    print("   🚀 vLLM E2E 테스트 — GCP L4 (Qwen2.5-VL-7B)")
    print("=" * 70)

    # 서버 상태 확인
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/v1/models")
        model_id = resp.json()["data"][0]["id"]
        print(f"\n✅ 서버 연결 OK — 모델: {model_id}\n")

    results = []
    for i, filename in enumerate(TEST_IMAGES, 1):
        filepath = IMAGE_DIR / filename
        if not filepath.exists():
            print(f"[{i:2d}/10] {filename:15s} ... ❌ 파일 없음 (skip)")
            continue

        image_bytes = filepath.read_bytes()
        size_kb = len(image_bytes) / 1024
        print(f"[{i:2d}/10] {filename:15s} ({size_kb:.0f}KB) ... ", end="", flush=True)

        try:
            parsed, elapsed = await analyze_image(image_bytes)

            if "error" in parsed:
                print(f"❌ JSON 파싱 실패 ({elapsed:.1f}s)")
                results.append(
                    {"file": filename, "status": "PARSE_ERROR", "time": elapsed}
                )
            else:
                category = parsed.get("major", {}).get("category", "N/A")
                colors = parsed.get("major", {}).get("color", [])
                material = parsed.get("major", {}).get("material", [])
                caption = parsed.get("extra", {}).get("caption", "")[:60]
                print(
                    f"✅ {category:10s} | color: {colors} | material: {material} ({elapsed:.1f}s)"
                )
                print(f"         caption: {caption}...")
                results.append(
                    {
                        "file": filename,
                        "category": category,
                        "colors": colors,
                        "time": elapsed,
                        "status": "OK",
                        "full_response": parsed,
                    }
                )
        except Exception as e:
            print(f"❌ 에러: {e}")
            results.append({"file": filename, "status": "ERROR", "error": str(e)})

    # 결과 요약
    print("\n" + "=" * 70)
    print("   📊 결과 요약")
    print("=" * 70)
    ok = [r for r in results if r.get("status") == "OK"]
    errors = [r for r in results if r.get("status") in ("ERROR", "PARSE_ERROR")]
    times = [r["time"] for r in ok]

    print(f"   총 테스트: {len(results)}")
    print(f"   ✅ 성공: {len(ok)}  |  ❌ 실패: {len(errors)}")
    if times:
        print(
            f"   ⏱  평균 응답: {sum(times)/len(times):.1f}s  |  최대: {max(times):.1f}s  |  최소: {min(times):.1f}s"
        )

    if ok:
        print("\n   카테고리 분포:")
        from collections import Counter

        for cat, cnt in Counter(r["category"] for r in ok).most_common():
            print(f"     {cat}: {cnt}개")

    print("=" * 70)

    # 상세 결과를 JSON 파일로 저장
    detail_path = "/tmp/e2e_vllm_details.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(
            [r for r in results if r.get("status") == "OK"],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n💾 상세 결과 저장: {detail_path}")


if __name__ == "__main__":
    asyncio.run(main())
