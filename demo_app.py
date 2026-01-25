"""
Streamlit 데모 - 이미지 검증 API 테스트

실행: streamlit run demo_app.py
"""

import time

import httpx
import streamlit as st

# FastAPI 서버 URL
VALIDATE_API_URL = "http://localhost:8000/v1/closet/validate"

st.set_page_config(page_title="패션 이미지 검증 데모", page_icon="👔", layout="wide")

st.title("👔 패션 이미지 검증 API 데모")
st.markdown("---")

# 사이드바: 테스트 이미지 URL
st.sidebar.header("📸 테스트 이미지")
st.sidebar.markdown(
    """
**샘플 URL:**
- [패션 1](https://images.unsplash.com/photo-1434389677669-e08b4cac3105)
- [패션 2](https://images.unsplash.com/photo-1490481651871-ab68de25d43d)
- [음식](https://images.unsplash.com/photo-1504674900247-0877df9cc836)

**검증 항목:**
- 포맷 (jpg, png)
- 파일 크기 (10MB 이하)
- NSFW
- 패션 도메인
- 중복 (Qdrant)
- 품질 (512x512 이상)

**Note:**
임베딩은 analyze API 완료 시 자동 저장됨
(검증 단계에서는 저장 안 함)
"""
)

# 메인 UI
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("입력")
    user_id = st.number_input("User ID", min_value=1, value=123, step=1)

    # 이미지 URL 입력
    st.markdown("**이미지 URL** (한 줄에 하나씩)")
    image_urls_text = st.text_area(
        "이미지 URL 입력",
        value="https://images.unsplash.com/photo-1434389677669-e08b4cac3105",
        height=150,
        label_visibility="collapsed",
    )

    # URL 파싱
    image_urls = [url.strip() for url in image_urls_text.split("\n") if url.strip()]

    st.info(f"총 {len(image_urls)}개 이미지")

    # 검증 버튼
    if st.button("🚀 검증 시작", type="primary", use_container_width=True):
        if not image_urls:
            st.error("이미지 URL을 입력하세요!")
        else:
            with col2:
                st.subheader("결과")

                # API 호출
                with st.spinner("검증 중..."):
                    start_time = time.time()

                    try:
                        response = httpx.post(
                            VALIDATE_API_URL,
                            json={"userId": user_id, "images": image_urls},
                            timeout=60.0,
                        )
                        response.raise_for_status()
                        result = response.json()

                        elapsed_time = time.time() - start_time

                        # 성공
                        st.success(f"✅ 완료! (소요 시간: {elapsed_time:.2f}초)")

                        # 요약
                        summary = result["validationSummary"]
                        col_total, col_passed, col_failed = st.columns(3)
                        col_total.metric("전체", summary["total"])
                        col_passed.metric("통과", summary["passed"], delta=None)
                        col_failed.metric("실패", summary["failed"], delta=None)

                        st.markdown("---")

                        # 각 이미지 결과
                        for idx, item in enumerate(result["validationResults"], 1):
                            url = item["originUrl"]
                            passed = item["passed"]
                            error = item.get("error")
                            embedding = item.get("embedding", [])

                            # 상태 표시
                            if passed:
                                st.success(f"**#{idx} ✅ 통과**")
                            else:
                                st.error(f"**#{idx} ❌ 실패: {error}**")

                            # URL 표시 (축약)
                            short_url = url[:60] + "..." if len(url) > 60 else url
                            st.caption(f"URL: {short_url}")

                            # 이미지 미리보기
                            try:
                                st.image(url, width=200)
                            except Exception:
                                st.caption("(이미지 미리보기 실패)")

                            # 임베딩 정보
                            if embedding:
                                with st.expander(f"임베딩 벡터 ({len(embedding)}차원)"):
                                    st.text(str(embedding[:10]) + " ... (처음 10개)")
                                    st.caption(
                                        "Note: 이 임베딩은 중복 검증용이며, analyze 완료 시 Qdrant에 저장됩니다."
                                    )

                            st.markdown("---")

                    except httpx.HTTPStatusError as e:
                        st.error(f"❌ HTTP 에러: {e.response.status_code}")
                        st.json(e.response.json())
                    except Exception as e:
                        st.error(f"❌ 에러: {str(e)}")

# 하단 정보
st.markdown("---")
st.markdown(
    """
### 검증 항목
1. ✅ **포맷 검증**: jpg, png만 허용
2. ✅ **파일 크기**: 10MB 이하
3. ✅ **NSFW 검증**: 부적절한 이미지 차단
4. ✅ **패션 도메인**: 패션 아이템만 허용
5. ✅ **중복 검증**: 유사도 0.95 이상 차단 (userId 필터링)
6. ✅ **품질 검증**: 해상도 512x512 이상

### 실패 코드
- `INVALID_FORMAT`: 잘못된 포맷
- `FILE_TOO_LARGE`: 파일 크기 초과
- `NSFW`: 부적절한 콘텐츠
- `NOT_FASHION`: 패션 아이템 아님
- `DUPLICATE`: 중복/유사 이미지 (같은 userId 내)
- `TOO_BLURRY`: 품질 부족 (해상도)

### 아키텍처 노트
- **검증(validate)**: 임베딩 생성 → Qdrant 조회 (중복 체크) → 결과 반환
- **분석(analyze)**: 배경 제거 → AI 분석 → Qdrant 저장 (임베딩 + 메타데이터)
- 임베딩 저장은 analyze 완료 시 AI 서비스가 자동 처리
"""
)
