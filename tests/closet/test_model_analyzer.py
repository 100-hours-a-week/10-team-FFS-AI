import io
import json
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from PIL import Image

from app.closet.model_analyzer import ModelServerAnalyzer
from app.common.llm_schemas import ImageAnalysisResult


def _make_image(width: int = 100, height: int = 100, mode: str = "RGB") -> bytes:
    """테스트용 이미지 bytes 생성."""
    img = Image.new(mode, (width, height), color="red")
    buf = io.BytesIO()
    fmt = "PNG" if mode == "RGBA" else "JPEG"
    img.save(buf, format=fmt)
    return buf.getvalue()


VALID_RESPONSE_JSON = json.dumps(
    {
        "major": {
            "category": "TOP",
            "color": ["흰색"],
            "material": ["면"],
            "style_tags": ["캐주얼"],
        },
        "extra": {
            "meta_data": {
                "gender": "유니섹스",
                "season": ["봄", "가을"],
                "formality": "캐주얼",
                "fit": "레귤러핏",
                "occasion": ["데이트"],
            },
            "caption": "깔끔한 흰색 면 셔츠",
        },
    }
)


def _vllm_response(content: str) -> dict:
    """vLLM OpenAI 호환 응답 형태."""
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def mock_settings() -> Generator[MagicMock, Any, None]:
    with patch("app.closet.model_analyzer.get_settings") as mock:
        settings = MagicMock()
        settings.ai_server_url = "http://localhost:8001"
        settings.ai_model_name = "test-model"
        mock.return_value = settings
        yield settings


# ── 초기화 테스트 ──


def test_init_uses_settings(mock_settings: MagicMock) -> None:
    analyzer = ModelServerAnalyzer()
    assert analyzer._base_url == "http://localhost:8001"
    assert analyzer._model == "test-model"


def test_init_custom_url(mock_settings: MagicMock) -> None:
    analyzer = ModelServerAnalyzer(base_url="http://custom:9000")
    assert analyzer._base_url == "http://custom:9000"


# ── analyze_image 성공 테스트 ──


@pytest.mark.asyncio
async def test_analyze_image_success(mock_settings: MagicMock) -> None:
    """정상 응답 시 ImageAnalysisResult를 반환하는지 검증."""
    mock_response = MagicMock()
    mock_response.json.return_value = _vllm_response(VALID_RESPONSE_JSON)
    mock_response.raise_for_status = MagicMock()

    with patch("app.closet.model_analyzer.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        analyzer = ModelServerAnalyzer()
        result = await analyzer.analyze_image(_make_image())

    assert isinstance(result, ImageAnalysisResult)
    assert result.major.category == "TOP"
    assert result.major.color == ["흰색"]
    assert result.extra.caption == "깔끔한 흰색 면 셔츠"
    assert result.extra.meta_data.gender == "유니섹스"


@pytest.mark.asyncio
async def test_analyze_image_with_code_block(mock_settings: MagicMock) -> None:
    """```json 코드 블록으로 감싸진 응답도 파싱."""
    content = f"여기 분석 결과입니다:\n```json\n{VALID_RESPONSE_JSON}\n```"
    mock_response = MagicMock()
    mock_response.json.return_value = _vllm_response(content)
    mock_response.raise_for_status = MagicMock()

    with patch("app.closet.model_analyzer.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        analyzer = ModelServerAnalyzer()
        result = await analyzer.analyze_image(_make_image())

    assert result.major.category == "TOP"


@pytest.mark.asyncio
async def test_analyze_image_rgba_conversion(mock_settings: MagicMock) -> None:
    """RGBA 이미지가 RGB로 변환되어 처리되는지 검증."""
    mock_response = MagicMock()
    mock_response.json.return_value = _vllm_response(VALID_RESPONSE_JSON)
    mock_response.raise_for_status = MagicMock()

    with patch("app.closet.model_analyzer.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        analyzer = ModelServerAnalyzer()
        result = await analyzer.analyze_image(_make_image(mode="RGBA"))

    assert isinstance(result, ImageAnalysisResult)


# ── 에러 처리 테스트 ──


@pytest.mark.asyncio
async def test_analyze_image_timeout(mock_settings: MagicMock) -> None:
    """타임아웃 시 RuntimeError 발생."""
    with patch("app.closet.model_analyzer.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        analyzer = ModelServerAnalyzer()
        with pytest.raises(RuntimeError, match="타임아웃"):
            await analyzer.analyze_image(_make_image())


@pytest.mark.asyncio
async def test_analyze_image_connect_error(mock_settings: MagicMock) -> None:
    """연결 실패 시 RuntimeError 발생."""
    with patch("app.closet.model_analyzer.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        analyzer = ModelServerAnalyzer()
        with pytest.raises(RuntimeError, match="연결 불가"):
            await analyzer.analyze_image(_make_image())


@pytest.mark.asyncio
async def test_analyze_image_http_error(mock_settings: MagicMock) -> None:
    """HTTP 500 에러 시 RuntimeError 발생."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch("app.closet.model_analyzer.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "error", request=MagicMock(), response=mock_response
            )
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        analyzer = ModelServerAnalyzer()
        with pytest.raises(RuntimeError, match="vLLM HTTP 500"):
            await analyzer.analyze_image(_make_image())


# ── _parse_json 테스트 ──


def test_parse_json_raw_json() -> None:
    """순수 JSON 문자열 파싱."""
    text = '{"major": {"category": "TOP"}, "extra": {}}'
    result = ModelServerAnalyzer._parse_json(text)
    assert result["major"]["category"] == "TOP"


def test_parse_json_code_block() -> None:
    """```json 코드 블록 파싱."""
    text = '```json\n{"major": {"category": "BOTTOM"}, "extra": {}}\n```'
    result = ModelServerAnalyzer._parse_json(text)
    assert result["major"]["category"] == "BOTTOM"


def test_parse_json_invalid_code_block_fallback() -> None:
    """코드 블록 내 잘못된 JSON → 기본값 폴백."""
    text = "```json\nnot valid json\n```"
    result = ModelServerAnalyzer._parse_json(text)
    assert result["major"]["category"] == "ETC"


def test_parse_json_no_json_fallback() -> None:
    """JSON이 아예 없으면 기본값 반환."""
    result = ModelServerAnalyzer._parse_json("이건 JSON이 아닙니다")
    assert result["major"]["category"] == "ETC"


def test_parse_json_empty_string() -> None:
    """빈 문자열 → 기본값."""
    result = ModelServerAnalyzer._parse_json("")
    assert result["major"]["category"] == "ETC"
    assert result["extra"]["meta_data"] == {}


# ── _normalize_category 테스트 ──


def test_normalize_category_valid() -> None:
    assert ModelServerAnalyzer._normalize_category("TOP") == "TOP"
    assert ModelServerAnalyzer._normalize_category("BOTTOM") == "BOTTOM"
    assert ModelServerAnalyzer._normalize_category("SHOES") == "SHOES"


def test_normalize_category_mapped() -> None:
    assert ModelServerAnalyzer._normalize_category("JACKET") == "TOP"
    assert ModelServerAnalyzer._normalize_category("SNEAKERS") == "SHOES"
    assert ModelServerAnalyzer._normalize_category("JEANS") == "BOTTOM"
    assert ModelServerAnalyzer._normalize_category("BAG") == "ACCESSORY"


def test_normalize_category_case_insensitive() -> None:
    assert ModelServerAnalyzer._normalize_category("top") == "TOP"
    assert ModelServerAnalyzer._normalize_category("Jacket") == "TOP"


def test_normalize_category_unknown() -> None:
    assert ModelServerAnalyzer._normalize_category("ALIEN_WEAR") == "ETC"


def test_normalize_category_empty() -> None:
    assert ModelServerAnalyzer._normalize_category("") == "ETC"


# ── _to_list 테스트 ──


def test_to_list_none() -> None:
    assert ModelServerAnalyzer._to_list(None) == []


def test_to_list_string() -> None:
    assert ModelServerAnalyzer._to_list("검정") == ["검정"]


def test_to_list_list() -> None:
    assert ModelServerAnalyzer._to_list(["검정", "흰색"]) == ["검정", "흰색"]


def test_to_list_other() -> None:
    assert ModelServerAnalyzer._to_list(42) == ["42"]


# ── _to_result 테스트 ──


def test_to_result() -> None:
    data = {
        "major": {
            "category": "SHOES",
            "color": ["검정"],
            "material": ["가죽"],
            "style_tags": ["포멀"],
        },
        "extra": {
            "meta_data": {
                "gender": "남성",
                "season": ["가을", "겨울"],
                "formality": "포멀",
                "fit": "레귤러핏",
                "occasion": ["출근"],
            },
            "caption": "검정 가죽 구두",
        },
    }
    result = ModelServerAnalyzer._to_result(data)
    assert isinstance(result, ImageAnalysisResult)
    assert result.major.category == "SHOES"
    assert result.extra.meta_data.gender == "남성"


def test_to_result_missing_meta() -> None:
    data = {
        "major": {"category": "ETC"},
        "extra": {"caption": "알 수 없는 아이템"},
    }
    result = ModelServerAnalyzer._to_result(data)
    assert result.major.category == "ETC"
    assert result.extra.meta_data.gender is None


# ── _normalize 테스트 ──


def test_normalize_full() -> None:
    data = {
        "major": {
            "category": "JACKET",
            "color": "검정",
            "material": ["가죽"],
            "style_tags": None,
        },
        "extra": {
            "meta_data": {
                "gender": "남성",
                "season": "겨울",
                "formality": "캐주얼",
                "fit": None,
                "occasion": None,
            },
            "caption": "가죽 자켓",
        },
    }
    result = ModelServerAnalyzer._normalize(data)
    assert result["major"]["category"] == "TOP"  # JACKET → TOP
    assert result["major"]["color"] == ["검정"]  # str → list
    assert result["major"]["style_tags"] == []  # None → []
    assert result["extra"]["meta_data"]["season"] == ["겨울"]  # str → list


def test_normalize_empty_data() -> None:
    result = ModelServerAnalyzer._normalize({})
    assert result["major"]["category"] == "ETC"
    assert result["extra"]["caption"] == "의류 아이템"
