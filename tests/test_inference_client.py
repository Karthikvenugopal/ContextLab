import httpx
import pytest

from contextlab.agent.models import Message
from contextlab.config import ModelConfig
from contextlab.inference.client import InferenceError, OpenAIClient


@pytest.mark.asyncio
async def test_openai_compatible_response_is_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(
            200,
            json={
                "model": "mock",
                "choices": [{"message": {"content": '{"action":"finish"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            },
        )

    client = OpenAIClient(ModelConfig(endpoint="http://mock/v1", model="mock"))
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url="http://mock/v1/", transport=httpx.MockTransport(handler)
    )
    result = await client.complete([Message(role="user", content="go")])
    await client.aclose()
    assert result.usage.total_tokens == 14


@pytest.mark.asyncio
async def test_timeout_is_reported_after_bounded_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    client = OpenAIClient(ModelConfig(endpoint="http://mock/v1", model="mock", retries=0))
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url="http://mock/v1/", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(InferenceError, match="too slow"):
        await client.complete([Message(role="user", content="go")])
    await client.aclose()
