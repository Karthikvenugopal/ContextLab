import threading
from http.server import ThreadingHTTPServer

import pytest

from contextlab.agent.models import Message
from contextlab.config import ModelConfig
from contextlab.inference.client import OpenAIClient
from contextlab.inference.mock_server import MockHandler


@pytest.mark.asyncio
async def test_mock_endpoint_supports_regular_and_streamed_openai_responses() -> None:
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    except PermissionError:
        pytest.skip("local socket binding is disabled by this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    client = OpenAIClient(ModelConfig(endpoint=endpoint, model="contextlab-mock"))
    try:
        regular = await client.complete([Message(role="user", content="go")])
        streamed = await client.stream([Message(role="user", content="go")])
        assert '"action": "finish"' in regular.content
        assert streamed.content == regular.content
        assert streamed.usage.total_tokens > 0
        assert len(streamed.inter_token_intervals_seconds) == 1
    finally:
        await client.aclose()
        server.shutdown()
        thread.join(timeout=2)
