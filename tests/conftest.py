import pytest

from mock_server import MockVLLMServer


@pytest.fixture()
def mock_server():
    server = MockVLLMServer(token_delay=0.001).start()
    yield server
    server.stop()
