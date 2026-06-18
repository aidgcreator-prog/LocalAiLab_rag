from langchain_core.messages import HumanMessage
import asyncio

from agent_input_middleware import InputContentSanitizerMiddleware


def test_input_content_sanitizer_rewrites_file_blocks_in_human_message():
    middleware = InputContentSanitizerMiddleware()
    state = {
        "messages": [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Convert this PDF to text."},
                    {"type": "file", "filename": "scan.pdf"},
                ]
            )
        ]
    }

    result = middleware.before_agent(state=state, runtime=None)

    assert result is not None
    rewritten = result["messages"].value
    assert rewritten[0].content == "Convert this PDF to text.\n[Attached file: scan.pdf]"


class _FakeRequest:
    def __init__(self, messages, system_message=None):
        self.messages = messages
        self.system_message = system_message

    def override(self, **kwargs):
        return _FakeRequest(
            kwargs.get("messages", self.messages),
            kwargs.get("system_message", self.system_message),
        )


def test_wrap_model_call_sanitizes_request_messages_before_handler():
    middleware = InputContentSanitizerMiddleware()
    request = _FakeRequest(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Use attachment."},
                    {"type": "file", "filename": "scan.pdf"},
                ]
            )
        ]
    )

    def _handler(req):
        return req.messages[0].content

    result = middleware.wrap_model_call(request, _handler)

    assert result == "Use attachment.\n[Attached file: scan.pdf]"


def test_awrap_model_call_sanitizes_request_messages_before_handler():
    middleware = InputContentSanitizerMiddleware()
    request = _FakeRequest(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Use attachment."},
                    {"type": "file", "filename": "scan.pdf"},
                ]
            )
        ]
    )

    async def _handler(req):
        return req.messages[0].content

    result = asyncio.run(middleware.awrap_model_call(request, _handler))

    assert result == "Use attachment.\n[Attached file: scan.pdf]"
