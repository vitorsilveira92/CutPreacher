import json
from types import SimpleNamespace


class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.message


class FakeClient:
    """Stands in for anthropic.Anthropic(); each stream() call returns the next canned response."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        clips = self.responses.pop(0)
        if isinstance(clips, SimpleNamespace):
            return FakeStream(clips)
        text = SimpleNamespace(type="text", text=json.dumps({"clips": clips}))
        return FakeStream(SimpleNamespace(stop_reason="end_turn", content=[text]))


def proposal(start, end, title="Título"):
    return {
        "start_seconds": start, "end_seconds": end, "title": title, "hook": "h", "reason": "r",
        "description": "d", "hashtags": ["fé", "#jesus"],
    }
