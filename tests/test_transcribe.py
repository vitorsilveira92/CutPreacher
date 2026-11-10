import sys
import types
from types import SimpleNamespace as NS

from cutpreacher import transcribe as tr


class FakeWhisperModel:
    created = []

    def __init__(self, name, device, compute_type):
        self.device = device
        FakeWhisperModel.created.append((name, device, compute_type))

    def transcribe(self, path, **kwargs):
        assert kwargs["word_timestamps"] and kwargs["language"] == "pt"
        if self.device == "cuda":
            raise RuntimeError("Library libcublas.so.12 is not found")
        words = [NS(start=0.0, end=0.4, word=" Deus"), NS(start=0.5, end=0.9, word=" é"), NS(start=1.0, end=1.5, word=" fiel.")]
        segs = iter([NS(start=0.0, end=1.5, text=" Deus é fiel.", words=words)])
        return segs, NS(duration=1.5, language="pt")


def test_gpu_failure_falls_back_to_cpu_and_caches(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setattr(tr, "_cuda_available", lambda: True)
    FakeWhisperModel.created.clear()
    cache = tmp_path / "t.json"

    t = tr.transcribe(tmp_path / "a.wav", cache, log=lambda _: None)
    assert [w.text for w in t.words] == ["Deus", "é", "fiel."]
    assert t.segments[0].text == "Deus é fiel."
    assert FakeWhisperModel.created == [("large-v3-turbo", "cuda", "float16"), ("small", "cpu", "int8")]
    assert cache.is_file()

    FakeWhisperModel.created.clear()
    again = tr.transcribe(tmp_path / "a.wav", cache, log=lambda _: None)
    assert FakeWhisperModel.created == []
    assert again.to_dict() == t.to_dict()
