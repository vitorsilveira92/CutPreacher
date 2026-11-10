import json

import pytest
from fakes import FakeClient, proposal

from cutpreacher import pipeline
from cutpreacher.captions import build_ass
from cutpreacher.config import Settings
from cutpreacher.models import Segment, Transcript, Word
from cutpreacher.video import detect_speaker_x, probe, render_clip

from conftest import SERMON


def fake_transcript(duration: float) -> Transcript:
    """Spreads the known sermon text evenly over the audio (the real Whisper model can't be downloaded here)."""
    tokens = (SERMON * 3).split()
    step = duration / len(tokens)
    words = [Word(round(i * step, 3), round(i * step + step * 0.85, 3), t) for i, t in enumerate(tokens)]
    segments, current = [], []
    for w in words:
        current.append(w)
        if w.text.endswith((".", "?")):
            segments.append(Segment(current[0].start, current[-1].end, " ".join(x.text for x in current), current))
            current = []
    if current:
        segments.append(Segment(current[0].start, current[-1].end, " ".join(x.text for x in current), current))
    return Transcript("pt", duration, segments)


def test_face_detection_finds_speaker(sermon_video):
    x = detect_speaker_x(sermon_video, 5, 40)
    assert x is not None
    assert 1300 < x < 1820  # the photo is overlaid at x=1300..1820


def test_full_pipeline(sermon_video, tmp_path, monkeypatch):
    info = probe(sermon_video)
    assert info.duration > 150
    monkeypatch.setattr(pipeline, "transcribe", lambda *a, **k: fake_transcript(info.duration))

    client = FakeClient([[
        proposal(5, 45, "A esperança vem pela manhã"),
        proposal(50, 140, "Para onde você está olhando?"),  # too long: must be trimmed to <= 59s
        proposal(150, 185, "Deus abre caminho"),
    ]])
    fonts = tmp_path / "fontes"
    fonts.mkdir()
    (fonts / "DejaVuSans-Bold.ttf").write_bytes(open("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "rb").read())
    settings = Settings(num_clips=3, font_name="DejaVu Sans", fonts_dir=str(fonts))

    logs = []
    result = pipeline.run(str(sermon_video), tmp_path / "saida", settings, work_root=tmp_path / "work", client=client, log=logs.append)

    assert len(result.files) == 3
    for f, clip in zip(result.files, result.clips):
        out = probe(f)
        assert (out.width, out.height) == (1080, 1920)
        assert out.has_audio
        assert out.duration <= 59.6
        assert out.duration == pytest.approx(clip.duration, abs=0.2)
    assert result.files[0].name == "corte_01_a-esperanca-vem-pela-manha.mp4"
    texts = (result.output_dir / "textos_para_postar.txt").read_text(encoding="utf-8")
    assert "Deus abre caminho" in texts and "#fé" in texts
    meta = json.loads((result.output_dir / "cortes.json").read_text(encoding="utf-8"))
    assert len(meta["cortes"]) == 3
    assert any("Pronto" in line for line in logs)


@pytest.mark.parametrize("framing", ["centro", "desfocado"])
def test_other_framings(sermon_video, tmp_path, framing):
    info = probe(sermon_video)
    words = [Word(0.2, 0.6, "TESTE"), Word(0.7, 1.2, "DE"), Word(1.3, 1.9, "LEGENDA")]
    out = render_clip(
        sermon_video, info, 10, 13, build_ass(words, 1080, 1920, "DejaVu Sans"),
        tmp_path / f"{framing}.mp4", tmp_path / "work", 1080, 1920, framing, None,
    )
    p = probe(out)
    assert (p.width, p.height) == (1080, 1920)
    assert p.duration == pytest.approx(3, abs=0.2)
