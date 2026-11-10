from types import SimpleNamespace

import pytest
from fakes import FakeClient, proposal

from cutpreacher.captions import build_ass, clip_words, format_time, group_words
from cutpreacher.config import Settings
from cutpreacher.download import DownloadError, fetch_video, is_youtube_url
from cutpreacher.highlights import CLIPS_SCHEMA, HighlightError, select_clips, snap_to_words
from cutpreacher.models import Segment, Transcript, Word
from cutpreacher.pipeline import slugify
from cutpreacher.video import crop_box


def make_words(n: int, step: float = 0.5, punct_every: int = 10) -> list[Word]:
    return [
        Word(start=i * step, end=i * step + step * 0.8, text=f"palavra{i}" + ("." if (i + 1) % punct_every == 0 else ""))
        for i in range(n)
    ]


def make_transcript(words: list[Word]) -> Transcript:
    segs = []
    for i in range(0, len(words), 10):
        chunk = words[i : i + 10]
        segs.append(Segment(chunk[0].start, chunk[-1].end, " ".join(w.text for w in chunk), chunk))
    return Transcript(language="pt", duration=words[-1].end, segments=segs)


# --- captions ---

def test_format_time():
    assert format_time(0) == "0:00:00.00"
    assert format_time(61.234) == "0:01:01.23"
    assert format_time(3723.5) == "1:02:03.50"


def test_clip_words_relative_and_cleaned():
    words = [Word(9.0, 9.5, "antes"), Word(10.2, 10.6, "Deus{x}"), Word(11.0, 11.4, "é"), Word(20.5, 21.0, "depois")]
    out = clip_words(words, 10.0, 20.0)
    assert [w.text for w in out] == ["DEUS(X)", "É"]
    assert out[0].start == pytest.approx(0.2)


def test_group_words_limits():
    words = [Word(i * 0.3, i * 0.3 + 0.25, t) for i, t in enumerate("o senhor é meu pastor, nada me faltará".split())]
    groups = group_words(words)
    assert all(len(g) <= 3 for g in groups)
    assert groups[1][-1].text == "pastor,"  # breaks on punctuation


def test_group_words_breaks_on_pause():
    words = [Word(0, 0.3, "a"), Word(0.35, 0.6, "b"), Word(2.0, 2.3, "c")]
    assert [len(g) for g in group_words(words)] == [2, 1]


def test_build_ass_highlights_each_word():
    words = [Word(0.0, 0.4, "JESUS"), Word(0.45, 0.8, "TE"), Word(0.85, 1.2, "AMA")]
    ass = build_ass(words, 1080, 1920, "Font")
    events = [line for line in ass.splitlines() if line.startswith("Dialogue")]
    assert len(events) == 3
    assert r"{\c&H00E5FF&}JESUS" in events[0]
    assert r"{\c&H00E5FF&}AMA" in events[2]
    assert "PlayResY: 1920" in ass


# --- clip boundaries ---

def test_snap_aligns_to_words():
    words = make_words(200)
    start, end = snap_to_words(words, 10.1, 40.3, 20, 59)
    assert start == pytest.approx(10.0 - 0.15)
    assert 39.5 < end < 41


def test_snap_trims_long_clip_to_sentence_end():
    words = make_words(400)
    start, end = snap_to_words(words, 0, 150, 20, 59)
    assert end - start <= 59
    last_word = max((w for w in words if w.end <= end), key=lambda w: w.end)
    assert last_word.text.endswith(".")


def test_snap_rejects_short():
    assert snap_to_words(make_words(200), 10, 15, 20, 59) is None


# --- AI selection ---

def test_select_clips_filters_and_retries():
    transcript = make_transcript(make_words(600))
    client = FakeClient([
        [proposal(10, 50, "A"), proposal(30, 70, "overlap"), proposal(100, 105, "curto"), proposal(120, 160, "B")],
        [proposal(200, 250, "C"), proposal(260, 300, "D")],
    ])
    clips = select_clips(transcript, Settings(num_clips=3), client=client, log=lambda _: None)
    assert [c.title for c in clips] == ["A", "B", "C"]
    assert clips[0].hashtags == ["#fé", "#jesus"]
    assert len(client.calls) == 2
    second_prompt = client.calls[1]["messages"][0]["content"]
    assert "NÃO os repita" in second_prompt
    req = client.calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"]["format"]["schema"] == CLIPS_SCHEMA
    assert "thinking" not in req and "temperature" not in req
    for c in clips:
        assert 20 <= c.duration <= 59


def test_select_clips_refusal():
    transcript = make_transcript(make_words(100))
    refusal = SimpleNamespace(stop_reason="refusal", content=[])
    with pytest.raises(HighlightError):
        select_clips(transcript, Settings(), client=FakeClient([refusal]), log=lambda _: None)


def test_schema_is_strict():
    item = CLIPS_SCHEMA["properties"]["clips"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])


# --- framing ---

def test_crop_box_landscape_centered():
    assert crop_box(1920, 1080, 1080, 1920) == (608, 1080, 656, 0)


def test_crop_box_follows_face_and_clamps():
    w, h, x, y = crop_box(1920, 1080, 1080, 1920, center_x=1900)
    assert x + w <= 1920 and x == 1920 - 608
    _, _, x0, _ = crop_box(1920, 1080, 1080, 1920, center_x=10)
    assert x0 == 0


def test_crop_box_vertical_source():
    w, h, x, y = crop_box(1080, 2400, 1080, 1920)
    assert (w, h, x) == (1080, 1920, 0) and y == 240


# --- misc ---

def test_slugify():
    assert slugify("Fé que Move Montanhas! | Pr. João") == "fe-que-move-montanhas-pr-joao"
    assert slugify("???") == "video"


def test_youtube_detection_and_missing_file(tmp_path):
    assert is_youtube_url("https://www.youtube.com/watch?v=abc")
    assert is_youtube_url("https://youtu.be/abc")
    assert not is_youtube_url("/content/drive/MyDrive/x.mp4")
    with pytest.raises(DownloadError):
        fetch_video(str(tmp_path / "nao_existe.mp4"), tmp_path)
