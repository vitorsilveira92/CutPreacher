from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Callable

from .models import Segment, Transcript, Word


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _preload_cuda_libs() -> None:
    """ctranslate2 needs cuDNN 9 / cuBLAS 12 on the loader path; preload them from the nvidia pip wheels if installed,
    since a missing cuDNN aborts the process instead of raising."""
    for module in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            pkg = __import__(module, fromlist=["__path__"])
        except ImportError:
            continue
        for lib_dir in (Path(p) / "lib" for p in pkg.__path__):
            for lib in sorted(lib_dir.glob("lib*.so*")):
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _run(media_path: Path, language: str, device: str, model_name: str, log: Callable[[str], None]) -> Transcript:
    from faster_whisper import WhisperModel

    log(f"Carregando Whisper '{model_name}' ({'GPU' if device == 'cuda' else 'CPU'})...")
    model = WhisperModel(model_name, device=device, compute_type="float16" if device == "cuda" else "int8")
    segments_iter, info = model.transcribe(
        str(media_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    segments: list[Segment] = []
    last_report = 0.0
    for seg in segments_iter:
        words = [
            Word(start=round(w.start, 3), end=round(w.end, 3), text=w.word.strip())
            for w in (seg.words or [])
            if w.word.strip()
        ]
        segments.append(Segment(start=round(seg.start, 3), end=round(seg.end, 3), text=seg.text.strip(), words=words))
        if info.duration and seg.end - last_report >= 120:
            last_report = seg.end
            log(f"  transcrito {seg.end / 60:.0f} de {info.duration / 60:.0f} min")
    return Transcript(language=info.language, duration=info.duration, segments=segments)


def transcribe(
    media_path: Path,
    cache_path: Path,
    language: str = "pt",
    model_name: str | None = None,
    log: Callable[[str], None] = print,
) -> Transcript:
    if cache_path.is_file():
        log("Transcrição já existe, reaproveitando.")
        return Transcript.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))

    transcript = None
    if _cuda_available():
        _preload_cuda_libs()
        try:
            transcript = _run(media_path, language, "cuda", model_name or "large-v3-turbo", log)
        except Exception as exc:
            log(f"Falha ao usar a GPU ({exc}); continuando na CPU.")
    if transcript is None:
        log("Sem GPU: a transcrição pode levar 20-40 min para uma pregação de 1 hora.")
        transcript = _run(media_path, language, "cpu", model_name or "small", log)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False), encoding="utf-8")
    return transcript
