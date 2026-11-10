from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from .captions import build_ass, clip_words
from .config import Settings
from .download import fetch_video
from .highlights import select_clips
from .models import Clip
from .transcribe import transcribe
from .video import detect_speaker_x, extract_audio, probe, render_clip


@dataclass
class Result:
    output_dir: Path
    clips: list[Clip]
    files: list[Path]


def slugify(text: str, max_len: int = 40) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:max_len].rstrip("-") or "video"


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def write_post_texts(path: Path, title: str, clips: list[Clip], files: list[Path]) -> None:
    lines = [f"Pregação: {title}", ""]
    for i, (clip, file) in enumerate(zip(clips, files), 1):
        lines += [
            f"=== Corte {i}: {clip.title} ===",
            f"Arquivo: {file.name}",
            f"Trecho original: {_fmt(clip.start)} a {_fmt(clip.end)} ({clip.duration:.0f}s)",
            f"Por que esse trecho: {clip.reason}",
            "",
            "Legenda para o post:",
            clip.description,
            "",
            " ".join(clip.hashtags),
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    source: str,
    output_root: Path,
    settings: Settings | None = None,
    work_root: Path | None = None,
    client=None,
    log: Callable[[str], None] = print,
) -> Result:
    settings = settings or Settings()
    t0 = time.time()
    key = hashlib.sha1(source.strip().encode()).hexdigest()[:12]
    work = (work_root or Path("/tmp/cutpreacher_work")) / key
    work.mkdir(parents=True, exist_ok=True)

    log("1/4 Obtendo o vídeo...")
    video, title = fetch_video(source, work / "download", settings.cookies_file)
    info = probe(video)
    log(f"   '{title}' — {info.duration / 60:.1f} min, {info.width}x{info.height}")

    log("2/4 Transcrevendo o áudio (essa é a etapa mais demorada)...")
    audio = extract_audio(video, work / "audio.wav")
    transcript = transcribe(audio, work / "transcricao.json", settings.language, settings.whisper_model, log)

    log("3/4 Escolhendo os melhores momentos com IA...")
    clips = select_clips(transcript, settings, client=client, log=log)
    for clip in clips:
        clip.end = min(clip.end, info.duration)

    out_dir = output_root / f"{date.today().isoformat()}_{slugify(title)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"4/4 Gerando {len(clips)} vídeos em 9:16 com legenda...")
    words = transcript.words
    files = []
    for i, clip in enumerate(clips, 1):
        name = f"corte_{i:02d}_{slugify(clip.title, 30)}.mp4"
        center_x = detect_speaker_x(video, clip.start, clip.end) if settings.framing == "rosto" else None
        ass = build_ass(clip_words(words, clip.start, clip.end), settings.width, settings.height, settings.font_name)
        render_clip(
            video, info, clip.start, clip.end, ass, out_dir / name, work / "render",
            settings.width, settings.height, settings.framing, center_x, settings.fonts_dir,
        )
        files.append(out_dir / name)
        log(f"   ✔ {name} ({clip.duration:.0f}s) — {clip.title}")

    write_post_texts(out_dir / "textos_para_postar.txt", title, clips, files)
    (out_dir / "cortes.json").write_text(
        json.dumps({"titulo": title, "fonte": source, "cortes": [asdict(c) for c in clips]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"Pronto em {(time.time() - t0) / 60:.1f} min. Arquivos em: {out_dir}")
    return Result(output_dir=out_dir, clips=clips, files=files)
