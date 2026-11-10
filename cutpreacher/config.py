from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    num_clips: int = 3
    min_clip_seconds: float = 20.0
    max_clip_seconds: float = 59.0
    language: str = "pt"
    # "rosto" segue o rosto do pregador, "centro" corta no meio, "desfocado" mostra o quadro inteiro com fundo desfocado
    framing: str = "rosto"
    whisper_model: str | None = None
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "high"
    width: int = 1080
    height: int = 1920
    font_name: str = "Montserrat ExtraBold"
    fonts_dir: str | None = None
    cookies_file: str | None = None
