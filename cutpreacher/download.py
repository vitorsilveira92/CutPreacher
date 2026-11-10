from __future__ import annotations

import re
from pathlib import Path

YOUTUBE_RE = re.compile(r"^https?://(www\.|m\.)?(youtube\.com|youtu\.be)/", re.IGNORECASE)


class DownloadError(RuntimeError):
    pass


def is_youtube_url(source: str) -> bool:
    return bool(YOUTUBE_RE.match(source.strip()))


def fetch_video(source: str, work_dir: Path, cookies_file: str | None = None) -> tuple[Path, str]:
    """Returns (video_path, title). `source` is a YouTube URL or a local/Drive video path."""
    source = source.strip()
    if not is_youtube_url(source):
        path = Path(source).expanduser()
        if not path.is_file():
            raise DownloadError(f"Arquivo não encontrado: {source}")
        return path, path.stem

    import yt_dlp

    work_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(work_dir / "original.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
        # YouTube requires a JS runtime for its signature challenges; Colab ships node.
        "js_runtimes": {"deno": {}, "node": {}},
    }
    if cookies_file and Path(cookies_file).is_file():
        opts["cookiefile"] = cookies_file

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source, download=True)
            path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    except yt_dlp.utils.DownloadError as exc:
        raise DownloadError(
            "Não foi possível baixar o vídeo do YouTube. Se o erro mencionar 'Sign in to confirm' "
            "ou 'bot', o YouTube bloqueou o servidor do Google. Alternativas: (1) coloque o arquivo "
            "cookies.txt na pasta CutPreacher do Drive, ou (2) baixe o vídeo pelo YouTube Studio, "
            "envie para a pasta CutPreacher/entrada do Drive e informe o caminho do arquivo.\n"
            f"Detalhe: {exc}"
        ) from exc

    if not path.is_file():
        matches = sorted(work_dir.glob("original.*"))
        if not matches:
            raise DownloadError("Download terminou mas o arquivo de vídeo não foi encontrado.")
        path = matches[0]
    return path, info.get("title") or "pregacao"
