from __future__ import annotations

import json
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


class VideoError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    has_audio: bool


def run_ffmpeg(args: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise VideoError(f"ffmpeg falhou: {proc.stderr.strip()[-2000:]}")


def probe(path: Path) -> VideoInfo:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise VideoError(f"Não foi possível ler o vídeo {path}: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise VideoError(f"O arquivo {path} não tem trilha de vídeo.")
    width, height = int(video["width"]), int(video["height"])
    rotation = abs(int(video.get("tags", {}).get("rotate", 0) or 0))
    for side in video.get("side_data_list", []):
        rotation = abs(int(side.get("rotation", rotation) or 0))
    if rotation in (90, 270):
        width, height = height, width
    return VideoInfo(
        width=width,
        height=height,
        duration=float(data["format"].get("duration", 0.0)),
        has_audio=any(s["codec_type"] == "audio" for s in data["streams"]),
    )


def extract_audio(video: Path, out: Path) -> Path:
    if not out.is_file():
        run_ffmpeg(["-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)])
    return out


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def crop_box(src_w: int, src_h: int, out_w: int, out_h: int, center_x: float | None = None) -> tuple[int, int, int, int]:
    """(w, h, x, y) of the largest out_w:out_h window, horizontally centred on center_x when given."""
    target = out_w / out_h
    if src_w / src_h > target:
        h = src_h - src_h % 2
        w = min(_even(h * target), src_w - src_w % 2)
        cx = src_w / 2 if center_x is None else center_x
        x = int(min(max(cx - w / 2, 0), src_w - w))
        return w, h, x - x % 2, 0
    w = src_w - src_w % 2
    h = min(_even(w / target), src_h - src_h % 2)
    y = (src_h - h) // 2
    return w, h, 0, y - y % 2


def detect_speaker_x(video: Path, start: float, end: float, samples: int = 15) -> float | None:
    """Median horizontal centre of the largest face over the clip, in source pixels."""
    import cv2

    if not hasattr(cv2, "CascadeClassifier"):  # removed from the main module in OpenCV 5
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    centers = []
    span = max(end - start - 1.0, 0.0)
    try:
        for i in range(samples):
            t = start + 0.5 + span * i / max(samples - 1, 1)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            scale = 960 / frame.shape[1] if frame.shape[1] > 960 else 1.0
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
            if len(faces):
                x, _, w, _ = max(faces, key=lambda f: f[2] * f[3])
                centers.append((x + w / 2) / scale)
    finally:
        cap.release()
    if len(centers) < max(2, samples // 5):
        return None
    return statistics.median(centers)


def build_filter(info: VideoInfo, out_w: int, out_h: int, framing: str, center_x: float | None, ass_name: str, fonts: bool) -> str:
    ass = f"ass={ass_name}" + (":fontsdir=fonts" if fonts else "")
    if framing == "desfocado":
        return (
            f"[0:v]split[a][b];"
            f"[a]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},boxblur=25:2[bg];"
            f"[b]scale={out_w}:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,{ass}[v]"
        )
    w, h, x, y = crop_box(info.width, info.height, out_w, out_h, center_x if framing == "rosto" else None)
    return f"[0:v]crop={w}:{h}:{x}:{y},scale={out_w}:{out_h},setsar=1,{ass}[v]"


def render_clip(
    video: Path,
    info: VideoInfo,
    start: float,
    end: float,
    ass_text: str,
    out_path: Path,
    work_dir: Path,
    out_w: int,
    out_h: int,
    framing: str,
    center_x: float | None,
    fonts_dir: str | None = None,
) -> Path:
    # ffmpeg filter paths are escaped awkwardly, so render inside work_dir with relative names.
    work_dir.mkdir(parents=True, exist_ok=True)
    ass_name = f"{out_path.stem}.ass"
    (work_dir / ass_name).write_text(ass_text, encoding="utf-8")
    has_fonts = False
    if fonts_dir and Path(fonts_dir).is_dir():
        shutil.copytree(fonts_dir, work_dir / "fonts", dirs_exist_ok=True)
        has_fonts = True

    graph = build_filter(info, out_w, out_h, framing, center_x, ass_name, has_fonts)
    args = ["-ss", f"{start:.3f}", "-i", str(video.resolve()), "-t", f"{end - start:.3f}", "-filter_complex"]
    if info.has_audio:
        graph += ";[0:a]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[a]"
        args += [graph, "-map", "[v]", "-map", "[a]", "-c:a", "aac", "-b:a", "160k"]
    else:
        args += [graph, "-map", "[v]"]
    tmp_out = work_dir / out_path.name
    args += [
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
        "-movflags", "+faststart", tmp_out.name,
    ]
    run_ffmpeg(args, cwd=work_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_out.resolve() != out_path.resolve():
        shutil.copyfile(tmp_out, out_path)
    return out_path
