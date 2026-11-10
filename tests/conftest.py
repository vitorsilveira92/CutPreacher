import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

SERMON = (
    "Irmãos, hoje eu quero falar sobre esperança. Quantas vezes você olhou para a sua vida e achou que não havia saída? "
    "A Bíblia diz que o choro pode durar uma noite, mas a alegria vem pela manhã. Deus não se esqueceu de você. "
    "Davi era o menor da casa do seu pai, e ninguém se lembrou dele. Mas Deus olhou para o coração. "
    "Não importa o que disseram sobre você, importa o que Deus diz sobre você. "
    "A fé não é ausência de medo, a fé é continuar andando mesmo com medo. "
    "Pedro andou sobre as águas enquanto olhava para Jesus. Quando olhou para o vento, começou a afundar. "
    "Para onde você está olhando hoje? Para o problema ou para aquele que é maior que o problema? "
    "O mesmo Deus que abriu o mar vermelho é o Deus que vai abrir caminho na sua vida. "
)


@pytest.fixture(scope="session")
def sermon_video(tmp_path_factory) -> Path:
    """~3 min 1920x1080 'sermon': Portuguese TTS speech plus a real face photo on the right side of the frame."""
    if not shutil.which("espeak-ng"):
        pytest.skip("espeak-ng não instalado")
    from skimage import data
    import cv2

    d = tmp_path_factory.mktemp("sermon")
    cv2.imwrite(str(d / "face.png"), cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR))
    subprocess.run(
        ["espeak-ng", "-v", "pt-br", "-s", "140", "-w", str(d / "speech.wav"), SERMON * 3],
        check=True,
    )
    out = d / "pregacao.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=0x1a2340:s=1920x1080:r=30",
            "-loop", "1", "-i", str(d / "face.png"),
            "-i", str(d / "speech.wav"),
            "-filter_complex", "[1:v]scale=520:-1[f];[0:v][f]overlay=1300:250[v]",
            "-map", "[v]", "-map", "2:a", "-shortest",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out),
        ],
        check=True,
    )
    return out
