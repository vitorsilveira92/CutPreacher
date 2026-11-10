import argparse
from pathlib import Path

from .config import Settings
from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(prog="cutpreacher", description="Gera cortes 9:16 legendados de uma pregação.")
    parser.add_argument("source", help="Link do YouTube ou caminho de um arquivo de vídeo")
    parser.add_argument("-o", "--output", default="saida", help="Pasta de saída")
    parser.add_argument("-n", "--cortes", type=int, default=3)
    parser.add_argument("--max", type=float, default=59.0, help="Duração máxima de cada corte (s)")
    parser.add_argument("--enquadramento", choices=["rosto", "centro", "desfocado"], default="rosto")
    parser.add_argument("--whisper", default=None, help="Modelo Whisper (ex: small, medium, large-v3-turbo)")
    parser.add_argument("--cookies", default=None, help="cookies.txt do YouTube (opcional)")
    parser.add_argument("--fontes", default=None, help="Pasta com fontes .ttf para a legenda")
    args = parser.parse_args()

    settings = Settings(
        num_clips=args.cortes,
        max_clip_seconds=args.max,
        framing=args.enquadramento,
        whisper_model=args.whisper,
        cookies_file=args.cookies,
        fonts_dir=args.fontes,
    )
    run(args.source, Path(args.output), settings)


if __name__ == "__main__":
    main()
