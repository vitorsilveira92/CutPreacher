"""Generates notebook/CutPreacher.ipynb with the cutpreacher package embedded in it."""

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebook" / "CutPreacher.ipynb"

INTRO = """# ✂️ CutPreacher — cortes de pregação para Reels e Shorts

Gera automaticamente **cortes verticais (9:16) de até 1 minuto, já legendados**, a partir de uma pregação do YouTube.
Uma IA (Claude) lê a pregação inteira e escolhe os momentos mais fortes.

### Como usar (toda semana)
1. Menu **Ambiente de execução → Alterar o tipo de ambiente de execução → GPU T4** (deixa tudo ~5x mais rápido). Só precisa fazer uma vez.
2. Clique no ▶ da etapa **1️⃣ Preparar**. Na primeira vez o Google vai pedir permissão para acessar o seu Drive — aceite.
3. Na etapa **2️⃣ Gerar os cortes**, cole o link do YouTube e clique no ▶.
4. Os vídeos ficam no seu Google Drive, na pasta **CutPreacher/saida**. Abra o app do Google Drive no celular para baixar e postar.
   O arquivo `textos_para_postar.txt` traz um título, uma legenda e hashtags para cada corte.

### Configuração (só uma vez)
A IA precisa de uma chave da API da Anthropic:
1. Crie a chave em https://platform.claude.com/settings/keys (é cobrado por uso; cada pregação custa centavos de dólar).
2. Aqui no Colab, clique no ícone de **chave 🔑 (Secrets)** na barra da esquerda → **Adicionar novo secret**.
3. Nome: `ANTHROPIC_API_KEY` — Valor: a sua chave. Ative a opção **Acesso do notebook**.

### Se o YouTube bloquear o download
Às vezes o YouTube bloqueia downloads vindos dos servidores do Google. Nesse caso:
baixe a pregação pelo **YouTube Studio** do canal, envie o arquivo para a pasta **CutPreacher/entrada** do Drive
e preencha o campo `arquivo_no_drive` com o nome do arquivo (ex: `culto-domingo.mp4`)."""

SETUP = '''#@title 1️⃣ Preparar o sistema (clique no ▶ — uma vez por sessão)
import base64, io, os, subprocess, sys, urllib.request, zipfile
from pathlib import Path

print("Instalando dependências (1-2 min)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "faster-whisper", "yt-dlp[default]", "anthropic"], check=True)

PACKAGE_ZIP = "{package_zip}"
APP_DIR = Path("/content/cutpreacher_app")
zipfile.ZipFile(io.BytesIO(base64.b64decode(PACKAGE_ZIP))).extractall(APP_DIR)
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
for name in [m for m in sys.modules if m == "cutpreacher" or m.startswith("cutpreacher.")]:
    del sys.modules[name]

from google.colab import drive, userdata
drive.mount("/content/drive")
BASE = Path("/content/drive/MyDrive/CutPreacher")
for sub in ("entrada", "saida", "fontes"):
    (BASE / sub).mkdir(parents=True, exist_ok=True)

FONT_NAME = "Montserrat ExtraBold"
font_file = BASE / "fontes" / "Montserrat-ExtraBold.ttf"
if not font_file.is_file():
    try:
        urllib.request.urlretrieve(
            "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-ExtraBold.ttf", font_file
        )
    except Exception as exc:
        print(f"Não consegui baixar a fonte Montserrat ({{exc}}); a legenda usará a fonte padrão.")
        FONT_NAME = "DejaVu Sans"

try:
    os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
    print("✔ Chave da API encontrada.")
except Exception:
    print("⚠️ Chave ANTHROPIC_API_KEY não encontrada nos Secrets (ícone 🔑 à esquerda). Veja as instruções no topo.")

import ctranslate2
if ctranslate2.get_cuda_device_count() > 0:
    print("✔ GPU ativa — a transcrição será rápida.")
else:
    print("⚠️ Sem GPU: vai funcionar, mas a transcrição demora mais. Ative em Ambiente de execução → Alterar o tipo → GPU T4.")

import cutpreacher
print(f"✔ CutPreacher {{cutpreacher.__version__}} pronto. Vídeos serão salvos em: {{BASE / 'saida'}}")
'''

RUN = '''#@title 2️⃣ Gerar os cortes (preencha e clique no ▶)
link_do_youtube = ""  #@param {type:"string"}
#@markdown Ou, se o YouTube bloquear: nome do arquivo enviado para a pasta CutPreacher/entrada do Drive
arquivo_no_drive = ""  #@param {type:"string"}
quantidade_de_cortes = 3  #@param {type:"slider", min:3, max:8, step:1}
duracao_maxima_segundos = 59  #@param {type:"slider", min:30, max:60, step:1}
enquadramento = "rosto"  #@param ["rosto", "centro", "desfocado"]
#@markdown **rosto**: segue o pregador · **centro**: corte no meio da imagem · **desfocado**: imagem inteira com fundo desfocado

from cutpreacher.config import Settings
from cutpreacher.pipeline import run

if arquivo_no_drive.strip():
    fonte = str(BASE / "entrada" / arquivo_no_drive.strip())
elif link_do_youtube.strip():
    fonte = link_do_youtube.strip()
else:
    raise ValueError("Preencha o link do YouTube (ou o nome do arquivo no Drive).")

config = Settings(
    num_clips=int(quantidade_de_cortes),
    max_clip_seconds=float(duracao_maxima_segundos),
    framing=enquadramento,
    font_name=FONT_NAME,
    fonts_dir=str(BASE / "fontes"),
    cookies_file=str(BASE / "cookies.txt"),
)
resultado = run(fonte, BASE / "saida", config, work_root=Path("/content/cutpreacher_work"))

print()
for arquivo, corte in zip(resultado.files, resultado.clips):
    print(f"🎬 {arquivo.name}\\n   {corte.title}\\n   {corte.description}\\n   {' '.join(corte.hashtags)}\\n")
print(f"📁 Abra o Google Drive → CutPreacher/saida/{resultado.output_dir.name}")
'''

DOWNLOAD = '''#@title 3️⃣ (Opcional) Baixar os vídeos direto neste aparelho
from google.colab import files
for arquivo in resultado.files:
    files.download(str(arquivo))
'''


def package_zip() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted((ROOT / "cutpreacher").glob("*.py")):
            info = zipfile.ZipInfo(f"cutpreacher/{path.name}", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, path.read_bytes())
    return base64.b64encode(buf.getvalue()).decode()


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"cellView": "form"},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build() -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": "CutPreacher.ipynb", "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": INTRO.splitlines(keepends=True)},
            code_cell(SETUP.format(package_zip=package_zip())),
            code_cell(RUN),
            code_cell(DOWNLOAD),
        ],
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Notebook gerado: {OUT}")
