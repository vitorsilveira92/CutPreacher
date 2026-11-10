# CutPreacher

Gera automaticamente **cortes verticais (9:16) de até 1 minuto, já legendados**, a partir de uma pregação do YouTube,
prontos para Reels e Shorts. Uma IA (Claude) lê a transcrição completa e escolhe os momentos mais fortes.

## Como funciona

1. **Download** do vídeo com `yt-dlp` (ou uso de um arquivo já enviado ao Google Drive).
2. **Transcrição** com `faster-whisper`, com o tempo de cada palavra.
3. **Agente de IA** (Claude Opus 5.5, resposta em JSON estruturado) escolhe os cortes, com título, legenda para o post e hashtags.
   Os tempos são ajustados ao início e fim das palavras, e cortes longos terminam no fim de uma frase.
4. **Enquadramento 9:16**: segue o rosto do pregador (OpenCV), corta no centro ou mostra o quadro inteiro com fundo desfocado.
5. **Legenda queimada** no vídeo, 3 palavras por vez, com a palavra falada em destaque amarelo.
6. **Saída**: um `.mp4` por corte (1080x1920, H.264/AAC, áudio normalizado) e o arquivo `textos_para_postar.txt`.

## Uso no Google Colab (gratuito)

Abra `notebook/CutPreacher.ipynb` no Google Colab (ele já está na pasta `CutPreacher` do Google Drive) e siga as instruções
no topo do notebook. Os vídeos são salvos em `Meu Drive/CutPreacher/saida`.

Configuração única: crie uma chave da API em https://platform.claude.com/settings/keys e salve-a nos **Secrets** do Colab
com o nome `ANTHROPIC_API_KEY`.

## Uso local (opcional)

Requer Python 3.10+ e `ffmpeg`.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python -m cutpreacher "https://www.youtube.com/watch?v=..." -o saida -n 3
```

## Desenvolvimento

O notebook carrega o pacote `cutpreacher/` embutido. Depois de alterar o código, gere o notebook de novo:

```bash
python scripts/build_notebook.py
pip install pytest scikit-image   # e apt install espeak-ng para os testes de integração
python -m pytest
```
