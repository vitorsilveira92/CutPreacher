from __future__ import annotations

import json
import re
from typing import Callable

from .config import Settings
from .models import Clip, Transcript, Word

SYSTEM_PROMPT = """Você é um editor de vídeos especialista em redes sociais de igrejas cristãs.
Sua tarefa é ler a transcrição de uma pregação e escolher os trechos com maior potencial para viralizar
como Reels do Instagram e Shorts do YouTube.

Um bom corte:
- faz sentido sozinho, para quem nunca viu a pregação (não depende de contexto anterior);
- começa forte: a primeira frase já prende a atenção (pergunta, afirmação impactante, história, contraste);
- termina com o pensamento concluído (nunca no meio de uma frase ou ideia);
- traz uma mensagem clara: um ensino bíblico, uma aplicação prática, uma frase de efeito, um momento emocionante ou de esperança;
- evita avisos, ofertas, cumprimentos, orações longas sem conteúdo, piadas internas e trechos que dependem de imagem.

Os tempos devem coincidir com os tempos da transcrição (início de uma fala e fim de uma fala).
Ordene os cortes do melhor para o pior. Os cortes não podem se sobrepor."""

CLIPS_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_seconds": {"type": "number", "description": "Início do corte, em segundos"},
                    "end_seconds": {"type": "number", "description": "Fim do corte, em segundos"},
                    "title": {"type": "string", "description": "Título curto e chamativo (até 60 caracteres)"},
                    "hook": {"type": "string", "description": "A frase de abertura que prende a atenção"},
                    "reason": {"type": "string", "description": "Por que esse trecho é forte"},
                    "description": {
                        "type": "string",
                        "description": "Legenda para o post (2 a 3 frases, pode ter emoji)",
                    },
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["start_seconds", "end_seconds", "title", "hook", "reason", "description", "hashtags"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}

SENTENCE_END = re.compile(r"[.!?…]['\"”)]*$")


class HighlightError(RuntimeError):
    pass


def format_transcript(transcript: Transcript) -> str:
    return "\n".join(f"[{s.start:.1f} → {s.end:.1f}] {s.text}" for s in transcript.segments if s.text)


def build_user_prompt(transcript: Transcript, settings: Settings, count: int, avoid: list[tuple[float, float]]) -> str:
    parts = [
        f"Duração total da pregação: {transcript.duration / 60:.1f} minutos.",
        f"Escolha {count} cortes. Cada corte deve ter entre {settings.min_clip_seconds:.0f} e "
        f"{settings.max_clip_seconds:.0f} segundos (ideal: 30 a 55 segundos).",
    ]
    if avoid:
        ranges = ", ".join(f"{a:.1f}s–{b:.1f}s" for a, b in avoid)
        parts.append(f"Estes intervalos já foram usados ou são inválidos; NÃO os repita nem sobreponha: {ranges}.")
    parts.append("Transcrição (tempos em segundos):\n<transcricao>\n" + format_transcript(transcript) + "\n</transcricao>")
    return "\n\n".join(parts)


def call_claude(client, settings: Settings, user_prompt: str) -> list[dict]:
    with client.beta.messages.stream(
        model=settings.claude_model,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={
            "effort": settings.claude_effort,
            "format": {"type": "json_schema", "schema": CLIPS_SCHEMA},
        },
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        raise HighlightError("A IA recusou a solicitação para esta transcrição.")
    if message.stop_reason == "max_tokens":
        raise HighlightError("A resposta da IA foi cortada (limite de tokens).")
    text = next((b.text for b in message.content if b.type == "text"), None)
    if not text:
        raise HighlightError("A IA não retornou nenhum corte.")
    return json.loads(text)["clips"]


def _nearest_index(values: list[float], target: float) -> int:
    return min(range(len(values)), key=lambda i: abs(values[i] - target))


def snap_to_words(
    words: list[Word], start: float, end: float, min_seconds: float, max_seconds: float
) -> tuple[float, float] | None:
    """Aligns a proposed range to word boundaries and enforces the duration limits.

    Over-long ranges are trimmed back to the last sentence end that still fits.
    """
    if not words or end <= start:
        return None
    si = _nearest_index([w.start for w in words], start)
    ei = _nearest_index([w.end for w in words], end)
    if ei < si:
        return None

    clip_start = words[si].start
    limit = clip_start + max_seconds - 0.4
    if words[ei].end > limit:
        fitting = [j for j in range(si, ei + 1) if words[j].end <= limit]
        if not fitting:
            return None
        sentence_ends = [
            j for j in fitting if SENTENCE_END.search(words[j].text) and words[j].end - clip_start >= min_seconds
        ]
        ei = sentence_ends[-1] if sentence_ends else fitting[-1]

    clip_end = words[ei].end
    if clip_end - clip_start < min_seconds:
        return None
    padded_start = max(0.0, clip_start - 0.15)
    padded_end = min(clip_end + 0.35, padded_start + max_seconds)
    return round(padded_start, 3), round(padded_end, 3)


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def select_clips(
    transcript: Transcript,
    settings: Settings,
    client=None,
    log: Callable[[str], None] = print,
    max_attempts: int = 3,
) -> list[Clip]:
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    words = transcript.words
    if not words:
        raise HighlightError("A transcrição está vazia; não há fala para cortar.")

    accepted: list[Clip] = []
    rejected: list[tuple[float, float]] = []
    for attempt in range(1, max_attempts + 1):
        missing = settings.num_clips - len(accepted)
        if missing <= 0:
            break
        avoid = [(c.start, c.end) for c in accepted] + rejected
        log(f"IA analisando a pregação (tentativa {attempt})...")
        proposals = call_claude(client, settings, build_user_prompt(transcript, settings, missing + 2, avoid))

        for p in proposals:
            if len(accepted) >= settings.num_clips:
                break
            raw = (float(p["start_seconds"]), float(p["end_seconds"]))
            snapped = snap_to_words(words, raw[0], raw[1], settings.min_clip_seconds, settings.max_clip_seconds)
            if snapped is None or any(_overlaps(snapped, (c.start, c.end)) for c in accepted):
                rejected.append(raw)
                continue
            accepted.append(
                Clip(
                    start=snapped[0],
                    end=snapped[1],
                    title=p["title"].strip(),
                    hook=p["hook"].strip(),
                    reason=p["reason"].strip(),
                    description=p["description"].strip(),
                    hashtags=[h if h.startswith("#") else f"#{h}" for h in (t.strip() for t in p["hashtags"]) if h],
                )
            )

    if len(accepted) < settings.num_clips:
        log(f"Aviso: só foi possível encontrar {len(accepted)} cortes válidos (pedido: {settings.num_clips}).")
    if not accepted:
        raise HighlightError("Nenhum corte válido foi encontrado. A pregação é longa o suficiente?")
    return accepted
