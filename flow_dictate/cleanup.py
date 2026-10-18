"""Turn a raw transcript into text you'd actually paste.

Stage 1 (always, free, instant): corrections + spoken commands + filler removal.
Stage 2 (optional): a language model smooths self-corrections and grammar.
"""
import json
import re
import sys
import urllib.error
import urllib.request

from .learning import apply_corrections

# Things people say out loud that should become formatting.
SPOKEN_COMMANDS = [
    (r"\b(?:new|next) paragraph\b[.,]?", "\n\n"),
    (r"\bnew line\b[.,]?", "\n"),
]

SELF_CORRECTION_MARKERS = r"(?:scratch that|strike that|no wait|actually no)"

# Punctuation spoken at the very end of a dictation.
SPOKEN_END_PUNCTUATION = [
    (r"[,.]?\s*\b(?:period|full stop)\b\.?\s*$", "."),
    (r"[,.]?\s*\bquestion mark\b\.?\s*$", "?"),
    (r"[,.]?\s*\bexclamation (?:point|mark)\b\.?\s*$", "!"),
    (r"[,.]?\s*\bcomma\b\.?\s*$", ","),
]

TERMINAL = ".!?"
KEEP_CAPITAL = {"i", "i'm", "i'll", "i've", "i'd"}


def assemble_words(words: list[tuple[str, float, float]], pause_seconds: float) -> str:
    """Join Whisper's timed words into text, ending a sentence wherever the speaker
    paused for `pause_seconds` or longer without any punctuation.

    words: (text, start, end) per word, in order. Whisper attaches punctuation to
    the word it follows ("thinking," or "water?"), so a trailing mark on the previous
    word means it already decided; we only step in when it wrote nothing.
    """
    out: list[str] = []
    prev_end = None
    for text, start, end in words:
        text = text.strip()
        if not text:
            continue
        if out and prev_end is not None and pause_seconds > 0 and start - prev_end >= pause_seconds:
            if out[-1][-1].isalnum():
                out[-1] += "."
            if out[-1][-1] in TERMINAL:
                text = text[:1].upper() + text[1:]
        out.append(text)
        prev_end = end
    return " ".join(out)


def rule_based_cleanup(text: str, corrections: dict[str, str], fillers: list[str]) -> str:
    text = text.strip()
    if not text:
        return ""

    text = apply_corrections(text, corrections)

    # "scratch that" drops the clause before it. "I said X, scratch that, Y" -> "Y"
    text = re.sub(
        r"(^|[.!?]\s+)[^.!?]*?\b" + SELF_CORRECTION_MARKERS + r"\b[,.]?\s*",
        lambda m: m.group(1),
        text,
        flags=re.IGNORECASE,
    )

    for pattern, replacement in SPOKEN_COMMANDS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    for pattern, replacement in SPOKEN_END_PUNCTUATION:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    for filler in fillers:
        if filler.endswith(","):
            # "like," and "you know," only count when they're parenthetical.
            pattern = r"(?<!\w)" + re.escape(filler[:-1]) + r"\s*,\s*"
        else:
            pattern = r"(?<!\w)" + re.escape(filler) + r"(?!\w)[,.]?\s*"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Tidy what the removals left behind.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:]){2,}", r"\1", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.rstrip().lstrip(" ,")

    # Capitalize sentence starts after punctuation. The first letter is handled by
    # join_with_previous(), which knows whether we're continuing a sentence.
    text = re.sub(r"([.!?]\s+|\n+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text.rstrip()


def _first_word(text: str) -> str:
    m = re.match(r"[A-Za-z][A-Za-z']*", text.lstrip())
    return m.group(0) if m else ""


def capitalize_first(text: str) -> str:
    i = len(text) - len(text.lstrip())
    return text[:i] + text[i:i + 1].upper() + text[i + 1:]


def lowercase_first(text: str, dictionary: list[str]) -> str:
    """Lowercase the first word unless it should stay capitalized: 'I', an acronym,
    or a term from the personal dictionary that is written with a capital."""
    word = _first_word(text)
    if not word:
        return text
    if word.lower() in KEEP_CAPITAL or (word.isupper() and len(word) > 1):
        return text
    for term in dictionary:
        if term[:1].isupper() and term.split()[0].lower() == word.lower():
            return text
    i = len(text) - len(text.lstrip())
    return text[:i] + text[i:i + 1].lower() + text[i + 1:]


def join_with_previous(previous: str | None, text: str, dictionary: list[str]) -> str:
    """Decide spacing and the first letter from how the previous dictation ended.

    previous=None means a fresh field: no leading space, capital letter.
    Previous ended with . ! ? or a newline: new sentence, capital letter.
    Previous ended mid-sentence (no punctuation, or a comma): continue it,
    lowercase first letter. A space is added whenever the previous text
    doesn't already end with one.
    """
    if not text:
        return text
    if previous is None or not previous.strip():
        return capitalize_first(text)
    tail = previous.rstrip()
    needs_space = not previous[-1].isspace() and not text[0].isspace()
    if previous.endswith("\n") or tail[-1] in TERMINAL or text.lstrip(" ")[:1] == "\n":
        out = capitalize_first(text)
    else:
        out = lowercase_first(text, dictionary)
    return (" " if needs_space else "") + out


def _llm_system_prompt(dictionary: list[str], style_notes: str) -> str:
    vocab = ", ".join(dictionary[-100:]) if dictionary else "(none yet)"
    return (
        "You clean up voice dictation. You receive a raw speech-to-text transcript and "
        "return the text the speaker meant to write, and nothing else: no preamble, no "
        "quotes, no commentary.\n\n"
        "Rules:\n"
        "- Keep the speaker's words, tone and meaning. Do not summarize, expand, or reword.\n"
        "- Remove filler (um, uh, you know), false starts, and repeated words.\n"
        "- Resolve self-corrections: 'send it Tuesday, no, Wednesday' becomes 'send it Wednesday'.\n"
        "- Fix punctuation, capitalization, and obvious misrecognitions.\n"
        "- Spell these personal vocabulary terms exactly as written when they appear: "
        f"{vocab}\n"
        "- 'new line' / 'new paragraph' are formatting commands, not words.\n"
        "- If the transcript is already clean, return it unchanged.\n"
        + (f"\nSpeaker's style notes: {style_notes}\n" if style_notes else "")
    )


def claude_cleanup(text: str, dictionary: list[str], style_notes: str, model: str) -> str | None:
    """Second pass with Claude. Returns None on any failure so the caller falls back."""
    try:
        import anthropic
    except ImportError:
        print("[cleanup] anthropic package not installed; run: pip install anthropic", file=sys.stderr)
        return None

    client = anthropic.Anthropic(timeout=20.0, max_retries=1)
    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=4096,  # output is the cleaned transcript, never longer than the input
            system=_llm_system_prompt(dictionary, style_notes),
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": text}],
        )
    except anthropic.AuthenticationError:
        print("[cleanup] Claude: no valid API key. Set ANTHROPIC_API_KEY or run `ant auth login`.", file=sys.stderr)
        return None
    except anthropic.RateLimitError:
        print("[cleanup] Claude: rate limited, using rule-based text.", file=sys.stderr)
        return None
    except anthropic.APIStatusError as e:
        print(f"[cleanup] Claude API error {e.status_code}: {e.message}", file=sys.stderr)
        return None
    except anthropic.APIConnectionError as e:
        print(f"[cleanup] Claude: connection problem ({e}), using rule-based text.", file=sys.stderr)
        return None

    if response.stop_reason == "refusal":
        return None
    out = "".join(b.text for b in response.content if b.type == "text").strip()
    return out or None


def ollama_cleanup(text: str, dictionary: list[str], style_notes: str, model: str, url: str) -> str | None:
    """Second pass with a local Ollama model. Free; quality depends on the model."""
    body = json.dumps({
        "model": model,
        "system": _llm_system_prompt(dictionary, style_notes),
        "prompt": text,
        "stream": False,
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[cleanup] Ollama not reachable ({e}), using rule-based text.", file=sys.stderr)
        return None
    out = (data.get("response") or "").strip().strip('"')
    return out or None


def sanity_check(raw: str, cleaned: str) -> bool:
    """Guard against a model that answered the dictation instead of cleaning it."""
    if not cleaned:
        return False
    raw_words = len(raw.split())
    out_words = len(cleaned.split())
    # Cleanup should shorten or keep length. A big expansion means it went off-script.
    return out_words <= max(raw_words * 1.3, raw_words + 5)


def clean_transcript(raw: str, cfg: dict, dictionary: list[str], corrections: dict[str, str],
                     previous: str | None = None) -> str:
    """Full pipeline. `previous` is the text of the last dictation pasted into the same
    place, or None if this is a fresh field."""
    text = _clean_body(raw, cfg, dictionary, corrections)
    if not text:
        return ""
    if cfg.get("add_final_period") and text[-1] not in TERMINAL + ",;:\n":
        text += "."
    return join_with_previous(previous, text, dictionary)


def _clean_body(raw: str, cfg: dict, dictionary: list[str], corrections: dict[str, str]) -> str:
    text = rule_based_cleanup(raw, corrections, cfg.get("fillers", []))
    if not text:
        return ""
    mode = cfg.get("llm_cleanup", "off")
    improved = None
    if mode == "claude":
        improved = claude_cleanup(text, dictionary, cfg.get("style_notes", ""), cfg["claude_model"])
    elif mode == "ollama":
        improved = ollama_cleanup(text, dictionary, cfg.get("style_notes", ""),
                                  cfg["ollama_model"], cfg["ollama_url"])
    if improved and sanity_check(text, improved):
        # Corrections are the user's explicit word; they win over the model.
        return apply_corrections(improved, corrections)
    return text
