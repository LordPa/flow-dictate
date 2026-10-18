"""Personal vocabulary and corrections. This is the 'learns as you go' part.

Two files, both plain text so you can edit them by hand:

  dictionary.txt    words/phrases the recognizer should know (names, jargon)
  corrections.json  {"what it hears": "what you meant"} applied to every transcript
"""
import json
import re
import time
from pathlib import Path

from .paths import CORRECTIONS_PATH, DICTIONARY_PATH, HISTORY_PATH, ensure_data_dir


def load_dictionary(path: Path = DICTIONARY_PATH) -> list[str]:
    if not path.exists():
        return []
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.append(line)
    # de-dupe, keep order
    seen, out = set(), []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return out


def add_to_dictionary(term: str, path: Path = DICTIONARY_PATH) -> bool:
    ensure_data_dir()
    term = term.strip()
    if not term:
        return False
    if term.lower() in {w.lower() for w in load_dictionary(path)}:
        return False
    with path.open("a", encoding="utf-8") as f:
        f.write(term + "\n")
    return True


def load_corrections(path: Path = CORRECTIONS_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in data.items()}


def add_correction(heard: str, meant: str, path: Path = CORRECTIONS_PATH) -> None:
    ensure_data_dir()
    data = load_corrections(path)
    data[heard.strip()] = meant.strip()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # If the right spelling is a name or term, the recognizer should know it too.
    add_to_dictionary(meant)


def _match_case(src: str, dst: str) -> str:
    """Make `dst` follow the casing of `src` when src is all-caps or capitalized."""
    if src.isupper() and len(src) > 1:
        return dst.upper()
    if src[:1].isupper() and not any(c.isupper() for c in dst[1:]):
        return dst[:1].upper() + dst[1:]
    return dst


def apply_corrections(text: str, corrections: dict[str, str]) -> str:
    # Longest phrases first so "whisper flow" wins over "whisper".
    for heard in sorted(corrections, key=len, reverse=True):
        meant = corrections[heard]
        pattern = r"(?<!\w)" + re.escape(heard) + r"(?!\w)"

        def repl(m: re.Match) -> str:
            # If the replacement has explicit capitals (a proper noun), keep them as given.
            if any(c.isupper() for c in meant):
                return meant
            return _match_case(m.group(0), meant)

        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def whisper_prompt(dictionary: list[str], max_terms: int = 60) -> str:
    """Whisper reads the 'initial_prompt' as prior context. Listing your vocabulary
    there nudges it to spell those words correctly. It only sees ~224 tokens, so
    we pass the most recently added terms."""
    terms = dictionary[-max_terms:]
    if not terms:
        return ""
    return "Vocabulary: " + ", ".join(terms) + "."


def log_history(raw: str, final: str, seconds: float, path: Path = HISTORY_PATH) -> None:
    ensure_data_dir()
    rec = {"t": int(time.time()), "seconds": round(seconds, 2), "raw": raw, "final": final}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_history(n: int = 20, path: Path = HISTORY_PATH) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-n:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out
