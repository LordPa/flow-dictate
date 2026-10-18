import json
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("FLOW_DICTATE_HOME", Path.home() / ".flow-dictate"))
CONFIG_PATH = DATA_DIR / "config.json"
DICTIONARY_PATH = DATA_DIR / "dictionary.txt"
CORRECTIONS_PATH = DATA_DIR / "corrections.json"
HISTORY_PATH = DATA_DIR / "history.jsonl"

DEFAULT_CONFIG = {
    # Hold this key to talk, release to transcribe and paste.
    # Any pynput key name works: alt_r, ctrl_r, cmd_r, f8, pause, scroll_lock ...
    "hotkey": "alt_r",
    # "hold" = push-to-talk. "toggle" = tap once to start, tap again to stop.
    "mode": "hold",
    # Whisper model. Bigger = more accurate, slower. Good choices:
    #   tiny.en, base.en, small.en, medium.en, large-v3-turbo, distil-large-v3
    "whisper_model": "small.en",
    "language": "en",
    # "auto" picks CUDA if available, otherwise CPU. Apple Silicon runs on CPU.
    "device": "auto",
    "compute_type": "auto",
    # Microphone: null = system default. Run `python -m flow_dictate devices` to list.
    "input_device": None,
    # "paste" simulates Cmd/Ctrl+V (fast). "type" types each character (works
    # in a few apps that block paste, but is slow).
    "paste_method": "paste",
    # Drop recordings shorter than this (accidental taps).
    "min_seconds": 0.4,
    # Filler words removed by the rule-based cleaner.
    "fillers": ["um", "uh", "uhm", "erm", "hmm", "mm", "you know,"],
    # Second-pass cleanup with a language model. "off" is free and local.
    # "claude" needs ANTHROPIC_API_KEY (or `ant auth login`).
    # "ollama" uses a local Ollama server (free, needs ~4 GB RAM for the default model).
    "llm_cleanup": "off",
    "claude_model": "claude-opus-5-5",
    "ollama_model": "llama3.2",
    "ollama_url": "http://localhost:11434",
    # Your writing style hints for the LLM pass, e.g. "casual, lowercase ok".
    "style_notes": "",
}


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n")
    if not DICTIONARY_PATH.exists():
        DICTIONARY_PATH.write_text("# One word or phrase per line. Names, products, jargon.\n")
    if not CORRECTIONS_PATH.exists():
        CORRECTIONS_PATH.write_text("{}\n")


def load_config() -> dict:
    ensure_data_dir()
    cfg = dict(DEFAULT_CONFIG)
    try:
        cfg.update(json.loads(CONFIG_PATH.read_text()))
    except json.JSONDecodeError as e:
        raise SystemExit(f"config.json is not valid JSON ({e}). File: {CONFIG_PATH}")
    return cfg
