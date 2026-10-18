"""Hold a key, talk, release. Clean text lands wherever your cursor is."""
import queue
import sys
import threading
import time

import numpy as np

from .cleanup import clean_transcript
from .learning import load_corrections, load_dictionary, log_history, whisper_prompt
from .paths import CONFIG_PATH, DATA_DIR

SAMPLE_RATE = 16000  # Whisper's native rate


def resolve_key(name: str):
    from pynput.keyboard import Key, KeyCode
    name = name.strip()
    if hasattr(Key, name):
        return getattr(Key, name)
    if len(name) == 1:
        return KeyCode.from_char(name)
    raise SystemExit(f"Unknown hotkey '{name}'. Use a pynput key name like alt_r, ctrl_r, cmd_r, f8.")


def pick_device(cfg: dict) -> tuple[str, str]:
    device, compute = cfg.get("device", "auto"), cfg.get("compute_type", "auto")
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


class Recorder:
    def __init__(self, input_device):
        import sounddevice as sd
        self.sd = sd
        self.input_device = input_device
        self.chunks: list[np.ndarray] = []
        self.stream = None
        self.started_at = 0.0

    def start(self) -> None:
        self.chunks = []
        self.stream = self.sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            device=self.input_device, callback=self._on_audio,
        )
        self.stream.start()
        self.started_at = time.time()

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        self.chunks.append(indata[:, 0].copy())

    def stop(self) -> tuple[np.ndarray, float]:
        seconds = time.time() - self.started_at
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        audio = np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype="float32")
        return audio, seconds


class Transcriber:
    def __init__(self, cfg: dict):
        from faster_whisper import WhisperModel
        device, compute = pick_device(cfg)
        print(f"Loading Whisper '{cfg['whisper_model']}' on {device} ({compute}). "
              f"First run downloads the model...")
        t = time.time()
        self.model = WhisperModel(cfg["whisper_model"], device=device, compute_type=compute)
        self.language = cfg.get("language") or None
        print(f"Model ready in {time.time() - t:.1f}s.")

    def transcribe(self, audio: np.ndarray, prompt: str) -> str:
        segments, _ = self.model.transcribe(
            audio, language=self.language, beam_size=5,
            initial_prompt=prompt or None, vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


def paste_text(text: str, method: str) -> None:
    import pyperclip
    from pynput.keyboard import Controller, Key
    kb = Controller()
    if method == "type":
        kb.type(text)
        return
    try:
        previous = pyperclip.paste()
    except pyperclip.PyperclipException:
        previous = None
    pyperclip.copy(text)
    time.sleep(0.05)
    mod = Key.cmd if sys.platform == "darwin" else Key.ctrl
    with kb.pressed(mod):
        kb.press("v")
        kb.release("v")
    if previous is not None:
        # Give the target app a moment to read the clipboard, then put the old contents back.
        def restore():
            time.sleep(0.6)
            try:
                pyperclip.copy(previous)
            except pyperclip.PyperclipException:
                pass
        threading.Thread(target=restore, daemon=True).start()


class App:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.hotkey = resolve_key(cfg["hotkey"])
        self.toggle = cfg.get("mode") == "toggle"
        self.recorder = Recorder(cfg.get("input_device"))
        self.transcriber = Transcriber(cfg)
        self.jobs: "queue.Queue[tuple[np.ndarray, float]]" = queue.Queue()
        self.recording = False
        self.lock = threading.Lock()

    # --- hotkey handling -------------------------------------------------
    def _is_hotkey(self, key) -> bool:
        return key == self.hotkey or getattr(key, "char", None) == getattr(self.hotkey, "char", object())

    def on_press(self, key) -> None:
        if not self._is_hotkey(key):
            return
        with self.lock:
            if self.toggle:
                if self.recording:
                    self._stop()
                else:
                    self._start()
            elif not self.recording:  # key-repeat sends many press events; ignore them
                self._start()

    def on_release(self, key) -> None:
        if self.toggle or not self._is_hotkey(key):
            return
        with self.lock:
            if self.recording:
                self._stop()

    def _start(self) -> None:
        try:
            self.recorder.start()
        except Exception as e:
            print(f"[audio] Could not open microphone: {e}", file=sys.stderr)
            return
        self.recording = True
        print("\r● listening...          ", end="", flush=True)

    def _stop(self) -> None:
        self.recording = False
        audio, seconds = self.recorder.stop()
        print("\r  transcribing...       ", end="", flush=True)
        self.jobs.put((audio, seconds))

    # --- background worker ---------------------------------------------
    def worker(self) -> None:
        while True:
            audio, seconds = self.jobs.get()
            if seconds < self.cfg.get("min_seconds", 0.4) or audio.size == 0:
                print("\r(too short, ignored)      ")
                continue
            # Reload every time so edits to dictionary/corrections apply immediately.
            dictionary = load_dictionary()
            corrections = load_corrections()
            t = time.time()
            raw = self.transcriber.transcribe(audio, whisper_prompt(dictionary))
            if not raw:
                print("\r(heard nothing)           ")
                continue
            final = clean_transcript(raw, self.cfg, dictionary, corrections)
            if not final:
                print("\r(nothing left after cleanup)")
                continue
            paste_text(final, self.cfg.get("paste_method", "paste"))
            log_history(raw, final, seconds)
            print(f"\r✓ {final}    [{seconds:.1f}s audio, {time.time() - t:.1f}s processing]")

    def run(self) -> None:
        from pynput import keyboard
        threading.Thread(target=self.worker, daemon=True).start()
        how = "tap to start / tap to stop" if self.toggle else "hold to talk"
        print(f"Ready. Hotkey: {self.cfg['hotkey']} ({how}). Ctrl+C to quit.")
        print(f"Settings: {CONFIG_PATH}   Vocabulary: {DATA_DIR}")
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\nBye.")
