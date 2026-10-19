"""Hold a key, talk, release. Clean text lands wherever your cursor is."""
import glob
import os
import queue
import sys
import sysconfig
import threading
import time

import numpy as np

from .cleanup import assemble_words, clean_transcript, find_cut, has_speech, ollama_warmup
from .learning import load_corrections, load_dictionary, log_history, whisper_prompt
from .paths import CONFIG_PATH, DATA_DIR, default_hotkey

SAMPLE_RATE = 16000  # Whisper's native rate


def resolve_key(name: str):
    from pynput.keyboard import Key, KeyCode
    name = (name or "auto").strip()
    if name == "auto":
        name = default_hotkey()
    if hasattr(Key, name):
        return getattr(Key, name)
    if len(name) == 1:
        return KeyCode.from_char(name)
    raise SystemExit(f"Unknown hotkey '{name}'. Use a key name like ctrl_r, alt_r, cmd_r, f8, pause.")


def add_cuda_dll_dirs() -> None:
    """Windows: make pip-installed NVIDIA libraries (nvidia-cublas-cu12, nvidia-cudnn-cu12)
    visible to the speech engine. They live in site-packages/nvidia/<lib>/bin."""
    if sys.platform != "win32":
        return
    for site in {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}:
        for d in glob.glob(os.path.join(site, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def foreground_window():
    """Identify the window that will receive the paste, so we know whether the next
    dictation continues the previous one. Windows only; elsewhere returns None."""
    if sys.platform == "win32":
        import ctypes
        return ctypes.windll.user32.GetForegroundWindow()
    return None


def gpu_summary() -> str:
    """Name and memory use of the NVIDIA card, for the startup line. Empty if none."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.splitlines()[0] if out else ""


def pick_device(cfg: dict) -> tuple[str, str]:
    device, compute = cfg.get("device", "auto"), cfg.get("compute_type", "auto")
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "int8_float16" if device == "cuda" else "int8"
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

    def snapshot(self) -> np.ndarray:
        chunks = list(self.chunks)  # list copy is atomic enough for a callback appender
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")

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
        self.cfg = cfg
        self.language = cfg.get("language") or None
        self.lock = threading.Lock()  # one transcription at a time
        add_cuda_dll_dirs()
        device, compute = pick_device(cfg)
        try:
            self._load(device, compute)
        except Exception as e:
            if device == "cpu":
                raise
            print(f"[whisper] GPU not usable ({type(e).__name__}: {str(e)[:160]}).")
            print("[whisper] Using the CPU instead. For GPU speed, run once:  "
                  "venv\\Scripts\\python -m pip install -r requirements-gpu.txt")
            self._load("cpu", "int8")

    def _load(self, device: str, compute: str) -> None:
        from faster_whisper import WhisperModel
        print(f"Loading Whisper '{self.cfg['whisper_model']}' on {device} ({compute}). "
              f"First run downloads the model...")
        t = time.time()
        self.model = WhisperModel(self.cfg["whisper_model"], device=device, compute_type=compute)
        # Push one second of noise through the model now, with the silence filter off so
        # the GPU is really exercised. On Windows a missing CUDA/cuDNN DLL only shows up
        # at the first transcription, and we'd rather find out here and fall back.
        noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE) * 0.01).astype("float32")
        self.transcribe(noise, "", vad=False)
        self.device = device
        gpu = gpu_summary()
        print(f"Model ready on {device} in {time.time() - t:.1f}s." + (f"  GPU: {gpu}" if gpu else ""))

    def beam_size(self) -> int:
        b = self.cfg.get("beam_size", "auto")
        if b == "auto":
            return 5 if getattr(self, "device", "cpu") == "cuda" else 1
        return int(b)

    def transcribe_words(self, audio: np.ndarray, prompt: str, vad: bool = True,
                         offset: float = 0.0) -> tuple[list[str], list[tuple]]:
        """Returns (segment texts, [(word, start, end)]) with times shifted by `offset`."""
        with self.lock:
            segments, _ = self.model.transcribe(
                audio, language=self.language, beam_size=self.beam_size(),
                initial_prompt=prompt or None, vad_filter=vad,
                condition_on_previous_text=False, word_timestamps=True,
            )
            texts, words = [], []
            for seg in segments:
                texts.append(seg.text.strip())
                for i, w in enumerate(seg.words or []):
                    words.append((w.word, w.start + offset, w.end + offset, i == 0))
        return texts, words

    def transcribe(self, audio: np.ndarray, prompt: str, vad: bool = True) -> str:
        texts, words = self.transcribe_words(audio, prompt, vad)
        return self.to_text(texts, words)

    def to_text(self, texts: list[str], words: list[tuple]) -> str:
        pause = float(self.cfg.get("pause_sentence_seconds", 0.8) or 0)
        if words:
            return assemble_words(words, pause, load_dictionary())
        return " ".join(texts).strip()


class LiveSession:
    """Transcribes finished stretches of speech while the key is still held.

    Every half second it looks at the audio recorded so far. If there's a pause of
    half a second after at least two seconds of new audio, everything before that
    pause is transcribed now. On release, only the tail after the last commit is
    left to do, so the text appears almost immediately."""

    def __init__(self, recorder: Recorder, transcriber: Transcriber, prompt: str):
        self.recorder = recorder
        self.transcriber = transcriber
        self.prompt = prompt
        self.committed = 0          # samples already transcribed
        self.texts: list[str] = []
        self.words: list[tuple] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def context_prompt(self) -> str:
        tail = " ".join(" ".join(self.texts).split()[-30:])
        return (self.prompt + " " + tail).strip()

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            audio = self.recorder.snapshot()
            cut = find_cut(audio, self.committed, SAMPLE_RATE)
            if cut is None:
                continue
            self._commit(audio[self.committed:cut], cut)

    def _commit(self, segment: np.ndarray, new_committed: int) -> None:
        if has_speech(segment, SAMPLE_RATE):
            texts, words = self.transcriber.transcribe_words(
                segment, self.context_prompt(), offset=self.committed / SAMPLE_RATE)
            self.texts.extend(t for t in texts if t)
            self.words.extend(words)
        self.committed = new_committed

    def finish(self, audio: np.ndarray) -> tuple[list[str], list[tuple]]:
        """Stop the background loop and transcribe whatever is left."""
        self._stop.set()
        self._thread.join()
        tail = audio[self.committed:]
        if tail.size > SAMPLE_RATE // 4 and has_speech(tail, SAMPLE_RATE):
            texts, words = self.transcriber.transcribe_words(
                tail, self.context_prompt(), offset=self.committed / SAMPLE_RATE)
            self.texts.extend(t for t in texts if t)
            self.words.extend(words)
        return self.texts, self.words


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
    # Put the text on the clipboard and make sure it actually landed before pasting.
    # Clipboard managers and other apps can briefly hold the clipboard on Windows.
    for attempt in range(10):
        try:
            pyperclip.copy(text)
            if pyperclip.paste() == text:
                break
        except pyperclip.PyperclipException:
            pass
        time.sleep(0.05)
    else:
        print("[paste] could not place text on the clipboard; typing it instead", file=sys.stderr)
        kb.type(text)
        return
    mod = Key.cmd if sys.platform == "darwin" else Key.ctrl
    # One clean Ctrl+V with explicit timing so no key is ever left held long enough
    # for the system to auto-repeat it.
    kb.press(mod)
    time.sleep(0.02)
    kb.press("v")
    time.sleep(0.02)
    kb.release("v")
    time.sleep(0.02)
    kb.release(mod)
    print(f"[paste] {len(text)} chars", file=sys.stderr)
    if previous is not None and previous != text:
        def restore():
            # Give the target app time to read the clipboard, then put the old contents
            # back, unless something else has changed the clipboard in the meantime.
            time.sleep(1.0)
            try:
                if pyperclip.paste() == text:
                    pyperclip.copy(previous)
            except pyperclip.PyperclipException:
                pass
        threading.Thread(target=restore, daemon=True).start()


class App:
    def __init__(self, cfg: dict, on_state=None):
        self.cfg = cfg
        self.on_state = on_state or (lambda state: None)   # "idle" | "listening" | "busy"
        self.paused = False
        self.hotkey_name = cfg.get("hotkey") or "auto"
        if self.hotkey_name == "auto":
            self.hotkey_name = default_hotkey()
        self.hotkey = resolve_key(self.hotkey_name)
        self.toggle = cfg.get("mode") == "toggle"
        self.recorder = Recorder(cfg.get("input_device"))
        if cfg.get("llm_cleanup") == "ollama":
            ollama_warmup(cfg["ollama_model"], cfg["ollama_url"])  # in the background
        self.transcriber = Transcriber(cfg)
        self.jobs: "queue.Queue[tuple[np.ndarray, float, LiveSession | None]]" = queue.Queue()
        self.live: LiveSession | None = None
        self.recording = False
        self.lock = threading.Lock()
        # Context for spacing/capitalization: what we last pasted, where, and when.
        self.last_pasted: str | None = None
        self.last_window = None
        self.last_time = 0.0

    # --- hotkey handling -------------------------------------------------
    def _is_hotkey(self, key) -> bool:
        return key == self.hotkey or getattr(key, "char", None) == getattr(self.hotkey, "char", object())

    def on_press(self, key) -> None:
        if self.paused or not self._is_hotkey(key):
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
        if self.cfg.get("live_transcription", True):
            self.live = LiveSession(self.recorder, self.transcriber, whisper_prompt(load_dictionary()))
        self.on_state("listening")
        print("\r● listening...          ", end="", flush=True)

    def _stop(self) -> None:
        self.recording = False
        audio, seconds = self.recorder.stop()
        self.on_state("busy")
        print("\r  transcribing...       ", end="", flush=True)
        self.jobs.put((audio, seconds, self.live))
        self.live = None

    # --- background worker ---------------------------------------------
    def worker(self) -> None:
        while True:
            audio, seconds, live = self.jobs.get()
            try:
                self._process(audio, seconds, live)
            except Exception as e:
                print(f"\r[error] {type(e).__name__}: {e}", file=sys.stderr)
                print("Still running. Try again, and send this message if it keeps happening.")
            finally:
                if not self.recording:
                    self.on_state("idle")

    def _process(self, audio: np.ndarray, seconds: float, live: "LiveSession | None") -> None:
        if seconds < self.cfg.get("min_seconds", 0.4) or audio.size == 0:
            if live:
                live.finish(np.zeros(0, dtype="float32"))
            print("\r(too short, ignored)      ")
            return
        # Reload every time so edits to dictionary/corrections apply immediately.
        dictionary = load_dictionary()
        corrections = load_corrections()
        t = time.time()
        if live:
            texts, words = live.finish(audio)
            raw = self.transcriber.to_text(texts, words)
        else:
            raw = self.transcriber.transcribe(audio, whisper_prompt(dictionary))
        t_whisper = time.time() - t
        if not raw:
            print("\r(heard nothing)           ")
            return
        window = foreground_window()
        same_place = (window == self.last_window and self.last_pasted is not None
                      and (window is not None or time.time() - self.last_time < 300))
        final = clean_transcript(raw, self.cfg, dictionary, corrections,
                                 previous=self.last_pasted if same_place else None)
        if not final:
            print("\r(nothing left after cleanup)")
            return
        t_cleanup = time.time() - t - t_whisper
        paste_text(final, self.cfg.get("paste_method", "paste"))
        self.last_pasted, self.last_window, self.last_time = final, window, time.time()
        log_history(raw, final, seconds)
        timing = f"{seconds:.1f}s audio, {t_whisper:.1f}s speech"
        if self.cfg.get("llm_cleanup", "off") != "off":
            timing += f" + {t_cleanup:.1f}s cleanup"
        print(f"\r✓ {final}    [{timing}]")

    def start(self):
        """Start listening in the background. Returns the keyboard listener."""
        from pynput import keyboard
        threading.Thread(target=self.worker, daemon=True).start()
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        return listener

    def run(self) -> None:
        how = "tap to start / tap to stop" if self.toggle else "hold to talk"
        try:
            mic = self.recorder.sd.query_devices(self.cfg.get("input_device"), "input")["name"]
        except Exception as e:  # no default input device, or a bad input_device setting
            mic = f"NONE FOUND ({e})"
        print(f"Ready. Hotkey: {self.hotkey_name} ({how}). Microphone: {mic}. Ctrl+C to quit.")
        print(f"Change the key with:  python -m flow_dictate hotkey f8   (or ctrl_r, alt_r, pause, scroll_lock)")
        print(f"Settings: {CONFIG_PATH}   Vocabulary: {DATA_DIR}")
        listener = self.start()
        try:
            listener.join()
        except KeyboardInterrupt:
            print("\nBye.")
