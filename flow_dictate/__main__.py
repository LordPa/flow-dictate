"""Command line entry point.

  python -m flow_dictate                 start dictating
  python -m flow_dictate learn "Term"    teach a word or name
  python -m flow_dictate correct "heard" "meant"
  python -m flow_dictate history [N]     show recent dictations (raw vs final)
  python -m flow_dictate devices         list microphones
  python -m flow_dictate hotkey ctrl_r   change the push-to-talk key
  python -m flow_dictate file audio.wav  transcribe a file (test the pipeline without a mic)
  python -m flow_dictate clean "text"    run only the cleanup step on some text
"""
import sys

from .paths import CONFIG_PATH, DICTIONARY_PATH, CORRECTIONS_PATH, load_config, save_config


def main(argv: list[str]) -> int:
    cfg = load_config()
    cmd = argv[0] if argv else "run"

    if cmd == "run":
        from .app import App
        App(cfg).run()
        return 0

    if cmd == "learn":
        from .learning import add_to_dictionary
        if len(argv) < 2:
            print('usage: python -m flow_dictate learn "Term"')
            return 2
        added = [t for t in argv[1:] if add_to_dictionary(t)]
        print(f"Added {len(added)} term(s) to {DICTIONARY_PATH}" if added else "Already known.")
        return 0

    if cmd == "correct":
        from .learning import add_correction
        if len(argv) != 3:
            print('usage: python -m flow_dictate correct "what it heard" "what you meant"')
            return 2
        add_correction(argv[1], argv[2])
        print(f'Will now replace "{argv[1]}" with "{argv[2]}". Saved to {CORRECTIONS_PATH}')
        return 0

    if cmd == "history":
        from .learning import read_history
        n = int(argv[1]) if len(argv) > 1 else 10
        for rec in read_history(n):
            print(f"raw:   {rec['raw']}\nfinal: {rec['final']}\n")
        return 0

    if cmd == "hotkey":
        from .app import resolve_key
        if len(argv) != 2:
            print(f"usage: python -m flow_dictate hotkey <key>    current: {cfg['hotkey']}")
            print("examples: ctrl_r  alt_r  cmd_r  f8  f9  pause  scroll_lock  auto")
            return 2
        try:
            resolve_key(argv[1])  # exits with a message if the name is unknown
        except ImportError:
            pass  # no display available to load the keyboard library; save anyway
        cfg["hotkey"] = argv[1]
        save_config(cfg)
        print(f"Hotkey set to {argv[1]}. Restart the app for it to take effect.")
        return 0

    if cmd == "devices":
        import sounddevice as sd
        print(sd.query_devices())
        print("\nSet \"input_device\" in config.json to a device number or name, or leave null for default.")
        return 0

    if cmd == "file":
        if len(argv) != 2:
            print("usage: python -m flow_dictate file path/to/audio.wav")
            return 2
        from .app import Transcriber
        from .cleanup import clean_transcript
        from .learning import load_corrections, load_dictionary, whisper_prompt
        from faster_whisper.audio import decode_audio
        audio = decode_audio(argv[1], sampling_rate=16000)
        dictionary, corrections = load_dictionary(), load_corrections()
        raw = Transcriber(cfg).transcribe(audio, whisper_prompt(dictionary))
        print(f"raw:   {raw}")
        print(f"final: {clean_transcript(raw, cfg, dictionary, corrections)}")
        return 0

    if cmd == "clean":
        from .cleanup import clean_transcript
        from .learning import load_corrections, load_dictionary
        text = " ".join(argv[1:]) or sys.stdin.read()
        print(clean_transcript(text, cfg, load_dictionary(), load_corrections()))
        return 0

    print(__doc__)
    print(f"Config file: {CONFIG_PATH}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
