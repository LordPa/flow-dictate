# Flow Dictate

Hold a key, talk, let go. Clean text appears wherever your cursor is.
A free, local replacement for the one Wispr Flow feature most people use.

Everything runs on your machine with no subscription:

| Step | What does it | Cost |
|------|--------------|------|
| Listen | your microphone, while the hotkey is held | free |
| Transcribe | Whisper (OpenAI's open speech model) via `faster-whisper`, on your CPU/GPU | free |
| Clean up | filler removal, "scratch that", "new paragraph", your personal corrections | free |
| Polish (optional) | a language model smooths grammar and self-corrections | free with Ollama, pennies with Claude |
| Paste | simulated Cmd/Ctrl+V into the active app | free |

## Install

You need Python 3.10 or newer (`python3 --version`).

```bash
cd flow-dictate
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m flow_dictate
```

The first run downloads the Whisper model (about 500 MB for `small.en`), then prints
`Ready`. Click into any text field, hold **right Alt / right Option**, speak, release.

### macOS permissions (one time)

macOS blocks apps from reading keys and the microphone until you allow it.
Open **System Settings → Privacy & Security** and enable your terminal app
(Terminal, iTerm, VS Code, whichever you launch from) under:

- **Microphone**
- **Accessibility** (needed to paste into other apps)
- **Input Monitoring** (needed to see the hotkey while another app is focused)

If the hotkey does nothing, this is almost always why. Quit and relaunch the terminal
after granting access.

### Windows

No special permissions. If pasting into an app running as Administrator fails,
run your terminal as Administrator too.

### Linux

Works on X11. On Wayland, global hotkeys from Python are restricted by the compositor;
bind a desktop shortcut to toggle mode instead (see `mode` below).

## Everyday use

```
python -m flow_dictate                       start (leave it running in a terminal)
python -m flow_dictate learn "Wispr Flow"    teach it a name, product or jargon
python -m flow_dictate correct "whisper flow" "Wispr Flow"
python -m flow_dictate history               see recent dictations, raw vs final
```

### How "learning" works

Wispr Flow's learning is mostly a personal dictionary plus remembered corrections.
Here those are two plain-text files in `~/.flow-dictate/` that you own:

- **dictionary.txt**: words the recognizer should know. They are fed to Whisper as
  context before every transcription, which is what makes it spell "Wispr" instead of
  "whisper". Add to it with `learn`, or edit the file.
- **corrections.json**: `{"what it heard": "what you meant"}`. Applied to every
  transcript, with case matching and whole-word boundaries. Add with `correct`.
  Using `correct` also adds the right spelling to the dictionary.

When you notice a repeated mistake, run `history`, copy the wrong phrase, and `correct`
it once. It never happens again. Edits apply immediately, no restart.

## Settings

`~/.flow-dictate/config.json` is created on first run. The ones worth knowing:

| Key | Default | Notes |
|-----|---------|-------|
| `hotkey` | `alt_r` | Right Alt / right Option. Others: `ctrl_r`, `cmd_r`, `f8`, `pause`, `scroll_lock` |
| `mode` | `hold` | `toggle` = tap to start, tap again to stop |
| `whisper_model` | `small.en` | See table below |
| `language` | `en` | Set to another code and use a non-`.en` model for other languages |
| `input_device` | `null` | Run `python -m flow_dictate devices` to pick a specific mic |
| `paste_method` | `paste` | `type` types character by character for the rare app that blocks paste |
| `fillers` | um, uh, ... | Words the free cleaner removes. Add `"like,"` if you want it stripped too |
| `llm_cleanup` | `off` | `off`, `ollama`, or `claude` |
| `style_notes` | `""` | Hints for the LLM pass, e.g. "casual, short sentences" |

### Choosing a Whisper model

| Model | Size | Speed on a laptop CPU | Accuracy |
|-------|------|------------------------|----------|
| `base.en` | 150 MB | instant | okay for short notes |
| `small.en` | 500 MB | about 1 s per 10 s of speech | good, the default |
| `medium.en` | 1.5 GB | 3 to 5 s per 10 s of speech | very good |
| `large-v3-turbo` | 1.6 GB | 2 to 4 s per 10 s of speech | best, multilingual |
| `distil-large-v3` | 1.5 GB | 2 to 3 s per 10 s of speech | near best, English |

With an NVIDIA GPU, `large-v3-turbo` is near instant. Apple Silicon runs on CPU
through this library; `small.en` and `medium.en` are both comfortable there.

### Optional language-model polish

The free rule-based cleanup handles fillers, spoken commands and your corrections.
A language model additionally fixes things like "send it Tuesday, no, Wednesday"
and awkward grammar. Two options:

**Ollama (free, local).** Install Ollama from ollama.com, run `ollama pull llama3.2`,
then set `"llm_cleanup": "ollama"`. Needs roughly 4 GB of RAM. Adds about a second
per dictation on a laptop.

**Claude (API, pay per use).** Set `"llm_cleanup": "claude"` and either run
`ant auth login` or export `ANTHROPIC_API_KEY`. The code uses `claude-opus-5-5` by
default, with server-side fallback enabled so a safety-classifier decline is retried on
another model rather than returned as an error. Opus is more than this task needs; for
dictation, `"claude_model": "claude-haiku-5-5"` is the sensible setting. Rough cost
for 100 dictations a day, every day:

| Model | Cost per month |
|-------|----------------|
| `claude-haiku-5-5` | about $0.30 |
| `claude-opus-5-5` | about $12 |

If the LLM step fails for any reason (offline, no key, rate limit), you still get the
rule-based text. The app never blocks on it.

## Troubleshooting

- **Hotkey does nothing on macOS**: grant Accessibility and Input Monitoring to the
  terminal app, then relaunch it.
- **Text pastes into the terminal instead of the app**: the terminal still had focus.
  Click into the target app first.
- **"Could not open microphone"**: run `python -m flow_dictate devices` and set
  `input_device` to the right one. On macOS, check the Microphone permission.
- **Slow**: switch to `base.en`, or if you have an NVIDIA GPU make sure CUDA is
  installed so `device` resolves to `cuda`.
- **Test without a mic**: `python -m flow_dictate file some.wav` prints raw and
  cleaned transcript for any audio file.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

## Layout

```
flow_dictate/
  app.py        hotkey, recording, Whisper, paste
  cleanup.py    rule-based cleanup + optional Claude/Ollama pass
  learning.py   dictionary, corrections, history
  paths.py      config defaults and file locations
  __main__.py   command line
tests/
```
