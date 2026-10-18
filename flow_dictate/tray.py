"""System tray mode: no terminal window. An icon by the clock shows the state
(grey = ready, red = listening, amber = transcribing) and a right-click menu
offers pause, settings, the log, and quit."""
import os
import subprocess
import sys
import threading
import webbrowser

from .paths import CONFIG_PATH, DATA_DIR, DICTIONARY_PATH, ensure_data_dir

LOG_PATH = DATA_DIR / "log.txt"
COLORS = {"idle": (120, 120, 120), "listening": (220, 50, 50), "busy": (240, 170, 30),
          "paused": (60, 60, 60), "error": (160, 40, 160)}


def redirect_output_to_log() -> None:
    """Without a console, print() has nowhere to go. Send it to a log file."""
    ensure_data_dir()
    try:
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > 1_000_000:
            LOG_PATH.unlink()
    except OSError:
        pass
    f = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = f


def make_icon(state: str):
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = COLORS.get(state, COLORS["idle"])
    # A microphone: capsule + stand.
    d.rounded_rectangle((20, 6, 44, 40), radius=12, fill=color)
    d.arc((12, 22, 52, 52), start=0, end=180, fill=color, width=5)
    d.line((32, 52, 32, 60), fill=color, width=5)
    d.line((20, 60, 44, 60), fill=color, width=5)
    if state == "paused":
        d.rectangle((24, 16, 29, 32), fill=(255, 255, 255))
        d.rectangle((35, 16, 40, 32), fill=(255, 255, 255))
    return img


def write_ico() -> str:
    """An .ico for shortcuts. Returns its path."""
    ensure_data_dir()
    path = DATA_DIR / "flow-dictate.ico"
    if not path.exists():
        make_icon("idle").resize((256, 256)).save(path, format="ICO", sizes=[(256, 256), (48, 48), (32, 32), (16, 16)])
    return str(path)


def open_path(path) -> None:
    path = str(path)
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        webbrowser.open(path)


def run_tray(cfg: dict) -> None:
    import pystray
    from .app import App

    redirect_output_to_log()
    print("--- Flow Dictate tray starting ---")
    icon = pystray.Icon("Flow Dictate", make_icon("idle"), "Flow Dictate: loading model...")
    state = {"app": None, "status": "loading"}

    def set_state(s: str) -> None:
        if state["app"] is not None and state["app"].paused and s == "idle":
            s = "paused"
        state["status"] = s
        icon.icon = make_icon(s)
        hotkey = getattr(state["app"], "hotkey_name", "the hotkey")
        icon.title = {"idle": f"Flow Dictate: hold {hotkey} to talk",
                      "listening": "Flow Dictate: listening",
                      "busy": "Flow Dictate: transcribing",
                      "paused": "Flow Dictate: paused",
                      "error": "Flow Dictate: failed to start, see log"}.get(s, "Flow Dictate")

    def toggle_pause(icon_, item) -> None:
        app = state["app"]
        if app is None:
            return
        app.paused = not app.paused
        set_state("paused" if app.paused else "idle")

    def is_paused(item) -> bool:
        return bool(state["app"] and state["app"].paused)

    def quit_app(icon_, item) -> None:
        icon.stop()

    icon.menu = pystray.Menu(
        pystray.MenuItem(lambda item: f"Flow Dictate ({state['status']})", None, enabled=False),
        pystray.MenuItem("Pause dictation", toggle_pause, checked=is_paused),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open settings (config.json)", lambda i, it: open_path(CONFIG_PATH)),
        pystray.MenuItem("Open dictionary.txt", lambda i, it: open_path(DICTIONARY_PATH)),
        pystray.MenuItem("Open data folder", lambda i, it: open_path(DATA_DIR)),
        pystray.MenuItem("View log", lambda i, it: open_path(LOG_PATH)),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", quit_app),
    )

    def boot() -> None:
        # Model loading takes a few seconds; do it after the icon is visible.
        try:
            app = App(cfg, on_state=set_state)
            state["app"] = app
            app.start()
            set_state("idle")
            print(f"Ready. Hotkey: {app.hotkey_name}")
        except Exception as e:
            print(f"[fatal] {type(e).__name__}: {e}")
            set_state("error")

    def setup(icon_) -> None:
        icon_.visible = True
        threading.Thread(target=boot, daemon=True).start()

    icon.run(setup=setup)
    print("--- Flow Dictate tray stopped ---")
    os._exit(0)  # the keyboard listener thread would otherwise keep the process alive


def install_autostart(project_dir: str, remove: bool = False) -> str:
    """Windows: shortcuts on the Desktop and in the Startup folder that launch the
    tray without a console. Returns a message for the user."""
    if sys.platform != "win32":
        return ("Autostart shortcuts are only automated on Windows. On macOS add "
                f"{project_dir}/start.sh tray to System Settings > General > Login Items.")
    vbs = os.path.join(project_dir, "FlowDictate.vbs")
    ico = write_ico()
    startup = os.path.join(os.environ["APPDATA"], r"Microsoft\Windows\Start Menu\Programs\Startup")
    desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")
    targets = [os.path.join(startup, "Flow Dictate.lnk"), os.path.join(desktop, "Flow Dictate.lnk")]
    if remove:
        for t in targets:
            try:
                os.remove(t)
            except FileNotFoundError:
                pass
        return "Removed the Startup and Desktop shortcuts."
    script = "; ".join(
        f'$s = (New-Object -ComObject WScript.Shell).CreateShortcut("{t}"); '
        f'$s.TargetPath = "wscript.exe"; $s.Arguments = \'"{vbs}"\'; '
        f'$s.WorkingDirectory = "{project_dir}"; $s.IconLocation = "{ico}"; '
        f'$s.Description = "Flow Dictate: hold a key, talk, release"; $s.Save()'
        for t in targets)
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], check=True)
    return ("Done. 'Flow Dictate' is on your Desktop and will start automatically when you log in.\n"
            "Start it now by double-clicking the Desktop shortcut.")
