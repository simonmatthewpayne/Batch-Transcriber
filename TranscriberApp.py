"""
Whisper Batch Transcriber (GUI) — CUDA + DirectML + CPU

- Auto device: CUDA (NVIDIA) → DirectML (AMD/Intel on Windows) → CPU
- Batch select many media files, save *_transcript.txt next to each
- "Use recommended" model size, progress, status
- Cancel selected (pending) and Cancel all (after current file)
- Bundled-ffmpeg friendly (works with PyInstaller onefile/onedir)
- Bundles base/small/medium models; offers auto-download of 'large' on suitable NVIDIA GPUs

Build-time deps on YOUR machine:
  pip install git+https://github.com/openai/whisper
  # Choose a CUDA PyTorch wheel matching your drivers (example: CUDA 12.1)
  pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
  pip install torch-directml
  pip install tiktoken torchaudio psutil pyinstaller

Project layout before building (next to this .py file):
  models/
    base.pt     (~140 MB)
    small.pt    (~460 MB)
    medium.pt   (~1.4 GB)
  ffmpeg/
    ffmpeg.exe
"""

# --- standard libs ---
import os                         # interacting with the operating system (paths, env vars, etc.)
import sys                        # access to interpreter details (argv, _MEIPASS for PyInstaller, etc.)
import time                       # simple timing (for elapsed seconds)
import shutil                     # file operations (copying, which/lookup)
import threading                  # run long work on a background thread so GUI stays responsive
import platform                   # detect OS info (Windows/Linux/Mac)
import pathlib                    # nicer path handling than os.path
import tkinter as tk              # the core Tkinter GUI toolkit
from tkinter import filedialog, messagebox, ttk  # file pickers, popups, themed widgets

# --- third-party libs ---
import torch                      # PyTorch (gives us CUDA + DirectML tensors, used by Whisper)
import whisper                    # OpenAI's Whisper transcription library

# Try to import psutil for RAM reporting; it's optional and we handle failure gracefully.
try:
    import psutil                 # for system memory info (RAM)
except Exception:
    psutil = None                 # if not installed, we simply won't show RAM

# Try to import torch-directml (AMD/Intel/NVIDIA via DirectML on Windows).
try:
    import torch_directml         # exposes torch_directml.device() → used to get a DML device
    _DML_AVAILABLE = True         # flag so we know DirectML can be used
except Exception:
    _DML_AVAILABLE = False        # if import fails, we don't offer/use DML

# --- runtime base dir (source or PyInstaller onefile/onedir) ---
# When packaged with PyInstaller --onefile, files are unpacked to a temp folder and
# sys._MEIPASS points to that runtime directory. When running from source, just use the script's folder.
BASE_DIR = pathlib.Path(getattr(sys, "_MEIPASS", pathlib.Path(__file__).parent))

# --- ffmpeg detection / PATH prep ---
# We expect ffmpeg.exe to be bundled inside ./ffmpeg/ffmpeg.exe relative to BASE_DIR (works in dev + packaged).
FFMPEG_PATH = BASE_DIR / "ffmpeg" / "ffmpeg.exe"
if FFMPEG_PATH.exists():
    # If we found a local ffmpeg, prepend its folder to PATH so subprocess calls can find it.
    os.environ["PATH"] = str(FFMPEG_PATH.parent) + os.pathsep + os.environ.get("PATH", "")
# Note: if it's missing, we warn the user later in the GUI (in __init__), so we keep quiet here.

# --- bundled Whisper models ---
# We want the EXE to contain base/small/medium so it works offline.
# These paths are resolved at runtime (dev or packaged) relative to BASE_DIR.
BUNDLED_MODELS = {
    "base":   BASE_DIR / "models" / "base.pt",    # multilingual "base" weight file
    "small":  BASE_DIR / "models" / "small.pt",   # multilingual "small" weight file
    "medium": BASE_DIR / "models" / "medium.pt",  # multilingual "medium" weight file
    # You *could* also add "tiny" if you want a super-small model.
}

def resolve_model_path(name: str) -> str:
    """
    Return a filesystem path to the bundled .pt if we have it; otherwise return the model name
    so whisper will use its cache / download mechanism (e.g., for 'large').
    """
    p = BUNDLED_MODELS.get(name)     # look up "base/small/medium" -> Path(...)
    if p and p.exists():             # make sure the file actually exists in packaged/dist
        return str(p)                # use the local file (works offline)
    return name                      # otherwise let whisper handle download/cache (online once)

def _gb(bytes_val: int) -> float:
    """Helper to convert bytes → gigabytes with one decimal place."""
    return round(bytes_val / (1024 ** 3), 1)

def detect_hardware() -> dict:
    """
    Probe the machine for friendly recommendations:
    - CPU cores
    - Total system RAM (if psutil available)
    - CUDA GPU name + VRAM (if available)
    """
    info = {
        "gpu_available": False,      # do we see a CUDA GPU?
        "gpu_name": None,            # CUDA GPU model name (string)
        "gpu_vram_gb": None,         # CUDA GPU VRAM in GB (float)
        "cpu_cores": os.cpu_count() or 1,  # number of logical CPU cores
        "ram_gb": None,              # total system RAM in GB (float)
        "os": platform.system(),     # e.g., 'Windows', 'Linux', 'Darwin'
    }
    # Try to fetch total RAM if psutil is installed.
    if psutil:
        try:
            info["ram_gb"] = _gb(psutil.virtual_memory().total)
        except Exception:
            pass
    # Try to detect a CUDA-capable NVIDIA GPU via torch.
    try:
        if torch.cuda.is_available():                         # checks if CUDA runtime is usable
            p = torch.cuda.get_device_properties(0)           # get properties of GPU 0
            info["gpu_available"] = True
            info["gpu_name"] = p.name                         # human-friendly GPU name
            info["gpu_vram_gb"] = _gb(p.total_memory)         # VRAM in GB
    except Exception:
        pass
    return info

def recommend_model(info: dict) -> tuple[str, str]:
    """
    Pick a model name ('base'/'small'/'medium'/'large') for the user to try,
    based on CUDA VRAM (preferred) or CPU RAM/cores if no CUDA GPU detected.
    Returns (model_name, reason_string).
    """
    # If we have a CUDA GPU with known VRAM, bias towards bigger models.
    if info.get("gpu_available") and info.get("gpu_vram_gb"):
        v = info["gpu_vram_gb"]
        if v >= 16: m = "large"      # very beefy GPU → suggest 'large' (best accuracy)
        elif v >= 10: m = "large"    # still fine for 'large' usually
        elif v >= 6:  m = "medium"   # mid-tier → 'medium'
        elif v >= 3:  m = "small"    # lower VRAM → 'small'
        else:        m = "base"      # very low VRAM → 'base'
        reason = f"GPU {info.get('gpu_name','')} ({v} GB VRAM)"
    else:
        # Otherwise, base on system RAM + core count (very rough heuristic).
        ram = info.get("ram_gb") or 8
        cores = info.get("cpu_cores") or 4
        if ram >= 24 and cores >= 8: m = "medium"  # more RAM/cores → try 'medium'
        elif ram >= 16:               m = "small"   # decent RAM → 'small'
        else:                         m = "base"    # entry-level → 'base'
        reason = f"CPU {cores} cores, {ram} GB RAM"
    return m, reason

def can_handle_large(dev: str, hw: dict) -> bool:
    """
    Decide if we should *offer* the 'large' model download.
    We only offer on CUDA (NVIDIA) with >= ~12 GB VRAM.
    You can raise this to 16 GB if you want to be stricter.
    """
    if dev != "cuda":                 # only consider 'large' on CUDA for sanity/speed
        return False
    try:
        vram = hw.get("gpu_vram_gb") or 0
        return vram >= 12
    except Exception:
        return False

# --- media types we accept ---
# These are common audio/video extensions that ffmpeg can decode and Whisper can process.
SUPPORTED = (
    ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".mts", ".m2ts", ".3gp"
)

class TranscriberApp(tk.Tk):
    """Main Tkinter window for the batch transcriber."""
    def __init__(self):
        super().__init__()                           # initialise the Tk root window
        self.title("Whisper Batch Transcriber")      # window title text
        self.geometry("900x560")                     # fixed size (pixels)
        self.resizable(False, False)                 # prevent resizing (keep layout simple)

        # --- state variables used across the app ---
        self.files: list[str] = []                   # user-chosen input file paths
        self.model = tk.StringVar(value="small")     # selected model; StringVar binds to UI combobox
        self.device = tk.StringVar(value="auto")     # selected device (auto/cpu/cuda/dml)
        self.rec_label_var = tk.StringVar(value="")  # text shown for "Recommended: ..."
        self._stop_after_current = False             # flag to stop after finishing current file
        self._running = False                        # are we currently transcribing?
        self._cancelled: set[str] = set()            # file paths marked as cancelled
        self._last_output_dir: str | None = None     # folder of last processed file (for "Open output")

        # Detect hardware once at startup for recs; cheap enough to call again if needed later.
        self.hw = detect_hardware()
        rec_model, rec_reason = recommend_model(self.hw)      # e.g. ('small', 'CPU 8 cores, 16 GB RAM')
        self.model.set(rec_model)                              # preselect the recommended model
        self.device.set("auto")                                # default to auto device choice
        self.rec_label_var.set(f"Recommended: {rec_model}  ·  {rec_reason}")  # show why

        # --- build the GUI layout ---
        pad = {"padx": 10, "pady": 8}              # default padding for neat spacing

        # Top row: buttons + model/device selectors.
        top = tk.Frame(self)                        # container frame
        top.pack(fill="x", **pad)                   # add to window (stretch horizontally)

        tk.Button(top, text="Select files…",        # opens a file picker for many files
                  command=self.pick_files).pack(side="left")

        tk.Label(top, text="Model:").pack(side="left", padx=(16, 4))  # label to the left of combobox
        self.model_box = ttk.Combobox(             # dropdown for model choices
            top, textvariable=self.model,
            values=["tiny", "base", "small", "medium", "large"],  # the visible choices
            width=10, state="readonly"             # readonly so user must pick from list
        )
        self.model_box.pack(side="left")

        tk.Label(top, text="Device:").pack(side="left", padx=(16, 4))  # label to the left of device combobox
        self.device_box = ttk.Combobox(            # dropdown for device choice
            top, textvariable=self.device,
            values=["auto", "cpu", "cuda", "dml"], # include DirectML option explicitly
            width=8, state="readonly"
        )
        self.device_box.pack(side="left")
        self.device_box.bind("<<ComboboxSelected>>", self.on_device_changed)  # when changed, refresh recommendation

        self.run_btn = tk.Button(top, text="Run",  # start the transcription
                                  command=self.run_clicked)
        self.run_btn.pack(side="right")

        # Recommendation row: shows the recommendation text + a "Use recommended" button.
        rec_row = tk.Frame(self)
        rec_row.pack(fill="x", padx=12)
        tk.Label(rec_row, textvariable=self.rec_label_var, fg="#555").pack(side="left", anchor="w")
        tk.Button(rec_row, text="Use recommended", command=self.apply_recommended).pack(side="right")

        # Middle section: a listbox showing queued files + scrollbar.
        mid = tk.Frame(self)
        mid.pack(fill="both", expand=True, **pad)
        self.listbox = tk.Listbox(mid, height=14, selectmode=tk.EXTENDED)  # allow multi-select for cancellation
        self.listbox.pack(fill="both", expand=True, side="left")
        sb = tk.Scrollbar(mid, command=self.listbox.yview)                 # scrollbar tied to listbox
        sb.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=sb.set)                          # keep scrollbar in sync

        # Action row: cancellation buttons + open output folder.
        actions = tk.Frame(self)
        actions.pack(fill="x", **pad)
        tk.Button(actions, text="Cancel selected (pending)",  # marks highlighted items as cancelled
                  command=self.cancel_selected).pack(side="left")
        tk.Button(actions, text="Cancel all (after this file)",  # sets a flag to stop after current item
                  command=self.cancel_all).pack(side="left", padx=8)
        tk.Button(actions, text="Open output folder",           # convenience to open last folder
                  command=self.open_output_folder).pack(side="right")

        # Bottom row: progress bar + status label.
        bottom = tk.Frame(self)
        bottom.pack(fill="x", **pad)
        self.progress = ttk.Progressbar(bottom, mode="determinate")  # determinate: we know how many files total
        self.progress.pack(fill="x")
        self.status = tk.Label(bottom, text="Idle.")                 # text status updates (current file, errors, etc.)
        self.status.pack(anchor="w")

        # Startup environment hints, to help users fix problems early.
        if shutil.which("ffmpeg") is None:  # if ffmpeg not found in PATH (and we didn't add a local one)
            self.status.config(text="⚠ ffmpeg not found on PATH. (Bundled ffmpeg recommended.)")
        if self.hw.get("gpu_available"):    # if CUDA GPU present, show its name + VRAM
            self.status.config(text=f"CUDA GPU: {self.hw.get('gpu_name')} ({self.hw.get('gpu_vram_gb')} GB VRAM)")
        elif _DML_AVAILABLE:                # otherwise if DirectML is imported, note it's available
            self.status.config(text="DirectML available (AMD/Intel). Auto will use 'dml' if CUDA is not present.")

    # --- small helper: safely update the status text from any thread ---
    def _set_status(self, s: str):
        # Tkinter must be updated from the main thread; .after queues a UI update safely.
        self.status.after(0, lambda: self.status.config(text=s))

    def _resolve_device(self) -> str:
        """
        Decide which device string to use for Whisper:
        - 'auto' → try CUDA, else try DirectML, else CPU
        - Forced 'cuda'/'dml'/'cpu' → fall back to 'cpu' if unavailable (so it never crashes).
        """
        d = self.device.get()               # read current UI selection
        if d == "auto":
            if torch.cuda.is_available():   # prefer CUDA if available
                return "cuda"
            if _DML_AVAILABLE:              # else try DirectML if imported successfully
                try:
                    _ = torch_directml.device()  # simple probe (will raise if not usable)
                    return "dml"
                except Exception:
                    pass
            return "cpu"                    # final fallback
        if d == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if d == "dml":
            return "dml" if _DML_AVAILABLE else "cpu"
        return "cpu"

    # When the user changes the device dropdown, refresh the recommendation text.
    def on_device_changed(self, _e=None):
        forced = self.device.get()          # what the user picked
        temp = dict(self.hw)                # copy of our detected hardware baseline
        if forced == "cpu":
            # If forced CPU, pretend we have no GPU for recommendation logic.
            temp["gpu_available"] = False
            temp["gpu_vram_gb"] = None
        elif forced == "auto":
            # If auto, re-detect (in case something changed).
            temp = detect_hardware()
        rec_model, rec_reason = recommend_model(temp)
        self.rec_label_var.set(f"Recommended: {rec_model}  ·  {rec_reason}")

    def apply_recommended(self):
        """
        Set the model dropdown to whatever the recommendation suggests.
        Keep DML conservative: don't auto-pick medium/large on DirectML.
        """
        forced = self.device.get()
        temp = detect_hardware() if forced == "auto" else dict(self.hw)
        if forced == "cpu":
            temp["gpu_available"] = False
            temp["gpu_vram_gb"] = None
        rec_model, _ = recommend_model(temp)
        # Be conservative on DML: clamp to 'small' if recommendation says medium/large.
        dev = self._resolve_device()
        if dev == "dml" and rec_model in ("medium", "large"):
            rec_model = "small"
        self.model.set(rec_model)
        # Trigger combobox notification (optional, keeps UI in sync if something listens to it).
        self.model_box.event_generate("<<ComboboxSelected>>")

    def pick_files(self):
        """
        Open a file dialog for the user to pick many media files.
        Populate the listbox and prepare the progress bar.
        """
        chosen = filedialog.askopenfilenames(
            title="Choose audio/video to transcribe",
            filetypes=[("Media files", " ".join(f"*{e}" for e in SUPPORTED)),
                       ("All files", "*.*")]
        )
        if not chosen:
            return
        self.files = list(chosen)                 # save the list of files
        self._cancelled.clear()                   # reset cancelled set
        self.listbox.delete(0, tk.END)            # clear the UI list
        for f in self.files:                      # add each file path to the listbox
            self.listbox.insert(tk.END, f)
        self.progress["maximum"] = len(self.files) # determinate bar: set total
        self.progress["value"] = 0                 # reset progress to 0
        self._set_status(f"{len(self.files)} file(s) selected.")

    def cancel_selected(self):
        """
        Mark highlighted listbox entries as 'to be skipped' if they haven't run yet.
        """
        if not self.files:
            return
        idxs = list(self.listbox.curselection())  # selected indices in the listbox
        if not idxs:
            return
        for i in idxs:
            p = self.files[i]                     # get the file path at that index
            self._cancelled.add(p)                # remember it's cancelled
            self.listbox.delete(i)                # visually mark as cancelled (replace row)
            self.listbox.insert(i, f"[CANCELLED] {p}")
        self._set_status(f"Marked {len(idxs)} file(s) to skip.")

    def cancel_all(self):
        """
        Stop after the *current* file is done (safer than hard-stopping in the middle).
        """
        self._stop_after_current = True
        self._set_status("Will stop after the current file finishes…")

    def _preflight(self):
        """
        Quick checks before starting the worker thread:
        - ensure files chosen
        - ensure ffmpeg available (PATH or bundled)
        - validate device availability
        - warn user about heavy model choices on CPU/DML
        """
        if not self.files:
            raise RuntimeError("No files selected.")
        if shutil.which("ffmpeg") is None:
            # If PATH can't find ffmpeg, ask the user to bundle it or install it.
            raise RuntimeError("ffmpeg not found. Put ffmpeg\\ffmpeg.exe next to the app or install ffmpeg on PATH.")
        dev = self._resolve_device()
        if dev == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Device set to CUDA but no CUDA GPU available.")
        if dev == "dml" and not _DML_AVAILABLE:
            raise RuntimeError("Device set to DirectML but torch-directml is not available.")
        # If user insists on 'medium' or 'large' without CUDA, prompt a warning (can be *very* slow).
        if dev in ("cpu", "dml") and self.model.get() in ("medium", "large"):
            if not messagebox.askyesno(
                "Heavy model on current device",
                f"You chose '{self.model.get()}' on {dev.upper()}.\n"
                "This may be very slow or run out of memory.\n\nContinue anyway?"
            ):
                # Raise to cancel the run if user said No.
                raise RuntimeError("Aborted by user (heavy model on non-CUDA device).")

    def run_clicked(self):
        """
        Called when the user presses 'Run'.
        - Guard against double starts.
        - Run preflight checks with nice error popups.
        - Start the worker on a background thread (so GUI doesn't freeze).
        """
        if self._running:
            messagebox.showinfo("Whisper Transcriber", "Already running.")
            return
        try:
            self._preflight()                     # may raise; caught below
        except Exception as e:
            messagebox.showerror("Preflight failed", str(e))
            return
        # Start the transcription loop in a daemon thread (dies if the main app exits).
        threading.Thread(target=self._transcribe_batch, daemon=True).start()

    def _transcribe_batch(self):
        """
        Background worker:
        - Resolve device + model (offer auto-upgrade to 'large' on beefy CUDA)
        - Load the model (from bundled file or via Whisper cache)
        - Iterate over files, transcribe, write *_transcript.txt next to each
        - Update progress/status; honour cancel flags
        """
        self._running = True
        self._stop_after_current = False
        try:
            start = time.perf_counter()           # time the whole batch run
            dev = self._resolve_device()          # 'cuda' / 'dml' / 'cpu'

            model_name = self.model.get()         # current model selection (string)

            # If we have a strong CUDA GPU, offer one-time auto-download of 'large' (~3 GB).
            try:
                if model_name in ("base", "small", "medium") and can_handle_large(dev, self.hw):
                    if messagebox.askyesno(
                        "High-end GPU detected",
                        "A CUDA GPU with ample VRAM was detected.\n"
                        "Would you like to auto-download and use the 'large' model (~3 GB) for best accuracy?\n\n"
                        "This happens once and is cached for future runs."
                    ):
                        model_name = "large"      # switch to 'large' (will download if not cached)
            except Exception:
                # If anything goes wrong with the prompt (rare), just continue with the chosen model.
                pass

            # For base/small/medium: use the bundled .pt file if present (offline).
            # For large: pass "large" so whisper downloads/uses cache.
            model_arg = resolve_model_path(model_name)

            self._set_status(f"Loading model '{model_name}'…")
            model = whisper.load_model(model_arg, device=dev)  # create the Whisper model on the target device

            done = 0                              # how many files completed
            total = len(self.files)               # total number of files queued
            self.progress["maximum"] = total      # configure progress bar
            self.progress["value"] = 0

            # Iterate each selected file in order.
            for idx, path in enumerate(self.files, start=1):
                if path in self._cancelled:       # if user marked this file as cancelled
                    self._set_status(f"[{idx}/{total}] Skipped (cancelled): {os.path.basename(path)}")
                    continue

                base, _ = os.path.splitext(path)  # split "C:\...\file.mp3" -> ("C:\...\file", ".mp3")
                out_txt = base + "_transcript.txt"  # output file path next to input
                self._last_output_dir = os.path.dirname(path)  # remember folder for "Open output"
                self._set_status(f"[{idx}/{total}] Transcribing: {os.path.basename(path)} on {dev} ({model_name})")

                try:
                    # Run Whisper. language="en" constrains transcription to English
                    # (remove if you want language auto-detection / multilingual).
                    result = model.transcribe(path, language="en")
                    # Write the plain text transcript to a .txt file.
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(result.get("text", "").strip())
                except Exception as e:
                    # If we hit an error, write a small sidecar file with the error message.
                    with open(base + "_transcript_ERROR.txt", "w", encoding="utf-8") as f:
                        f.write(f"Error transcribing {path}\n\n{e}")

                done += 1                          # increment completed count
                self.progress["value"] = done      # bump progress bar
                if self._stop_after_current:       # if user pressed "Cancel all", stop after this file
                    self._set_status("Batch cancelled after current file (by user).")
                    break

            # Pop a friendly completion message + show elapsed time.
            elapsed = time.perf_counter() - start
            messagebox.showinfo(
                "Whisper Transcriber",
                "Done.\nTranscripts saved next to originals."
                f"\nElapsed: {elapsed:.1f}s"
            )
            self._set_status(f"✅ Finished {done} file(s) in {elapsed:.1f}s.")
        finally:
            # Always clear the running flag, even if we error out or the user cancels.
            self._running = False

    def open_output_folder(self):
        """
        Convenience: open the folder of the last processed file in File Explorer.
        """
        if self._last_output_dir and os.path.isdir(self._last_output_dir):
            os.startfile(self._last_output_dir)   # Windows-specific; opens Explorer at that path
        else:
            messagebox.showinfo("Open folder", "No output folder yet.")

# Standard "entry point" guard: only run the GUI if this file is the main program.
if __name__ == "__main__":
    app = TranscriberApp()   # create the Tkinter app instance
    app.mainloop()           # hand control to Tk's event loop (keeps the window alive)
