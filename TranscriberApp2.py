# Modernised UI for Whisper Batch Transcriber (PySide6)
# ----------------------------------------------------
# Goals
# - A clean, resizable, dark-mode-friendly UI
# - Drag & drop file queue with per-file status + progress
# - Side control panel (Model, Device, Use Recommended)
# - Start / Cancel buttons, overall progress + status bar
# - Keeps your existing device detection & model recommendation logic
#
# Packaging: PyInstaller works well for PySide6. Bundle ffmpeg + models like before.
#
# NOTE: This file focuses on UI + wiring. Replace the stub `transcribe_file` with your
# Whisper call. See the TODO sections.

from __future__ import annotations
import os
import sys
import time
import shutil
import platform
import pathlib
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import re

# Third-party
from PySide6 import QtCore, QtGui, QtWidgets

# Optional deps; we handle if missing
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    import torch_directml  # type: ignore
    _DML_AVAILABLE = True
except Exception:
    _DML_AVAILABLE = False

# --- Whisper asset path safety for dev runs (not needed in final build) ---
try:
    import whisper, whisper.audio as wa
    # If it ever points at a _MEI path (PyInstaller temp), redirect to package assets
    if "_MEI" in str(getattr(sys, "_MEIPASS", "")) or not os.path.exists(wa.MEL_FILTERS_PATH):
        pkg_root = pathlib.Path(whisper.__file__).parent
        cand = pkg_root / "assets" / "mel_filters.npz"
        if cand.exists():
            wa.MEL_FILTERS_PATH = str(cand)
except Exception:
    pass

TQDM_PERCENT_RE = re.compile(r'(\d{1,3})%\|')      # " 40%|████..."
TQDM_FRAC_RE    = re.compile(r'(\d{1,})/(\d{1,})') # "16128/40679"

# ------------------------------
# Constants & Helpers
# ------------------------------
BASE_DIR = pathlib.Path(getattr(sys, "_MEIPASS", pathlib.Path(__file__).parent))
FFMPEG_PATH = BASE_DIR / "ffmpeg" / "ffmpeg.exe"
SUPPORTED = (
    ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".mts", ".m2ts", ".3gp"
)

if FFMPEG_PATH.exists():
    os.environ["PATH"] = str(FFMPEG_PATH.parent) + os.pathsep + os.environ.get("PATH", "")

def _gb(bytes_val: int) -> float:
    return round(bytes_val / (1024 ** 3), 1)

def detect_hardware() -> dict:
    info = {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": None,
        "os": platform.system(),
    }
    if psutil:
        try:
            info["ram_gb"] = _gb(psutil.virtual_memory().total)
        except Exception:
            pass
    try:
        if torch and hasattr(torch, "cuda") and torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu_available"] = True
            info["gpu_name"] = p.name
            info["gpu_vram_gb"] = _gb(p.total_memory)
    except Exception:
        pass
    return info

def recommend_model(info: dict) -> tuple[str, str]:
    if info.get("gpu_available") and info.get("gpu_vram_gb"):
        v = info["gpu_vram_gb"]
        if v >= 16:
            m = "large"
        elif v >= 10:
            m = "large"
        elif v >= 6:
            m = "medium"
        elif v >= 3:
            m = "small"
        else:
            m = "base"
        reason = f"GPU {info.get('gpu_name','')} ({v} GB VRAM)"
    else:
        ram = info.get("ram_gb") or 8
        cores = info.get("cpu_cores") or 4
        if ram >= 24 and cores >= 8:
            m = "medium"
        elif ram >= 16:
            m = "small"
        else:
            m = "base"
        reason = f"CPU {cores} cores, {ram} GB RAM"
    return m, reason

def resolve_device(selection: str) -> str:
    sel = selection.lower()
    if sel == "auto":
        if torch and hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        if _DML_AVAILABLE:
            try:
                _ = torch_directml.device()
                return "dml"
            except Exception:
                pass
        return "cpu"
    if sel == "cuda":
        return "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    if sel == "dml":
        return "dml" if _DML_AVAILABLE else "cpu"
    return "cpu"

# ------------------------------
# Data model
# ------------------------------
@dataclass
class QueueItem:
    path: str
    status: str = "Queued"
    progress: int = 0  # 0-100

class FileJob(QtCore.QObject):
    progressChanged = QtCore.Signal(int, int)  # row, percent
    started         = QtCore.Signal(int, str)  # row, basename
    finished        = QtCore.Signal(int, int)  # row, exitCode
    error           = QtCore.Signal(int, str)  # row, message

    def __init__(self, row: int, cmd: list[str], workdir: str | None = None,
                 parent=None, display_name: str | None = None):
        super().__init__(parent)
        self.row = row
        self._cmd = cmd
        self._workdir = workdir
        self._display_name = display_name or ""
        self._buf = ""

        self._proc = QtCore.QProcess(self)
        self._proc.setProcessChannelMode(QtCore.QProcess.SeparateChannels)
        self._proc.readyReadStandardError.connect(self._on_stderr)
        self._proc.readyReadStandardOutput.connect(self._on_stdout)
        self._proc.finished.connect(self._on_finished)

    def start(self):
        program, *args = self._cmd
        if self._workdir:
            self._proc.setWorkingDirectory(self._workdir)

        name = self._display_name or os.path.basename(self._cmd[-1])
        self.started.emit(self.row, name)

        self._proc.start(program, args)

    def kill(self):
        try:
            self._proc.kill()
        except Exception:
            pass

    # --- internal: parse tqdm from STDERR ---
    def _on_stderr(self):
        data = bytes(self._proc.readAllStandardError()).decode("utf-8", "ignore")
        self._buf += data
        # tqdm streams partial lines; consume complete ones only
        parts = self._buf.splitlines(keepends=True)
        tail = ""
        for line in parts:
            if not line.endswith("\n") and not line.endswith("\r"):
                tail = line
                continue

            # Try match "NN%|"
            m = TQDM_PERCENT_RE.search(line)
            if m:
                pct = min(100, max(0, int(m.group(1))))
                self.progressChanged.emit(self.row, pct)
                continue

            # Try match "current/total"
            m = TQDM_FRAC_RE.search(line)
            if m:
                cur = int(m.group(1))
                tot = int(m.group(2)) or 1
                pct = max(0, min(100, int(cur * 100 / tot)))
                self.progressChanged.emit(self.row, pct)

        self._buf = tail

    def _on_stdout(self):
        _ = bytes(self._proc.readAllStandardOutput())

    def _on_finished(self, exitCode, exitStatus):
        self.progressChanged.emit(self.row, 100 if exitCode == 0 else 0)
        self.finished.emit(self.row, int(exitCode))

# ------------------------------
# (Legacy) Worker — kept for reference, not used in current run path
# ------------------------------
class BatchWorker(QtCore.QObject):
    file_started = QtCore.Signal(int, str)
    file_progress = QtCore.Signal(int, int)  # row, percent
    file_done = QtCore.Signal(int, str)
    file_error = QtCore.Signal(int, str)
    batch_done = QtCore.Signal(float, int)   # elapsed, processed

    def transcribe_file(self, model, in_path: str, row: int):
        """
        Transcribe a single file and write outputs next to the source:
        - <name>_transcript.txt
        Emits progress at a few milestones.
        """
        p = Path(in_path)
        stem = p.with_suffix("")  # remove extension

        # Get media duration (for rough progress later)
        duration = self._probe_duration(in_path) or 0.0
        # Decoding/transcribing
        fp16 = (self._device == "cuda")
        result = model.transcribe(str(in_path), fp16=fp16, verbose=False)

        # Write TXT
        txt_path = f"{stem}_transcript.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write((result.get("text") or "").strip() + "\n")

        segments = result.get("segments") or []
        if segments:
            pass

    def _probe_duration(self, path: str) -> Optional[float]:
        """Return duration in seconds using ffprobe if available."""
        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "json", path
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            info = json.loads(out.decode("utf-8", errors="ignore"))
            dur = float(info.get("format", {}).get("duration", 0.0))
            return dur if dur > 0 else None
        except Exception:
            return None

    def _secs_to_timestamp(self, t: float, vtt: bool = False) -> str:
        """Format seconds -> SRT/VTT timestamp."""
        if t < 0: t = 0
        ms = int(round((t - int(t)) * 1000))
        s = int(t) % 60
        m = (int(t) // 60) % 60
        h = int(t) // 3600
        if vtt:
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
        else:
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _write_srt(self, path: str, segments: list):
        with open(path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, start=1):
                start = self._secs_to_timestamp(seg.get("start", 0.0))
                end = self._secs_to_timestamp(seg.get("end", 0.0))
                text = (seg.get("text") or "").strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    def _write_vtt(self, path: str, segments: list):
        with open(path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in segments:
                start = self._secs_to_timestamp(seg.get("start", 0.0), vtt=True)
                end = self._secs_to_timestamp(seg.get("end", 0.0), vtt=True)
                text = (seg.get("text") or "").strip()
                f.write(f"{start} --> {end}\n{text}\n\n")

    def __init__(self, items: List[QueueItem], model: str, device: str, parent=None):
        super().__init__(parent)
        self._items = items
        self._model = model
        self._device = device
        self._cancel_all = False

    @QtCore.Slot()
    def run(self):
        start = time.perf_counter()
        processed = 0
        try:
            import whisper
            model = whisper.load_model(self._model, device=self._device)

            for idx, item in enumerate(self._items):
                if self._cancel_all:
                    break
                if item.status.startswith("[CANCELLED]"):
                    continue

                self.file_started.emit(idx, os.path.basename(item.path))
                try:
                    self.transcribe_file(model, item.path, idx)
                    self.file_done.emit(idx, "Done")
                    processed += 1
                except Exception as e:
                    self.file_error.emit(idx, str(e))
        finally:
            self.batch_done.emit(time.perf_counter() - start, processed)

    def cancel_all(self):
        self._cancel_all = True

# ------------------------------
# Main Window
# ------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Batch Transcriber")
        self.setWindowIcon(QtGui.QIcon("waveform.ico"))
        self.resize(1100, 700)
        self.setAcceptDrops(True)
        self._smooth_targets = {}

        # State
        self.items: List[QueueItem] = []
        self.hw = detect_hardware()
        rec_model, rec_reason = recommend_model(self.hw)

        # Central table
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["File", "Status", "Progress"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setCentralWidget(self.table)

        # Toolbar actions
        tb = QtWidgets.QToolBar("Main")
        tb.setIconSize(QtCore.QSize(18, 18))
        self.addToolBar(tb)

        act_add = QtGui.QAction("Add files…", self)
        act_add.triggered.connect(self.add_files)
        tb.addAction(act_add)

        act_clear = QtGui.QAction("Clear", self)
        act_clear.triggered.connect(self.clear_queue)
        tb.addAction(act_clear)

        tb.addSeparator()

        self.act_run = QtGui.QAction("Run", self)
        self.act_run.setCheckable(False)
        self.act_run.triggered.connect(self.start_batch)
        tb.addAction(self.act_run)

        self.act_cancel = QtGui.QAction("Cancel all", self)
        self.act_cancel.triggered.connect(self.cancel_all)
        self.act_cancel.setEnabled(False)
        tb.addAction(self.act_cancel)

        # Right dock: Controls
        self.controls = QtWidgets.QWidget()
        right = QtWidgets.QDockWidget("Controls", self)
        right.setWidget(self.controls)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, right)

        form = QtWidgets.QFormLayout(self.controls)

        self.combo_model = QtWidgets.QComboBox()
        self.combo_model.addItems(["tiny", "base", "small", "medium", "large"])
        self.combo_model.setCurrentText(rec_model)
        form.addRow("Model:", self.combo_model)

        self.combo_device = QtWidgets.QComboBox()
        self.combo_device.addItems(["auto", "cpu", "cuda", "dml"])
        self.combo_device.setCurrentText("auto")
        form.addRow("Device:", self.combo_device)

        self.lbl_rec = QtWidgets.QLabel(f"Recommended: {rec_model} · {rec_reason}")
        self.lbl_rec.setStyleSheet("color: #666;")
        form.addRow(self.lbl_rec)

        self.btn_apply_rec = QtWidgets.QPushButton("Use recommended")
        self.btn_apply_rec.clicked.connect(lambda: self.combo_model.setCurrentText(rec_model))
        form.addRow(self.btn_apply_rec)

        # Bottom: progress + status bar
        self.overall = QtWidgets.QProgressBar()
        self.status = self.statusBar()
        self.status.addPermanentWidget(self.overall, 1)
        self.overall.setRange(0, 100)
        self.overall.setValue(0)
        self.status.showMessage("Idle.")

        # Context menu for table: cancel selected (pending)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.on_table_menu)

        # Drag & drop hint (optional future overlay)
        hint = QtWidgets.QLabel("Drop audio/video files here to add to the queue")
        hint.setAlignment(QtCore.Qt.AlignCenter)
        hint.setStyleSheet("color:#888; font-style: italic; padding:12px;")
        lay = QtWidgets.QVBoxLayout()
        lay.addWidget(hint)
        w = QtWidgets.QWidget()
        w.setLayout(lay)
        self.setCentralWidget(self.table)  # table is main

        # Sanity hints
        if shutil.which("ffmpeg") is None:
            self.status.showMessage("⚠ ffmpeg not found on PATH. Place ffmpeg/ffmpeg.exe next to the app.")
        elif self.hw.get("gpu_available"):
            self.status.showMessage(f"CUDA GPU: {self.hw.get('gpu_name')} ({self.hw.get('gpu_vram_gb')} GB VRAM)")
        elif _DML_AVAILABLE:
            self.status.showMessage("DirectML available. Auto will use DML if CUDA is not present.")

        # Worker thread placeholders (unused in current sequential run)
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[BatchWorker] = None

    # ------------- Tweened progress helper -------------
    def _tick_smoothing(self):
        remove_rows = []
        for row, target in list(self._smooth_targets.items()):
            w = self.table.cellWidget(row, 2)
            if not isinstance(w, QtWidgets.QProgressBar):
                remove_rows.append(row)
                continue
            val = w.value()
            if val >= target:
                remove_rows.append(row)
                continue
            step = max(1, int((target - val) * 0.2))  # 20% of gap, min 1
            w.setValue(min(target, val + step))
        for r in remove_rows:
            self._smooth_targets.pop(r, None)

    @QtCore.Slot(int, int)
    def on_file_progress(self, row: int, pct: int):
        w = self.table.cellWidget(row, 2)
        if isinstance(w, QtWidgets.QProgressBar):
            if w.value() < pct:
                w.setValue(w.value())
            self._smooth_targets[row] = pct

    @QtCore.Slot(int, str)
    def on_file_started(self, row: int, name: str):
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"Transcribing: {name}"))
        pb = self.table.cellWidget(row, 2)
        if isinstance(pb, QtWidgets.QProgressBar):
            pb.setRange(0, 0)     # marquee until done
            pb.setFormat("Working…")

    @QtCore.Slot(int, str)
    def on_file_done(self, row: int, msg: str):
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem("✅ Finished"))
        pb = self.table.cellWidget(row, 2)
        if isinstance(pb, QtWidgets.QProgressBar):
            pb.setRange(0, 100)   # determinate now
            pb.setValue(100)
            pb.setFormat("Done")
        self._smooth_targets.pop(row, None)
        self.overall.setValue(self.overall.value() + 1)

    @QtCore.Slot(int, str)
    def on_file_error(self, row: int, err: str):
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"❌ Error: {err}"))
        pb = self.table.cellWidget(row, 2)
        if isinstance(pb, QtWidgets.QProgressBar):
            pb.setRange(0, 100)
            pb.setValue(0)
            pb.setFormat("Failed")
        self._smooth_targets.pop(row, None)
        self.overall.setValue(self.overall.value() + 1)

    # ------------- Drag & Drop -------------
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            urls = [u.toLocalFile() for u in e.mimeData().urls()]
            if any(str(p).lower().endswith(SUPPORTED) for p in urls):
                e.acceptProposedAction()
                return
        e.ignore()

    def dropEvent(self, e: QtGui.QDropEvent):
        paths = [u.toLocalFile() for u in e.mimeData().urls()]
        media = [p for p in paths if str(p).lower().endswith(SUPPORTED)]
        if media:
            self.enqueue(media)

    # ------------- Queue ops -------------
    def add_files(self):
        dlg = QtWidgets.QFileDialog(self, "Choose audio/video to transcribe")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        if dlg.exec():
            self.enqueue(dlg.selectedFiles())

    def clear_queue(self):
        self.items.clear()
        self.table.setRowCount(0)
        self.overall.setValue(0)
        self.status.showMessage("Queue cleared.")

    def enqueue(self, paths: List[str]):
        added = 0
        for p in paths:
            if not p.lower().endswith(SUPPORTED):
                continue
            item = QueueItem(p)
            self.items.append(item)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(p))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(item.status))

            pb = QtWidgets.QProgressBar()
            pb.setRange(0, 100)
            pb.setValue(0)
            self.table.setCellWidget(row, 2, pb)
            added += 1
        if added:
            self.status.showMessage(f"{added} file(s) added.")

    def on_table_menu(self, pos: QtCore.QPoint):
        menu = QtWidgets.QMenu(self)
        act_cancel_sel = menu.addAction("Cancel selected (pending)")
        action = menu.exec(self.table.mapToGlobal(pos))
        if action == act_cancel_sel:
            for row in set(i.row() for i in self.table.selectedIndexes()):
                p = self.table.item(row, 0).text()
                self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"[CANCELLED] {p}"))
                if 0 <= row < len(self.items):
                    self.items[row].status = f"[CANCELLED] {p}"

    def _build_whisper_cli(self, input_path: str) -> list[str]:
        model_choice = self.combo_model.currentText()
        device_choice = resolve_device(self.combo_device.currentText())
        out_dir = os.path.dirname(input_path)

        cmd = [
            sys.executable, "-m", "whisper", input_path,
            "--model", model_choice,
            "--task", "transcribe",
            "--output_format", "txt",
            "--output_dir", out_dir,
            "--verbose", "False",
        ]
        if device_choice != "cuda":
            cmd += ["--fp16", "False"]
        return cmd

    # ------------- Run / Cancel -------------
    def start_batch(self):
        if not self.items:
            QtWidgets.QMessageBox.information(self, "Whisper Transcriber", "No files in the queue.")
            return
        if shutil.which("ffmpeg") is None:
            QtWidgets.QMessageBox.critical(self, "Preflight failed",
                                        "ffmpeg not found. Put ffmpeg/ffmpeg.exe next to the app or install ffmpeg on PATH.")
            return

        model_choice = self.combo_model.currentText()
        device_choice = resolve_device(self.combo_device.currentText())
        if device_choice in ("cpu", "dml") and model_choice in ("medium", "large"):
            res = QtWidgets.QMessageBox.question(
                self, "Heavy model on current device",
                f"You chose '{model_choice}' on {device_choice.upper()}.\nThis may be very slow or run out of memory.\n\nContinue anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if res != QtWidgets.QMessageBox.Yes:
                return

        # Sequential QProcess jobs (no QThread worker)
        self._thread = None
        self._worker = None

        self.act_run.setEnabled(False)
        self.act_cancel.setEnabled(True)
        self.status.showMessage("Running…")
        self.overall.setRange(0, len(self.items))
        self.overall.setValue(0)

        self._jobs: list[FileJob] = []
        for row, item in enumerate(self.items):
            if item.status.startswith("[CANCELLED]"):
                continue
            cmd = self._build_whisper_cli(item.path)
            job = FileJob(
                row=row,
                cmd=cmd,
                workdir=None,
                parent=self,
                display_name=os.path.basename(item.path)
            )
            job.started.connect(self._on_job_started)
            job.progressChanged.connect(self._on_job_progress)
            job.finished.connect(self._on_job_finished)
            self._jobs.append(job)

        self._run_index = 0
        self._run_next_job()

    def _run_next_job(self):
        # Advance to next non-cancelled job
        while self._run_index < len(self._jobs):
            job = self._jobs[self._run_index]
            row = job.row
            it = self.table.item(row, 1)
            if it and it.text().startswith("[CANCELLED]"):
                self._run_index += 1
                continue
            job.start()
            return

        # All done
        self.status.showMessage("Finished all jobs.")
        self.act_run.setEnabled(True)
        self.act_cancel.setEnabled(False)
        QtWidgets.QMessageBox.information(self, "Whisper Transcriber", "Done.")

    @QtCore.Slot(int, str)
    def _on_job_started(self, row: int, name: str):
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"Transcribing: {name}"))
        pb = self.table.cellWidget(row, 2)
        if isinstance(pb, QtWidgets.QProgressBar):
            pb.setRange(0, 0)   # indeterminate until first % arrives
            pb.setFormat("…")

    @QtCore.Slot(int, int)
    def _on_job_progress(self, row: int, pct: int):
        pb = self.table.cellWidget(row, 2)
        if isinstance(pb, QtWidgets.QProgressBar):
            if pb.maximum() == 0:
                pb.setRange(0, 100)   # switch to determinate once we have real %s
            pb.setValue(pct)
            pb.setFormat(f"{pct}%")

    @QtCore.Slot(int, int)
    def _on_job_finished(self, row: int, exitCode: int):
        pb = self.table.cellWidget(row, 2)
        if isinstance(pb, QtWidgets.QProgressBar):
            if pb.maximum() == 0:
                pb.setRange(0, 100)
            pb.setValue(100 if exitCode == 0 else 0)
            pb.setFormat("Done" if exitCode == 0 else "Failed")

        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem("✅ Finished" if exitCode == 0 else "❌ Failed"))
        self.overall.setValue(self.overall.value() + 1)

        self._run_index += 1
        self._run_next_job()

    def cancel_all(self):
        # Kill current running job (if any) and mark remaining as cancelled
        if hasattr(self, "_jobs") and 0 <= getattr(self, "_run_index", -1) < len(self._jobs):
            try:
                self._jobs[self._run_index].kill()
            except Exception:
                pass
        self.status.showMessage("Cancelled. Current file will stop.")
        self.act_cancel.setEnabled(False)

    # ------------- Worker signals (legacy path) -------------
    @QtCore.Slot(float, int)
    def on_batch_done(self, elapsed: float, processed: int):
        QtWidgets.QMessageBox.information(self, "Whisper Transcriber", f"Done.\nProcessed {processed} file(s).\nElapsed: {elapsed:.1f}s")
        self.status.showMessage(f"Finished {processed} file(s) in {elapsed:.1f}s.")
        self.act_run.setEnabled(True)
        self.act_cancel.setEnabled(False)
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

# ------------------------------
# App bootstrap
# ------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # Respect OS dark mode, allow High DPI scaling
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app.setStyle("Fusion")  # neutral, modern look

    # Optional: subtle dark palette
    palette = app.palette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(36, 36, 36))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(76, 110, 245))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())
