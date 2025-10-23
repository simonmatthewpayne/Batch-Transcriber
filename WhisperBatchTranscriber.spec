# ─────────────────────────────────────────────────────────────────────────────
# WhisperBatchTranscriber.spec  (CLEANED + FIXED)
#
# Build from your venv with:
#   python -m PyInstaller WhisperBatchTranscriber.spec
#
# Folder layout expected:
#   Transcriber/
#     TranscriberApp.py
#     models/base.pt, models/small.pt, models/medium.pt
#     ffmpeg/ffmpeg.exe
# ─────────────────────────────────────────────────────────────────────────────

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
)

# 1) Data files (non-Python) we want inside the EXE
datas = []
datas += collect_data_files('whisper')   # tokenizer/config data for whisper
datas += collect_data_files('tiktoken')  # tokenizer data

# Your bundled models (so base/small/medium work offline)
datas += [
    ('models/base.pt',   'models'),
    ('models/small.pt',  'models'),
    ('models/medium.pt', 'models'),
]

# ffmpeg bundled so users don’t need a system install
datas += [('ffmpeg/ffmpeg.exe', 'ffmpeg')]

# 2) Native binaries (DLLs/.pyd) — torch CUDA wheel ships lots of these
binaries = collect_dynamic_libs('torch')

# 3) Hidden imports — include dynamic imports; drop torchaudio to avoid errors
hiddenimports = (
    collect_submodules('whisper')
    + collect_submodules('torch')
    + collect_submodules('tiktoken')
)

# 4) Build graph analysis
block_cipher = None

a = Analysis(
    ['TranscriberApp.py'],   # <— your actual entry script
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],             # leave empty unless you know what to exclude
    noarchive=False,
)

# 5) Pack pure-Python into a zipped archive
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# 6) Final single-file EXE (GUI → no console)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='WhisperBatchTranscriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                   # GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
