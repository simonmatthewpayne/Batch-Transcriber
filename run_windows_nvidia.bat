@echo off
setlocal
cd /d "%~dp0"

echo.
echo ==========================================
echo  Whisper Batch Transcriber - NVIDIA Setup
echo ==========================================
echo.

REM --------------------------------------------------
REM 1. Create local virtual environment if missing
REM --------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Creating local Python virtual environment...

    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Python 3.11 not found. Trying Python 3.12...
        py -3.12 -m venv .venv
    )
    if errorlevel 1 (
        echo Trying default Python...
        python -m venv .venv
    )
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create virtual environment.
        echo Please install Python 3.11 or 3.12 and tick "Add Python to PATH".
        pause
        exit /b 1
    )
)

echo.
echo Activating virtual environment...
call ".venv\Scripts\activate.bat"

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM --------------------------------------------------
REM 2. Install GUI / utility dependencies first
REM    Do NOT install openai-whisper yet, because it may
REM    pull CPU-only torch before CUDA torch is installed.
REM --------------------------------------------------
echo.
echo Installing GUI and utility dependencies...
python -m pip install PySide6 psutil tiktoken imageio-ffmpeg

REM --------------------------------------------------
REM 3. Install CUDA-enabled PyTorch for NVIDIA GPUs
REM --------------------------------------------------
echo.
echo Installing CUDA-enabled PyTorch for NVIDIA GPUs...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM --------------------------------------------------
REM 4. Install Whisper after CUDA torch is already present
REM --------------------------------------------------
echo.
echo Installing OpenAI Whisper...
python -m pip install openai-whisper

REM --------------------------------------------------
REM 5. Prepare local ffmpeg
REM --------------------------------------------------
echo.
echo Preparing local ffmpeg...
if not exist "ffmpeg" mkdir ffmpeg

python -c "import imageio_ffmpeg, shutil, os; src=imageio_ffmpeg.get_ffmpeg_exe(); dst=os.path.join('ffmpeg','ffmpeg.exe'); shutil.copyfile(src,dst); print('ffmpeg copied to', dst)"

set "PATH=%CD%\ffmpeg;%PATH%"

REM --------------------------------------------------
REM 6. Check CUDA status
REM --------------------------------------------------
echo.
echo Checking installation...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None detected')"

echo.
echo If CUDA available says False, the app can still run on CPU, but NVIDIA acceleration is not active.
echo.

REM --------------------------------------------------
REM 7. Start app
REM --------------------------------------------------
echo Starting Whisper Batch Transcriber...
python TranscriberApp2_nvidia.py

echo.
pause