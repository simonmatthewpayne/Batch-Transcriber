@echo off
setlocal
cd /d "%~dp0"

echo.
echo ==========================================
echo  Whisper Batch Transcriber - NVIDIA Setup
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating local virtual environment...
    py -3.11 -m venv .venv

    if errorlevel 1 (
        echo Python 3.11 not found. Trying default Python...
        python -m venv .venv
    )
)

echo.
echo Activating virtual environment...
call ".venv\Scripts\activate.bat"

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing app dependencies...
pip install PySide6 psutil openai-whisper tiktoken

echo.
echo Installing CUDA-enabled PyTorch for NVIDIA GPUs...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Checking CUDA...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None detected')"

echo.
echo Starting Whisper Batch Transcriber...
python TranscriberApp2_nvidia.py

echo.
pause