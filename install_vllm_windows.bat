@echo off
REM vLLM Installation Script for Windows with RTX 5090
REM Based on: https://discuss.vllm.ai/t/vllm-on-rtx5090-working-gpu-setup-with-torch-2-9-0-cu128/1492
REM Adapted for Windows environment

echo ========================================
echo vLLM Installation for RTX 5090 (Windows)
echo ========================================
echo.

REM Check prerequisites
echo Checking prerequisites...
echo.

REM Check CUDA
nvcc --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: CUDA not found. Please install CUDA 12.8
    echo Download from: https://developer.nvidia.com/cuda-downloads
    exit /b 1
)
echo CUDA detected
nvcc --version | findstr "release"

REM Check NVIDIA driver
nvidia-smi --query-gpu=driver_version --format=csv,noheader > nul 2>&1
if errorlevel 1 (
    echo ERROR: NVIDIA driver not found. Please install latest driver (575+)
    exit /b 1
)
echo NVIDIA Driver:
nvidia-smi --query-gpu=driver_version --format=csv,noheader

REM Check Python
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.12+
    exit /b 1
)
echo Python detected:
python --version

echo.
echo ========================================
echo Step 1: Creating Virtual Environment
echo ========================================
echo.

REM Create venv directory
set VENV_PATH=vllm-env
if exist %VENV_PATH% (
    echo Removing existing virtual environment...
    rmdir /s /q %VENV_PATH%
)

echo Creating new virtual environment...
python -m venv %VENV_PATH%

REM Activate virtual environment
echo Activating virtual environment...
call %VENV_PATH%\Scripts\activate.bat

echo.
echo ========================================
echo Step 2: Installing PyTorch Nightly (CUDA 12.8)
echo ========================================
echo.

echo Installing PyTorch 2.9.0 nightly with CUDA 12.8...
python -m pip install --upgrade pip
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

REM Verify PyTorch installation
echo.
echo Verifying PyTorch installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo.
echo ========================================
echo Step 3: Cloning vLLM Repository
echo ========================================
echo.

set VLLM_PATH=vllm
if exist %VLLM_PATH% (
    echo vLLM directory already exists. Updating...
    cd %VLLM_PATH%
    git pull
) else (
    echo Cloning vLLM repository...
    git clone https://github.com/vllm-project/vllm.git
    cd %VLLM_PATH%
)

echo.
echo ========================================
echo Step 4: Preparing vLLM Build
echo ========================================
echo.

echo Running use_existing_torch.py...
python use_existing_torch.py

echo Installing build requirements...
pip install -r requirements-build.txt

echo.
echo ========================================
echo Step 5: Setting Environment Variables
echo ========================================
echo.

echo Setting critical build environment variables...

REM Set environment variables for build
set VLLM_FLASH_ATTN_VERSION=2
set TORCH_CUDA_ARCH_LIST=12.0
set MAX_JOBS=6

echo VLLM_FLASH_ATTN_VERSION = %VLLM_FLASH_ATTN_VERSION%
echo TORCH_CUDA_ARCH_LIST = %TORCH_CUDA_ARCH_LIST%
echo MAX_JOBS = %MAX_JOBS%

REM Set CUDA_HOME (adjust path if needed)
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
echo CUDA_HOME = %CUDA_HOME%

echo.
echo ========================================
echo Step 6: Building and Installing vLLM
echo ========================================
echo This will take 30-60 minutes depending on your system...
echo.

REM Build vLLM from source
pip install --no-build-isolation -e .

echo.
echo ========================================
echo Step 7: Verifying Installation
echo ========================================
echo.

echo Checking vLLM version...
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

echo.
echo ========================================
echo vLLM Installation Complete!
echo ========================================
echo.
echo To activate the environment in the future, run:
echo   %VENV_PATH%\Scripts\activate.bat
echo.
echo To test vLLM, run:
echo   python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m
echo.

cd ..
