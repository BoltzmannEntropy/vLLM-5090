# vLLM Installation Script for Windows with RTX 5090
# Based on: https://discuss.vllm.ai/t/vllm-on-rtx5090-working-gpu-setup-with-torch-2-9-0-cu128/1492
# Adapted for Windows environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "vLLM Installation for RTX 5090 (Windows)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

# Check CUDA
try {
    $cudaVersion = nvcc --version 2>&1 | Select-String "release" | ForEach-Object { $_.ToString() }
    Write-Host "CUDA detected: $cudaVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: CUDA not found. Please install CUDA 12.8 from:" -ForegroundColor Red
    Write-Host "https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    exit 1
}

# Check NVIDIA driver
try {
    $nvidiaDriver = nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1
    Write-Host "NVIDIA Driver: $nvidiaDriver" -ForegroundColor Green
} catch {
    Write-Host "ERROR: NVIDIA driver not found. Please install latest driver (575+)" -ForegroundColor Red
    exit 1
}

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Please install Python 3.12+" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 1: Creating Virtual Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Create venv directory
$venvPath = "vllm-env"
if (Test-Path $venvPath) {
    Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvPath
}

Write-Host "Creating new virtual environment..." -ForegroundColor Yellow
python -m venv $venvPath

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "$venvPath\Scripts\Activate.ps1"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 2: Installing PyTorch Nightly (CUDA 12.8)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "Installing PyTorch 2.9.0 nightly with CUDA 12.8..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install setuptools-scm
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify PyTorch installation
Write-Host ""
Write-Host "Verifying PyTorch installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 3: Cloning vLLM Repository" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$vllmPath = "vllm"
if (Test-Path $vllmPath) {
    Write-Host "vLLM directory already exists. Updating..." -ForegroundColor Yellow
    Set-Location $vllmPath
    git pull
} else {
    Write-Host "Cloning vLLM repository..." -ForegroundColor Yellow
    git clone https://github.com/vllm-project/vllm.git
    Set-Location $vllmPath
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 4: Preparing vLLM Build" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "Running use_existing_torch.py..." -ForegroundColor Yellow
python use_existing_torch.py

Write-Host "Installing build requirements..." -ForegroundColor Yellow
pip install -r requirements-build.txt

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 5: Setting Environment Variables" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "Setting critical build environment variables..." -ForegroundColor Yellow

# Set environment variables for build
$env:VLLM_FLASH_ATTN_VERSION = "2"
$env:TORCH_CUDA_ARCH_LIST = "12.0"
$env:MAX_JOBS = "6"  # Adjust based on available RAM

Write-Host "VLLM_FLASH_ATTN_VERSION = $env:VLLM_FLASH_ATTN_VERSION" -ForegroundColor Green
Write-Host "TORCH_CUDA_ARCH_LIST = $env:TORCH_CUDA_ARCH_LIST" -ForegroundColor Green
Write-Host "MAX_JOBS = $env:MAX_JOBS" -ForegroundColor Green

# Additional Windows-specific environment variables
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
Write-Host "CUDA_HOME = $env:CUDA_HOME" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 6: Building and Installing vLLM" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "This will take 30-60 minutes depending on your system..." -ForegroundColor Yellow
Write-Host ""

# Build vLLM from source
pip install --no-build-isolation -e .

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 7: Verifying Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "Checking vLLM version..." -ForegroundColor Yellow
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "vLLM Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment in the future, run:" -ForegroundColor Cyan
Write-Host "  .\vllm-env\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "To test vLLM, run:" -ForegroundColor Cyan
Write-Host "  python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m" -ForegroundColor Yellow
Write-Host ""
