$env:HF_HUB_DOWNLOAD_MAX_WORKERS=1

python -m venv .venv
.\.venv\Scripts\activate.ps1
python -m pip install --upgrade pip

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "NVIDIA driver detected." -ForegroundColor Green
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
}
else {
    Write-Host "NVIDIA driver not found." -ForegroundColor Yellow
	pip install torch torchvision
}

if (Test-Path requirements.txt) {
    python -m pip install -r requirements.txt
} else {
    Write-Host "File requirements.txt not found." -ForegroundColor Red
}

mkdir models -Force | Out-Null
hf download openai/whisper-tiny --local-dir ./models/whisper-tiny
