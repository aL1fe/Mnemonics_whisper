python -m venv .venv
.\.venv\Scripts\activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r .\requirements.txt
