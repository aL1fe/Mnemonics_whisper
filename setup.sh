#!/usr/bin/env bash

export HF_HUB_DOWNLOAD_MAX_WORKERS=1
export HF_HUB_DISABLE_XET=1

conda create -n mnemonics_whisper python=3.12 -y
conda activate mnemonics_whisper

python -m pip install --upgrade pip

if command -v nvidia-smi &> /dev/null; then
    echo -e "\e[32mNVIDIA driver detected.\e[0m"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
else
    echo -e "\e[33mNVIDIA driver not found.\e[0m"
    pip install torch torchvision
fi

pip install -r requirements.txt

mkdir -p models
hf download openai/whisper-tiny --local-dir ./models/whisper-tiny
