from fastapi import FastAPI, UploadFile
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import asyncio
import time
import torch.version
import string

import file_utils


if torch.cuda.is_available():
    device = "cuda:0"
    num_devices = torch.cuda.device_count()
    print(f"CUDA is available. Total devices: {num_devices}")
    print(f"Torch CUDA version {torch.version.cuda}")
    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        print(f"Device {i}: {device_name}")
        print(f" - CUDA Capability: {torch.cuda.get_device_capability(i)}")
else:
    device = "cpu"
    print("CUDA is not available. Using CPU.")

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "openai/whisper-large-v3"
# model_id = "openai/whisper-tiny"
model_id = ".\\models\\models--openai--whisper-tiny\\snapshots\\169d4a4341b33bc18d8881c4b69c2e104e1cc0af"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
model.to(device)
model.generation_config.language = "en"
model.generation_config.task = "transcribe"

processor = AutoProcessor.from_pretrained(model_id)


pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    # batch_size=1,
    dtype=torch_dtype,
    device=device,
    return_timestamps=False,
    chunk_length_s=30
)

app = FastAPI()


async def transcribe_file(file_path):
    with torch.inference_mode():
        start = time.perf_counter()
        result = pipe(file_path)
        duration = time.perf_counter() - start
    return result, duration


@app.post("/upload/")
async def upload_file(file: UploadFile):
    FOLDER = "incoming_files"
    try:
        file_path = await file_utils.save_file(file, FOLDER)
        if file_path is None: return {"Error": "Error processing file"}

        result, duration = await transcribe_file(file_path)
        await file_utils.delete_file(file, FOLDER)

        return result["text"]  # type: ignore
        return (result["text"]).strip().rstrip(string.punctuation).lower() # type: ignore
        return {"time": f"{duration:.3f}s", "Text": result["text"]}  # type: ignore
    except Exception as e:
        return {"Error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
