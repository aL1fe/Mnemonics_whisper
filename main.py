from fastapi import FastAPI, File, UploadFile
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
import torch.version
import string
from pathlib import Path

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
model_id = Path("models") / "whisper-tiny"

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
    dtype=torch_dtype,
    device=device,
    return_timestamps=False,
    chunk_length_s=30,
    generate_kwargs={
        "max_new_tokens": 128,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,  # default "repetition_penalty=0.0 no penalty
        "temperature": 0.0,
    }
)

app = FastAPI()


async def transcribe_file(file_path):
    with torch.inference_mode():
        start = time.perf_counter()
        result = pipe(file_path)  #, max_new_tokens=5)
        duration = time.perf_counter() - start
    return result, duration


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    FOLDER = "incoming_files"
    try:
        file_path = await file_utils.save_file(file, FOLDER)
        if file_path is None: return {"Error": "Error processing file"}

        result, duration = await transcribe_file(file_path)
        await file_utils.delete_file(file, FOLDER)

        return (result["text"]).strip().rstrip(string.punctuation).lower() # type: ignore
    except Exception as e:
        return {"Error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
