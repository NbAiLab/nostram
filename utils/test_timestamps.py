# HF Imports
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Whisper-Jax Imports
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

model_paths=["NbAiLab/nb-whisper-small-RC1","NbAiLab/nb-whisper-medium-RC1"]
audio_path="audio.mp3"

# Loop over the models, and compare the predictions
for model_path in model_paths:
    # HuggingFace
    processor = WhisperProcessor.from_pretrained(model_path, from_flax=True)
    model = WhisperForConditionalGeneration.from_pretrained(model_path, from_flax=True)
    
    with open(audio_path, "rb") as f:
        inputs = f.read()
    inputs = ffmpeg_read(inputs, sampling_rate=16000)

    input_features = processor(inputs, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features, task="transcribe", language="no", return_timestamps=True)
    hf_prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True, decode_with_timestamps=True)

    print(f"HuggingFace prediction for {model_path} : \n {hf_prediction}")


    # Whisper-Jax
    pipeline = FlaxWhisperPipline(model_path, dtype=jnp.bfloat16)
    whisper_jax_prediction = pipeline(audio_path, task="transcribe", language="no", return_timestamps=True)

    print(f"WhisperJax prediction for {model_path}: \n {whisper_jax_prediction}")




