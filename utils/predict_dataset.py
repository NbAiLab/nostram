import argparse
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from datasets import load_dataset
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/perk/service_account_nancy.json"

def process_audio_data(dataset_path, split, model_path, num_examples):
    # Load the dataset using the datasets library
    dataset = load_dataset(dataset_path, split=split, streaming=True)

    # Initialize the model and processor
    processor = WhisperProcessor.from_pretrained(model_path, from_flax=True)
    model = WhisperForConditionalGeneration.from_pretrained(model_path, from_flax=True)
    
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Header for the Markdown table
    print("| Original Text | Prediction |")
    print("|---------------|------------|")

    # Process each example in the dataset
    for idx, example in enumerate(dataset):
        if idx >= num_examples:
            break

        waveform = np.array(example["audio"]["array"], dtype=np.float32)
        sampling_rate = example["audio"]["sampling_rate"]

        # Pre-process the waveform to get the input features
        input_features = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)

        # Generate the token IDs using the model
        language = example["text_language"]
        predicted_ids = model.generate(input_features, task="transcribe", language=language,  return_timestamps=True,max_new_tokens=256)
        
        # Decode the token IDs to get the transcription
        transcription = processor.batch_decode(predicted_ids, decode_with_timestamps=True,skip_special_tokens=False)[0]
        
        # Print in the Markdown table format
        print(f"| {example['text']} | {transcription} |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio data using a Whisper model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path or identifier to the dataset.")
    parser.add_argument("--split", type=str, required=True, help="Dataset split to use (train, test, validation).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained Whisper model.")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to process.")
    
    args = parser.parse_args()
    process_audio_data(args.dataset_path, args.split, args.model_path, args.num_examples)
