import argparse
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from datasets import load_dataset
import os
import warnings
import logging

# Suppress specific warning categories
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set TensorFlow logging to error level only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set other logging levels
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('datasets').setLevel(logging.ERROR)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/perk/service_account_nancy.json"

def process_audio_data(dataset_path, split, model_path, num_examples, task, language):
    # Load the dataset using the datasets library
    dataset = load_dataset(dataset_path, split=split, streaming=True)

    # Initialize the model and processor
    processor = WhisperProcessor.from_pretrained(model_path, from_flax=True)
    model = WhisperForConditionalGeneration.from_pretrained(model_path, from_flax=True)
    
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Header for the Markdown table
    if task != "both":
        print("| Original Text | Prediction |")
    else:
        print("| Original Text | Transcribe | Translate | Equal |")

    # Process each example in the dataset
    for idx, example in enumerate(dataset):
        if idx >= num_examples:
            break

        waveform = np.array(example["audio"]["array"], dtype=np.float32)
        sampling_rate = example["audio"]["sampling_rate"]

        # Pre-process the waveform to get the input features
        input_features = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)

        if task != "both":
            # Generate the token IDs using the model for either transcription or translation
            predicted_ids = model.generate(input_features, task=task, language=language, return_timestamps=True, max_new_tokens=256)
            transcription = processor.batch_decode(predicted_ids, decode_with_timestamps=False, skip_special_tokens=True)[0]
            print(f"| {example['text']} | {transcription} |")
        else:
            # Perform both transcription and translation
            transcribe_ids = model.generate(input_features, task="transcribe", language=language, return_timestamps=True, max_new_tokens=256)
            transcribe_text = processor.batch_decode(transcribe_ids, decode_with_timestamps=False, skip_special_tokens=True)[0]

            translate_ids = model.generate(input_features, task="translate", language=language, return_timestamps=True, max_new_tokens=256)
            translate_text = processor.batch_decode(translate_ids, decode_with_timestamps=False, skip_special_tokens=True)[0]

            equal = transcribe_text == translate_text
            print(f"| {example['text']} | {transcribe_text} | {translate_text} | {equal} |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio data using a Whisper model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path or identifier to the dataset.")
    parser.add_argument("--split", type=str, required=True, help="Dataset split to use (train, test, validation).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained Whisper model.")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to process.")
    parser.add_argument("--task", type=str, default="transcribe", help="Transcribe, translate or both.")
    parser.add_argument("--language", type=str, default="auto", help="Specify language (ie no, nn or en) if you want to override the setting in the dataset.")

    args = parser.parse_args()
    process_audio_data(args.dataset_path, args.split, args.model_path, args.num_examples, args.task, args.language)
