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

# Just needed if the dataset requires authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/perk/service_account_nancy.json"

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate between reference and hypothesis."""
    return wer(reference, hypothesis)

def process_audio_data(dataset_path, split, model_path, num_examples, task, language, print_predictions, calculate_wer_flag):
    dataset = load_dataset(dataset_path, split=split, streaming=True)
    processor = WhisperProcessor.from_pretrained(model_path, from_flax=True)
    model = WhisperForConditionalGeneration.from_pretrained(model_path, from_flax=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_wer = 0

    for idx, example in enumerate(dataset):
        if idx >= num_examples:
            break

        waveform = np.array(example["audio"]["array"], dtype=np.float32)
        sampling_rate = example["audio"]["sampling_rate"]
        input_features = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)

        # [Existing prediction generation code...]

        if print_predictions:
            print(f"| {example['text']} | {transcription} |")

        if calculate_wer_flag:
            total_wer += calculate_wer(example['text'], transcription)

    if calculate_wer_flag:
        average_wer = total_wer / num_examples
        print(f"Average WER: {average_wer:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio data using a Whisper model.")
    
    # [Existing argument parser code...]
    parser.add_argument("--print_predictions", action="store_true", help="Print predictions if set.")
    parser.add_argument("--calculate_wer", action="store_true", help="Calculate WER if set.")

    args = parser.parse_args()
    process_audio_data(args.dataset_path, args.split, args.model_path, args.num_examples, args.task, args.language, args.print_predictions, args.calculate_wer)