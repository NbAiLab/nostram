import argparse
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from datasets import load_dataset
import os
import warnings
import jiwer
import json
from datetime import datetime
import logging

# Suppress specific warning categories
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set TensorFlow logging to error level only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set other logging levels
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('datasets').setLevel(logging.ERROR)

# Just needed if the dataset requires authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/perk/service_account_nancy.json"

def normalizer(text):
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveWhiteSpace(replace_by_space=True)
    ])
    return transformation(text)

def calculate_wer(references, predictions):
    normalized_references = [normalizer(ref) for ref in references]
    normalized_predictions = [normalizer(pred) for pred in predictions]
    return jiwer.wer(normalized_references, normalized_predictions)

def process_audio_data(dataset_path, split, model_path, subset, num_examples, task, language, print_predictions, calculate_wer_flag, save_file):
    if subset !="":
        dataset = load_dataset(dataset_path, subset, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_path, split=split, streaming=True)

    
    processor = WhisperProcessor.from_pretrained(model_path, from_flax=False)
    model = WhisperForConditionalGeneration.from_pretrained(model_path, from_flax=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    references = []
    predictions = []
    
    for idx, example in enumerate(dataset):
        if idx >= num_examples:
            break

        waveform = np.array(example["audio"]["array"], dtype=np.float32)
        sampling_rate = example["audio"]["sampling_rate"]
        input_features = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)

        predicted_ids = model.generate(input_features, task=task, language=language, return_timestamps=True, max_new_tokens=256)
        transcription = processor.batch_decode(predicted_ids, decode_with_timestamps=False, skip_special_tokens=True)[0]

        if print_predictions:
            print(f"| {example['text']} | {transcription} |")

        if calculate_wer_flag:
            references.append(example['text'])
            predictions.append(transcription)

    if calculate_wer_flag:
        overall_wer = calculate_wer(references, predictions)
        print(f"Average WER: {overall_wer:.2f}")

        if save_file:
            result = {
                "model_path": model_path,
                "dataset_path": dataset_path,
                "split": split,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "example_count": num_examples,
                "wer": overall_wer
            }
            with open(save_file, 'a') as f:
                f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio data using a Whisper model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path or identifier to the dataset.")
    parser.add_argument("--subset", type=str, required=False, default="",help="Path to the dataset subset.")
    parser.add_argument("--split", type=str, required=True, help="Dataset split to use (train, test, validation).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained Whisper model.")
    parser.add_argument("--num_examples", type=int, default=999999999, help="Number of examples to process.")
    parser.add_argument("--task", type=str, default="transcribe", help="Transcribe, translate or both.")
    parser.add_argument("--language", type=str, default="no", help="Specify language (ie no, nn or en) if you want to override the setting in the dataset.")
    parser.add_argument("--print_predictions", action="store_true", help="Print predictions if set.")
    parser.add_argument("--calculate_wer", action="store_true", help="Calculate WER if set.")
    parser.add_argument("--save_file", type=str, help="Path to save results in JSON Lines format.")
    
    args = parser.parse_args()
    process_audio_data(args.dataset_path, args.split, args.model_path, args.data_dir,args.num_examples, args.task, args.language, args.print_predictions, args.calculate_wer, args.save_file)
