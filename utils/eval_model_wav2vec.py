import argparse
import numpy as np
import torch
from datasets import load_dataset
import os
import warnings
import jiwer
import json
from datetime import datetime
import librosa
import re
from transformers import Wav2Vec2Processor, pipeline

# Suppress specific warning categories
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Just needed if the dataset requires authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/perk/service_account_nancy.json"

def normalizer(text, extra_clean=False):
    before_clean = text
    if extra_clean:
        text = re.sub(r'\b(emm|hmm|heh|eee|mmm|qqq)\b', '', text)
        text = re.sub(r'<[^>]*>', '', text)
    if text == "":
        text = before_clean
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveWhiteSpace(replace_by_space=True)
    ])
    return transformation(text)

def calculate_wer(references, predictions, extra_clean=False):
    normalized_references = [normalizer(ref, extra_clean) for ref in references]
    normalized_predictions = [normalizer(pred, extra_clean) for pred in predictions]
    return jiwer.wer(normalized_references, normalized_predictions)

def transcribe_with_model(asr_pipeline, waveform, processor):
    input_values = processor(waveform, return_tensors="pt", padding=True).input_values
    transcription = asr_pipeline(input_values)[0]['text']
    return transcription

def process_audio_data(dataset_path, split, text_field, model_path, name, num_examples, print_predictions, calculate_wer_flag, device, save_file, extra_clean):
    dataset = load_dataset(dataset_path, name=name, split=split, streaming=True)
    
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    asr_pipeline = pipeline("automatic-speech-recognition", model=model_path, device=device)

    references = []
    predictions = []
    processed_examples = 0
    
    for idx, example in enumerate(dataset):
        if idx >= num_examples:
            break
        processed_examples += 1
        waveform = np.array(example["audio"]["array"], dtype=np.float32)
        sampling_rate = example["audio"]["sampling_rate"]
        
        if sampling_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=16000)

        transcription = transcribe_with_model(asr_pipeline, waveform, processor)

        if print_predictions:
            print(f"| {example[text_field]} | {transcription} |")

        if calculate_wer_flag:
            references.append(example[text_field])
            predictions.append(transcription)

    if calculate_wer_flag:
        overall_wer = calculate_wer(references, predictions, extra_clean)
        print(f"Average WER for {processed_examples} examples: {overall_wer * 100:.1f}%")

        if save_file:
            result = {
                "dataset_path": dataset_path,
                "extra_clean": extra_clean,
                "model_path": model_path,
                "name": name,
                "split": split,
                "language": language,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "example_count": processed_examples,
                "wer": overall_wer * 100
            }
            with open(save_file, 'a') as f:
                f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio data using a Wav2Vec2 model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path or identifier to the dataset.")
    parser.add_argument("--name", type=str, required=False, default="", help="Name of the dataset subset.")
    parser.add_argument("--split", type=str, required=True, help="Dataset split to use (train, test, validation).")
    parser.add_argument("--text_field", type=str, default="text", help="Field where the text is stored in the dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained Wav2Vec2 model.")
    parser.add_argument("--num_examples", type=int, default=999999999, help="Number of examples to process.")
    parser.add_argument("--print_predictions", action="store_true", help="Print predictions if set.")
    parser.add_argument("--calculate_wer", action="store_true", help="Calculate WER if set