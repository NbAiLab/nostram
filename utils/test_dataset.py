## Old script - use eval_model.py instead


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
import librosa
import re

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

def normalizer(text, extra_clean=False, super_normalize=False):
    before_clean = text
    if extra_clean:
        # Remove specific words and text within star brackets
        TODO here as well!!!
        
        text = re.sub(r'\b(emm|hmm|heh|eee|mmm|qqq)\b', '', text)
        text = re.sub(r'<[^>]*>', '', text)

    # If the text is empty after cleaning, use the original text
    if text == "":
        text = before_clean
    
    if super_normalize:
        ...
      
    # Standard transformations
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveWhiteSpace(replace_by_space=True)
    ])

    return transformation(text)

def calculate_wer(references, predictions,extra_clean=False,super_normalize=False):
    normalized_references = [normalizer(ref, extra_clean, super_normalize) for ref in references]
    normalized_predictions = [normalizer(pred, extra_clean, super_normalize) for pred in predictions]
    return jiwer.wer(normalized_references, normalized_predictions)

def process_audio_data(dataset_path, split, text_field, model_path, name, num_examples, task, language, print_predictions, calculate_wer_flag, device, save_file, from_flax, num_beams, extra_clean=False, super_normalize=False):

    dataset = load_dataset(dataset_path, name=name, split=split, streaming=True)

    
    processor = WhisperProcessor.from_pretrained(model_path, from_flax=from_flax)
    model = WhisperForConditionalGeneration.from_pretrained(model_path, from_flax=from_flax)
    
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
            sampling_rate = 16000  # Update the sampling rate
        
        input_features = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)

        predicted_ids = model.generate(input_features, task=task, language=language, return_timestamps=True, max_new_tokens=256, num_beams=num_beams)
        transcription = processor.batch_decode(predicted_ids, decode_with_timestamps=False, skip_special_tokens=True)[0]

        if print_predictions:
            print(f"| {example[text_field]} | {transcription} |")

        if calculate_wer_flag:
            references.append(example[text_field])
            predictions.append(transcription)

    if calculate_wer_flag:
        overall_wer = calculate_wer(references, predictions, extra_clean, super_normalize)
        print(f"Average WER for {processed_examples} examples: {overall_wer*100:.1f}")

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
                "num_beams": num_beams,
                "wer": overall_wer
            }
            with open(save_file, 'a') as f:
                f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio data using a Whisper model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path or identifier to the dataset.")
    parser.add_argument("--name", type=str, required=False, default="",help="Name of the dataset subset.")
    parser.add_argument("--split", type=str, required=True, help="Dataset split to use (train, test, validation).")
    parser.add_argument("--text_field", type=str, default="text", help="Field where the text is stored in the dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained Whisper model.")
    parser.add_argument("--num_examples", type=int, default=999999999, help="Number of examples to process.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams.")
    parser.add_argument("--task", type=str, default="transcribe", help="Transcribe, translate or both.")
    parser.add_argument("--language", type=str, default="no", help="Specify language (ie no, nn or en) if you want to override the setting in the dataset.")
    parser.add_argument("--print_predictions", action="store_true", help="Print predictions if set.")
    parser.add_argument("--calculate_wer", action="store_true", help="Calculate WER if set.")
    parser.add_argument("--from_flax", action="store_true", help="Use flax weights.")
    parser.add_argument("--extra_clean", action="store_true", help="Cleans the text for hesitations and star brackets")
    parser.add_argument("--super_normalize", action="store_true", help="Uses the normalisation from the Wav2Vec article")
    parser.add_argument("--device", type=int, required=False, default=0, help="For GPU only. The device to load the model to")
    parser.add_argument("--save_file", type=str, help="Path to save results in JSON Lines format.")
    
    args = parser.parse_args()
    process_audio_data(args.dataset_path, args.split, args.text_field, args.model_path, args.name,args.num_examples, args.task, args.language, args.print_predictions, args.calculate_wer, args.device, args.save_file, args.from_flax, args.num_beams,args.extra_clean,args.super_normalize)
