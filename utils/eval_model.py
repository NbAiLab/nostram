import argparse
import numpy as np
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
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline


# Suppress specific warning categories
#warnings.filterwarnings('ignore', category=UserWarning)
#warnings.filterwarnings('ignore', category=FutureWarning)

# Set TensorFlow logging to error level only
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set other logging levels
#logging.getLogger('transformers').setLevel(logging.ERROR)
#logging.getLogger('datasets').setLevel(logging.ERROR)

# Just needed if the dataset requires authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/perk/service_account_nancy.json"

def normalizer(text, extra_clean=False):
    before_clean = text
    if extra_clean:
        # Remove specific words and text within star brackets
        text = re.sub(r'\b(emm|hmm|heh|eee|mmm|qqq)\b', '', text)
        text = re.sub(r'<[^>]*>', '', text)

    # If the text is empty after cleaning, use the original text
    if text == "":
        text = before_clean

    # Standard transformations
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

def load_model_and_processor(model_path, model_type, from_flax):
    if model_type == 'whisper':
        processor = WhisperProcessor.from_pretrained(model_path, from_flax=from_flax)
        model = WhisperForConditionalGeneration.from_pretrained(model_path, from_flax=from_flax)
    elif model_type == 'wav2vec2':
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
    else:
        raise ValueError("Unsupported model type")
    return processor, model

def transcribe_with_model(processor, model, waveform, sampling_rate, model_type, device):
    if model_type == 'whisper':
        input_features = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)
        predicted_ids = model.generate(input_features, task='transcribe', language='no', return_timestamps=True, max_new_tokens=256)
        transcription = processor.batch_decode(predicted_ids, decode_with_timestamps=False, skip_special_tokens=True)[0]
    elif model_type == 'wav2vec2':
        # Use pipeline for Wav2Vec2
        pipe = pipeline(model=model, tokenizer=processor)
        transcription = pipe(waveform, sampling_rate=sampling_rate)[0]['text']

    return transcription

def process_audio_data(dataset_path, split, text_field, model_path, name, num_examples, task, language, print_predictions, calculate_wer_flag, device, save_file, from_flax, extra_clean, model_type):

    dataset = load_dataset(dataset_path, name=name, split=split, streaming=True)

    processor, model = load_model_and_processor(model_path, model_type, from_flax)
    
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
            sampling_rate = 16000
        
        transcription = transcribe_with_model(processor, model, waveform, sampling_rate, model_type, device)

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
    parser = argparse.ArgumentParser(description="Process audio data using a Whisper or Wav2Vec model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path or identifier to the dataset.")
    parser.add_argument("--name", type=str, required=False, default="", help="Name of the dataset subset.")
    parser.add_argument("--split", type=str, required=True, help="Dataset split to use (train, test, validation).")
    parser.add_argument("--text_field", type=str, default="text", help="Field where the text is stored in the dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--num_examples", type=int, default=999999999, help="Number of examples to process.")
    parser.add_argument("--task", type=str, default="transcribe", help="Transcribe, translate or both.")
    parser.add_argument("--language", type=str, default="no", help="Specify language (ie no, nn or en) if you want to override the setting in the dataset.")
    parser.add_argument("--print_predictions", action="store_true", help="Print predictions if set.")
    parser.add_argument("--calculate_wer", action="store_true", help="Calculate WER if set.")
    parser.add_argument("--from_flax", action="store_true", help="Use flax weights.")
    parser.add_argument("--extra_clean", action="store_true", help="Cleans the text for hesitations and star brackets")
    parser.add_argument("--device", type=int, required=False, default=0, help="For GPU only. The device to load the model to")
    parser.add_argument("--save_file", type=str, help="Path to save results in JSON Lines format.")
    parser.add_argument("--model_type", type=str, choices=['whisper', 'wav2vec2'], default='whisper', help="Type of the model to use ('whisper' or 'wav2vec2').")
    
    args = parser.parse_args()
    process_audio_data(args.dataset_path, args.split, args.text_field, args.model_path, args.name, args.num_examples, args.task, args.language, args.print_predictions, args.calculate_wer, args.device, args.save_file, args.from_flax, args.extra_clean, args.model_type)
