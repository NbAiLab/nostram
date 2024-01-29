import argparse
import numpy as np
from datasets import load_dataset
import os
import warnings
import jiwer
import json
from datetime import datetime
import librosa
import re
from transformers import pipeline

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

def process_audio_data(dataset_path, split, text_field, model_path, name, num_examples, print_predictions, calculate_wer_flag, device, save_file, extra_clean):
    dataset = load_dataset(dataset_path, name=name, split=split, streaming=True)
    
    asr_pipeline = pipeline("automatic-speech-recognition", model=model_path, device=device)

    references = []
    predictions = []
    processed_examples = 0
    
    for idx, example in enumerate(dataset):
        if idx >= num_examples:
            break
        processed_examples += 1
        waveform = np.array(example["audio"]["array"], dtype=np.float32)
        #sampling_rate = example["audio"]["sampling_rate"]

        transcription = asr_pipeline(waveform)['text']

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
    parser.add_argument("--calculate_wer", action="store_true", help="Calculate WER if set.")
    parser.add_argument("--device", type=int, required=False, default=0, help="For GPU only. The device to load the model to.")
    parser.add_argument("--save_file", type=str, help="Path to save results in JSON Lines format.")
    parser.add_argument("--extra_clean", action="store_true", help="Apply extra cleaning to the text for hesitations and star brackets.")

    args = parser.parse_args()
    process_audio_data(args.dataset_path, args.split, args.text_field, args.model_path, args.name, args.num_examples, args.print_predictions, args.calculate_wer, args.device, args.save_file, args.extra_clean)
