import argparse, json, logging, os, re, warnings
import torch, librosa, jiwer
import numpy as np
from datetime import datetime
from datasets import load_dataset
from transformers import pipeline

# Set up logging to report only errors and suppress specific warnings for cleaner output.
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set environment variable for Google Cloud authentication if needed for dataset access.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/perk/service_account_nancy.json"

def normalizer(text, extra_clean=False, super_normalize=False):
    """
    Normalize the given text by applying various transformations.
    
    Args:
    text (str): The text to be normalized.
    extra_clean (bool): If True, removes specific words and text within star brackets.
    super_normalize (bool): If True, applies additional normalization (implementation pending).

    Returns:
    str: The normalized text.
    """
    before_clean = text
    if extra_clean:
        text = re.sub(r'\b(emm|hmm|heh|eee|mmm|qqq)\b', '', text)
        text = re.sub(r'<[^>]*>', '', text)

    if text == "":
        text = before_clean
    
    if super_normalize:
        # Implementation pending
        pass
      
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveWhiteSpace(replace_by_space=True)
    ])

    return transformation(text)

def calculate_wer(references, predictions, extra_clean=False, super_normalize=False):
    """
    Calculate the Word Error Rate (WER) between reference and predicted texts.
    
    Args:
    references (list): The list of reference texts.
    predictions (list): The list of predicted texts.
    extra_clean (bool): Apply extra cleaning to texts.
    super_normalize (bool): Apply super normalization to texts.

    Returns:
    float: The calculated WER.
    """
    normalized_references = [normalizer(ref, extra_clean, super_normalize) for ref in references]
    normalized_predictions = [normalizer(pred, extra_clean, super_normalize) for pred in predictions]
    return jiwer.wer(normalized_references, normalized_predictions)

def process_audio_data(dataset_path, split, text_field, model_path, name, num_examples, task, language, 
                       print_predictions, calculate_wer_flag, device, save_file, num_beams=1,
                       extra_clean=False, super_normalize=False, model_type="whisper"):
    """
    Process audio data from a dataset using a Whisper model pipeline and calculate WER if requested.
    
    Args:
    see the argparse arguments below
    """
    dataset = load_dataset(dataset_path, name=name, split=split, streaming=True)
    
    device = 0 if torch.cuda.is_available() else -1
    
    if model_type == 'whisper':
        model_pipeline = pipeline("automatic-speech-recognition", model=model_path, device=device)
        generate_kwargs = {'task': task, 'language': language, device=device}
    elif model_type == 'wav2vec':
        # Note: Adjust the pipeline task if necessary for wav2vec
        model_pipeline = pipeline("automatic-speech-recognition", model=model_path, device=device)
        generate_kwargs = {}
        
    if num_beams > 1:
            generate_kwargs['num_beams'] = num_beams

    references, predictions = [], []
    processed_examples = 0
    
    for idx, example in enumerate(dataset):
        if idx >= num_examples:
            break
        processed_examples += 1
        waveform = np.array(example["audio"]["array"], dtype=np.float32)
        sampling_rate = example["audio"]["sampling_rate"]
        
        if sampling_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=16000)

        transcription = model_pipeline(waveform, **generate_kwargs)["text"]


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
                "wer": overall_wer,
                "num_beams": num_beams
            }
            with open(save_file, 'a') as f:
                f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio data using a Whisper model pipeline.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path or identifier to the dataset.")
    parser.add_argument("--name", type=str, required=False, default="", help="Name of the dataset subset.")
    parser.add_argument("--split", type=str, required=True, help="Dataset split to use (train, test, validation).")
    parser.add_argument("--text_field", type=str, default="text", help="Field where the text is stored in the dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained Whisper model.")
    parser.add_argument("--num_examples", type=int, default=999999999, help="Number of examples to process.")
    parser.add_argument("--task", type=str, default="transcribe", help="Transcribe, translate or both.")
    parser.add_argument("--language", type=str, default="no", help="Specify language (ie no, nn or en) if you want to override the setting in the dataset.")
    parser.add_argument("--print_predictions", action="store_true", help="Print predictions if set.")
    parser.add_argument("--calculate_wer", action="store_true", help="Calculate WER if set.")
    parser.add_argument("--extra_clean", action="store_true", help="Cleans the text for hesitations and star brackets")
    parser.add_argument("--super_normalize", action="store_true", help="Uses the normalisation from the Wav2Vec article")
    parser.add_argument("--device", type=int, required=False, default=0, help="For GPU only. The device to load the model to")
    parser.add_argument("--save_file", type=str, help="Path to save results in JSON Lines format.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams to use for decoding.")
    parser.add_argument("--model_type", type=str, default="whisper", help="Model type: 'whisper' or 'wav2vec'.")

    
    args = parser.parse_args()
    process_audio_data(args.dataset_path, args.split, args.text_field, args.model_path, args.name,
                       args.num_examples, args.task, args.language, args.print_predictions, args.calculate_wer,
                       args.device, args.save_file, args.extra_clean, args.super_normalize, args.num_beams, args.model_type)
