from datasets import load_dataset
import torch
import torchaudio
from torch_vggish_yamnet import yamnet
from torch_vggish_yamnet.input_proc import WaveformToInput
import pandas as pd
import argparse

# Load YAMNet model with pretrained weights
model = yamnet.yamnet(pretrained=True)
model.eval()

# Load YAMNet class names from the CSV file
class_map_path = "yamnet_class_map.csv"  # Path to the class map file
class_map = pd.read_csv(class_map_path, sep=",", quotechar='"')["display_name"].tolist()

# Define speech-related classes
speech_classes = {
    "Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue",
    "Speech synthesizer", "Shout", "Yell", "Whispering", "Silence"
}

# Function to ensure waveform has the correct shape
def check_waveform_shape(waveform, expected_sample_rate=16000):
    # Reshape to ensure waveform is in [1, num_samples] format if it's a 1D tensor
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Add batch dimension
    elif waveform.ndim == 2 and waveform.size(0) > 1:
        # Average multiple channels if stereo or multi-channel
        waveform = waveform.mean(dim=0, keepdim=True)
    
    return waveform

# Function to classify audio frames by summing probabilities for speech classes
def classify_audio_frames(waveform, no_speech_threshold=0.9):
    # Ensure the waveform shape is correct
    waveform = check_waveform_shape(waveform)
    converter = WaveformToInput()
    input_tensor = converter(waveform.float(), 16000)  # Assumes 16kHz sample rate

    # Pass through YAMNet model to get embeddings and logits
    with torch.no_grad():
        embeddings, logits = model(input_tensor)

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Initialize counts
    speech_count = 0
    non_speech_count = 0
    non_speech_details = {}

    for prob_dist in probabilities:
        # Sum probabilities for all speech-related classes
        speech_prob_sum = sum(prob_dist[class_map.index(cls)] for cls in speech_classes if cls in class_map)

        # Classify based on speech probability sum and threshold
        if speech_prob_sum >= no_speech_threshold:
            speech_count += 1
        else:
            non_speech_count += 1
            # Identify the most probable non-speech class
            max_prob, class_idx = torch.max(prob_dist, dim=0)
            predicted_class = class_map[class_idx]
            if predicted_class not in speech_classes:
                non_speech_details[predicted_class] = non_speech_details.get(predicted_class, 0) + 1

    return speech_count, non_speech_count, non_speech_details

# Main function to load the dataset and classify each sample
def main(parquet_file, no_speech_threshold):
    # Load dataset
    dataset = load_dataset("parquet", data_files=parquet_file)["train"]

    # Iterate through the dataset and process each sample
    for i, sample in enumerate(dataset):
        audio_tensor = torch.tensor(sample["audio"]["array"])
        waveform = check_waveform_shape(audio_tensor)

        try:
            speech_count, non_speech_count, non_speech_details = classify_audio_frames(waveform, no_speech_threshold)
            print(f"Sample {i + 1} - Speech Frames: {speech_count}, Non-speech Frames: {non_speech_count}")
            for cls, count in non_speech_details.items():
                print(f"  {cls}: {count} frames")
        except Exception as e:
            print(f"Error processing audio for ID {sample['id']}: {e}")

# Argument parser for command-line usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify speech and non-speech frames in an audio dataset.")
    parser.add_argument("--parquet_file", type=str, required=True, help="Path to the input Parquet file.")
    parser.add_argument("--no_speech_threshold", type=float, default=0.9, help="Threshold for non-speech classification.")

    args = parser.parse_args()
    main(args.parquet_file, args.no_speech_threshold)
