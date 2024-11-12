import pyarrow.parquet as pq
import torch
import torchaudio
import pandas as pd
import base64
import io
import argparse
from torch_vggish_yamnet import yamnet
from torch_vggish_yamnet.input_proc import WaveformToInput

# Load YAMNet model with pretrained weights
model = yamnet.yamnet(pretrained=True)
model.eval()

# Load YAMNet class names from the CSV file
class_map_path = "yamnet_class_map.csv"  # Path to the class map file
class_map = pd.read_csv(class_map_path, sep=",", quotechar='"')["display_name"].tolist()

# Define human-related classes
human_voice_classes = {
    "Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue",
    "Speech synthesizer", "Shout", "Yell", "Whispering", "Silence"
}

# Function to load and preprocess the audio file from base64-encoded bytes
def load_audio_from_bytes(audio_bytes, target_sample_rate=16000):
    try:
        audio_stream = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(audio_stream, format='mp3')  # Assuming MP3 format
        if sample_rate != target_sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
        return waveform
    except Exception as e:
        print(f"Error opening audio data: {e}")
        return None

# Map audio to YAMNet classes with non-human speech threshold
def map_audio_to_yamnet_classes(audio_data, id, no_speech_threshold=0.9):
    print(f"\nProcessing ID: {id} with no-speech threshold: {no_speech_threshold * 100}% confidence")
    
    # Convert the audio data to waveform
    waveform = load_audio_from_bytes(audio_data)
    if waveform is None:
        print(f"Error processing audio for ID {id}: Unable to load audio data.")
        return
    
    # Convert the waveform to YAMNet's input format
    converter = WaveformToInput()
    input_tensor = converter(waveform.squeeze(0).float(), 16000)  # Assumes 16kHz sample rate
    
    # Pass through YAMNet model to get embeddings and logits
    with torch.no_grad():
        embeddings, logits = model(input_tensor)
    
    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)
    
    # Calculate predictions and apply thresholds for classification
    non_human_count = 0
    non_human_details = {}

    for i, prob_dist in enumerate(probabilities):
        max_prob, class_idx = torch.max(prob_dist, dim=0)
        predicted_class = class_map[class_idx]
        
        if predicted_class not in human_voice_classes and max_prob >= no_speech_threshold:
            non_human_count += 1
            if predicted_class not in non_human_details:
                non_human_details[predicted_class] = 0
            non_human_details[predicted_class] += 1
    
    # Display results
    print(f"Non-human frames (above threshold): {non_human_count}")
    print("Detailed breakdown of non-human classes with high confidence:")
    for cls, count in non_human_details.items():
        print(f"{cls}: {count} frames")

# Main function to process the Parquet file
def main(parquet_file, no_speech_threshold=0.9):
    # Read the Parquet file
    table = pq.read_table(parquet_file)
    records = table.to_pandas()  # Convert to Pandas DataFrame for easy iteration
    
    for index, row in records.iterrows():
        audio_base64 = row['audio']['bytes']  # Adjust if audio is stored differently
        audio_bytes = base64.b64decode(audio_base64)  # Decode base64 to bytes
        map_audio_to_yamnet_classes(audio_bytes, row['ID'], no_speech_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio from a Parquet file.")
    parser.add_argument("--parquet_file", type=str, required=True, help="Path to the input Parquet file.")
    parser.add_argument("--no_speech_threshold", type=float, default=0.9, help="Confidence threshold for non-human speech detection.")

    args = parser.parse_args()
    main(args.parquet_file, args.no_speech_threshold)
