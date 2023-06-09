import os
import json
import argparse
from tinytag import TinyTag
import glob
from pathlib import Path

def validate_arguments(json_file, audio_path):
    """
    Validates that the json file and audio directory exist
    """
    if not os.path.isfile(json_file):
        raise Exception(f"File {json_file} does not exist.")
        
    with open(json_file, 'r') as f:
        try:
            # Try to parse file to check if it's a valid json
            for line in f:
                json.loads(line)
        except Exception:
            raise Exception(f"File {json_file} is not a valid JSON file.")

    if not os.path.isdir(audio_path):
        raise Exception(f"Directory {audio_path} does not exist.")
    
    return audio_path

def create_audio_index(audio_path):
    """Create an index of all MP3 files in the given directory and its subdirectories."""
    audio_index = {}
    for root, dirs, files in os.walk(audio_path):
        for file in files:
            if file.endswith(".mp3"):
                audio_index[file] = os.path.join(root, file)
    return audio_index

def main(json_file):
    """Main function for processing the JSON file and audio files."""

    json_file_path = Path(json_file).resolve()
    dataset = json_file_path.parts[-3]  # Extract the dataset name
    audio_path = json_file_path.parents[2] / "audio" / dataset / "audio"

    # Validate the input arguments
    validate_arguments(json_file, str(audio_path))

    audio_index = create_audio_index(str(audio_path))

    # Process the JSON file
    with open(json_file, 'r') as f:
        checked = 0
        errors = 0
        for line in f:
            checked += 1
            data = json.loads(line)
            audio_file_name = f'{data["id"]}.mp3'
            audio_file_path = audio_index.get(audio_file_name)
            

            if audio_file_path is None:
                print(f"Error: File {audio_file_name} does not exist in the audio path.")
                errors += 1
                continue

            tag = TinyTag.get(audio_file_path)
            audio_duration = tag.duration * 1000  # convert to milliseconds

            if audio_duration < 1000:
                print(f"Error: File {audio_file_name} is less than 1 second long.")
                errors += 1
                continue

            if abs(audio_duration - data["audio_duration"]) > 2000:
                print(f"Error: File {audio_file_name} duration mismatch with the json file. Json claims {int(audio_duration)} while mp3 is {data['audio_duration']}")
                errors += 1
                continue

    print(f"\n Checked {checked} mp3-files. Found {errors} errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check mp3 files corresponding to a json.")
    parser.add_argument("json_file", help="Path to the json file.")
    
    args = parser.parse_args()
    main(args.json_file)

