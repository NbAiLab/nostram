import argparse
import os
import glob
import json
from tinytag import TinyTag

def validate_arguments(json_file, audio_path):
    """
    Validate the input arguments: JSON file path and audio directory path.
    """

    # Validate JSON file
    if not os.path.isfile(json_file):
        raise Exception(f"JSON file {json_file} does not exist.")

    # Validate audio path for MP3 files
    mp3_files = glob.glob(os.path.join(audio_path, '*.mp3'))
    if len(mp3_files) == 0:
        raise Exception(f"No MP3 files found in the audio path {audio_path}.")

def main(json_file, audio_path):
    """
    Main function to process a JSON file and corresponding MP3 files.
    """

    # Validate the input arguments
    validate_arguments(json_file, audio_path)

    # Counter for successfully checked MP3 files
    success_counter = 0

    # Open the JSON file
    with open(json_file, 'r') as f:
        # Process each line in the JSON file
        for i, line in enumerate(f):

            # Load JSON data from the line
            entry = json.loads(line)

            # Retrieve the id and audio_duration from the JSON entry
            id = entry.get('id', None)
            audio_duration = entry.get('audio_duration', None)

            # If either the id or audio_duration is missing, skip this entry
            if id is None or audio_duration is None:
                print(f"Skipping entry {i} due to missing id or audio_duration.")
                continue

            # Construct the path to the corresponding MP3 file
            mp3_file = os.path.join(audio_path, f"{id}.mp3")

            # If the MP3 file does not exist, print a message and skip to the next entry
            if not os.path.isfile(mp3_file):
                print(f"MP3 file {mp3_file} does not exist.")
            else:
                # Use TinyTag to get the duration of the MP3 file
                audio = TinyTag.get(mp3_file)
                file_duration = audio.duration
                
                # Convert JSON audio_duration from milliseconds to seconds for comparison
                json_duration = audio_duration / 1000.0

                # Check if the audio file duration matches the JSON duration
                if file_duration < 1 or abs(file_duration - json_duration) > 1:
                    print(f"ERROR: MP3 file {mp3_file} duration mismatch. Expected: {json_duration}s, got: {file_duration}s.")
                else:
                    # Increase the counter for successful checks
                    success_counter += 1

    # Print the number of successfully checked MP3 files
    print(f"\nTotal MP3 files successfully checked: {success_counter}")

if __name__ == '__main__':
    """
    This block is executed when the script is run directly, not when imported as a module.
    It sets up argument parsing and calls the main function.
    """

    # Set up argument parser
    parser = argparse.ArgumentParser(description='A program that processes a JSON file and MP3 files.')
    parser.add_argument('--json_file', type=str, required=True, help='The path to the JSON file.')
    parser.add_argument('--audio_path', type=str, required=True, help='The path to the directory containing MP3 files.')
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.json_file, args.audio_path)

