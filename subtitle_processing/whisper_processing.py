import jsonlines
import numpy as np
from jax import numpy as jnp
from datasets import load_dataset
from whisper_jax.pipeline import FlaxWhisperPipline
from jiwer import wer

def process_dataset(dataset, output_folder):
    # Initialize the Whisper pipeline
    pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, task="transcribe", language="no")

    # Create a single JSONL file for the output
    output_filename = "processed_output.jsonl"
    output_path = f"{output_folder}/{output_filename}"

    total_wer = 0
    count = 0

    with jsonlines.open(output_path, "w") as output_file:
        for example in dataset:
            audio = {
                'array': np.array(example['audio']['array']),
                'sampling_rate': example['audio']['sampling_rate']
            }
            text = example['text']
            mp3_id = example['id']

            # Get the transcription from the Whisper pipeline
            transcription = pipeline(audio)['text'].strip()

            # Calculate the Word Error Rate (WER)
            current_wer = wer(text, transcription)
            total_wer += current_wer
            count += 1

            # Save the result in JSONL format
            output_line = {
                "id": mp3_id,
                "text": text,
                "whisper-large-v2": transcription,
                "wer": current_wer
            }
            output_file.write(output_line)

            # Print a message every 1000 samples
            if count % 1000 == 0:
                print(f"Processed {count} samples")

    print(f"Total WER: {total_wer / count}")

def main():
    dataset = load_dataset("NbAiLab/NCC_speech_v5", split="train", streaming=True)
    output_folder = '../output'
    process_dataset(dataset, output_folder)

if __name__ == "__main__":
    main()

