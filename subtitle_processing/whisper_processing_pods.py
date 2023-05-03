import jax
import jax.numpy as jnp
from datasets import load_dataset
from whisper_jax import FlaxWhisperPipline
from jiwer import wer, cer
import jsonlines
import numpy as np

def process_dataset(dataset, output_folder):
    pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16)

    output_filename = "processed_output.jsonl"
    output_path = f"{output_folder}/{output_filename}"

    total_wer = 0
    total_cer = 0
    count = 0

    with jsonlines.open(output_path, "w") as output_file:
        for example in dataset:
            audio = {
                'array': np.array(example['audio']['array']),
                'sampling_rate': example['audio']['sampling_rate']
            }
            text = example['text']
            mp3_id = example['id']

            transcription = pipeline(audio,task="transcribe")['text'].strip()

            current_wer = wer(text, transcription)
            current_cer = cer(text, transcription)
            total_wer += current_wer
            total_cer += current_cer
            count += 1

            output_line = {
                "id": mp3_id,
                "text": text,
                "whisper-large-v2": transcription,
                "wer": current_wer,
                "cer": current_cer
            }
            output_file.write(output_line)

            if count % 1000 == 0:
                print(f"Processed {count} samples")

    print(f"Total WER: {total_wer / count}")
    print(f"Total CER: {total_cer / count}")

def main():
    num_of_hosts = jax.process_count()
    current_host_idx = jax.process_index()

    dataset = load_dataset("NbAiLab/NCC_speech_v5_mini", split="train", streaming=True)
    node_dataset = split_dataset_by_node(dataset, rank=current_host_idx, world_size=num_of_hosts)


    output_folder = '/home/perk/models/output'
    process_dataset(node_dataset, output_folder)

if __name__ == "__main__":
    main()

