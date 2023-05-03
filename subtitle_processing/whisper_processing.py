import jax
import jax.numpy as jnp
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from whisper_jax import FlaxWhisperPipline
from jiwer import wer, cer
import jsonlines
import numpy as np
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CMALLOC_VERBOSE'] = '0'
os.environ['TCMALLOC_VERBOSE'] = '0'
os.environ['TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD'] = '10000000000'

def process_dataset(dataset, output_filename):
    pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16)

    total_wer = 0
    total_cer = 0
    count = 0

    with jsonlines.open(output_filename, "w") as output_file:
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

    dataset = load_dataset("NbAiLab/NCC_speech_v5", split="train", streaming=True)
    node_dataset = split_dataset_by_node(dataset, rank=current_host_idx, world_size=num_of_hosts)


    output_filename = f'/home/perk/models/output/processed_idx{current_host_idx}.json'
    process_dataset(node_dataset, output_filename)

if __name__ == "__main__":
    main()

