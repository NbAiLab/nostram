import argparse
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

def convert_to_whisper_format(timestamps):
    whisper_timestamps = ''
    for ts in timestamps:
        start_time = '{:.2f}'.format(ts['timestamp'][0])
        end_time = '{:.2f}'.format(ts['timestamp'][1])
        #Remove em dash
        text = ts['text'].replace('—', '')

        # Generate the timestamp
        whisper_timestamps += ' <|{}|>{}<|{}|>'.format(start_time, text, end_time)
        
        # Remove any double spacing
        whisper_timestamps = ' '.join(whisper_timestamps.split())
    
    return whisper_timestamps



def process_dataset(dataset, output_filename, return_timestamps, add_whisper_timestamps):
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
            
            transcription = pipeline(audio,task="transcribe", return_timestamps=return_timestamps)
            transcribed_text = transcription['text'].strip()
            transcribed_chunks = transcription['chunks']

            current_wer = wer(text, transcribed_text)
            current_cer = cer(text, transcribed_text)
            total_wer += current_wer
            total_cer += current_cer
            count += 1

            output_line = {
                "id": mp3_id,
                "text": text,
                "whisper-large-v2": transcription["text"],
                "whisper-large-v2-chunks": transcribed_chunks,
                "wer": current_wer,
                "cer": current_cer
            }
            
            if add_whisper_timestamps:
                
                # If timestamp corpus - print only when timestamped text
                if float(current_cer)<=0.1:
                    whisper_timestamp_string = convert_to_whisper_format(transcribed_chunks)	
                    print(whisper_timestamp_string)
                    
                    output_line["text_with_timestamps"] = whisper_timestamp_string
                    output_file.write(output_line)
            else:
                output_file.write(output_line)
            

            if count % 1000 == 0:
                print(f"Processed {count} samples")

    print(f"Total WER: {total_wer / count}")
    print(f"Total CER: {total_cer / count}")

def main(args):
    num_of_hosts = jax.process_count()
    current_host_idx = jax.process_index()

    dataset = load_dataset(args.dataset_name, split=args.dataset_split_name, streaming=True)
    node_dataset = split_dataset_by_node(dataset, rank=current_host_idx, world_size=num_of_hosts)
    
    output_filename = args.output_foldername + "/processed_idx"+str(current_host_idx)+".json"
    process_dataset(node_dataset, output_filename, args.return_timestamps, args.return_timestamps_corpus)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--dataset_split_name', type=str, required=True, help='Dataset split name')
    parser.add_argument('--output_foldername', type=str, required=True, help='Output folder name')
    parser.add_argument('--return_timestamps', action='store_true', help='Return timestamps')
    parser.add_argument('--add_whisper_timestamps', action='store_true', help='Add whisper timestamps')

    args = parser.parse_args()
    main(args)

