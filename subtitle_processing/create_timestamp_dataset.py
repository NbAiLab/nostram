import argparse
import jsonlines
import numpy as np
import os
import re

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CMALLOC_VERBOSE'] = '0'
os.environ['TCMALLOC_VERBOSE'] = '0'
os.environ['TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD'] = '10000000000'

def convert_to_whisper_format(timestamps, remove_em_dash=True):
    
    start_time = 0.00
    end_time = 0.00
    
    whisper_timestamps = ''
    for ts in timestamps:
        if ts['timestamp'][0] == None or ts['timestamp'][1] == None:
            return False
        
        start_time = '{:.2f}'.format(ts['timestamp'][0])
        end_time = '{:.2f}'.format(ts['timestamp'][1])
        
        #Remove em dash
        if remove_em_dash:
            text = ts['text'].replace('—', '')

        # Generate the timestamp
        whisper_timestamps += ' <|{}|>{} <|{}|>'.format(start_time, text, end_time)
        
        # Remove any double spacing
        whisper_timestamps = ' '.join(whisper_timestamps.split())
        
        # The middle timestamps seems to not have space
        whisper_timestamps = whisper_timestamps.replace('|> <|', '|><|')

    # Basic check to see if it is a sound timestamp
    timestamp_pattern = re.compile(r'<\|([\d.]+)\|>')
    timestamps = [float(t) for t in timestamp_pattern.findall(whisper_timestamps)]
    
    if not all(x <= y for x, y in zip(timestamps, timestamps[1:])):
        return False
        
    return whisper_timestamps



def main(input_filename, output_filename, cer):
    with jsonlines.open(input_filename, 'r') as infile, jsonlines.open(output_filename, 'w') as outfile:
        for data in infile:
            factor = 1
            # If there are multiple chunks, we want to be more lenient
            if len(data['whisper-large-v2']['chunks']) > 1:
                factor = 10
                
            if data['cer'] <= (cer * factor):
                
                output_line = {
                    "id": data['id'],
                    "text": convert_to_whisper_format(data['whisper-large-v2']['chunks']),
                }
                
                if output_line['text']:
                    print(output_line)
                    #outfile.write(data)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename', type=str, required=True, help='Input file name')
    parser.add_argument('--output_filename', type=str, required=True, help='Output file name')
    parser.add_argument('--cer', type=float, required=True, help='Minimum CER value')
    
    args = parser.parse_args()
    main(args.input_filename, args.output_filename, args.cer)
    
