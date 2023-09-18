#!/usr/bin/env python3
import argparse
import os

def main(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Extract the last field, which is the output file path for ffmpeg
            output_path = line.strip().split(' ')[-1]

            # Check if the file exists and if its size is greater than 1000 bytes
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                continue
            
            # Write the line to the output file
            outfile.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter ffmpeg commands based on output file conditions")
    parser.add_argument("--input_file", required=True, help="Path to the input file containing ffmpeg commands")
    parser.add_argument("--output_file", required=True, help="Path to the output file to store filtered ffmpeg commands")
    
    args = parser.parse_args()
    main(args.input_file, args.output_file)

