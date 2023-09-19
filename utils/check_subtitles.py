import argparse
import json
import math
from collections import Counter

def check_timestamp_within_duration(json_line, audio_duration):
    error_count = 0
    timestamps = json_line['timestamped_text'].split('<|')[1:]
    for t in timestamps:
        time = float(t.split('|>')[0])
        if time > math.ceil((audio_duration + 25) / 1000):
            print(f"Error: Timestamp out of bounds for ID {json_line['id']}. Audio duration: {audio_duration}. Timestamped text: {json_line['timestamped_text']}")
            error_count += 1
    return error_count

def check_text_within_timestamp(json_line):
    error_count = 0
    segments = json_line['timestamped_text'].split('<|')[1:]
    for segment in segments:
        parts = segment.split('|>')
        if len(parts) != 2:
            continue
        start_time, text = parts
        if not text:
            print(f"Error: Text not within timestamp for ID {json_line['id']}. Timestamped text: {json_line['timestamped_text']}")
            error_count += 1
    return error_count

def check_text_length(json_line, max_length):
    error_count = 0
    segments = json_line['timestamped_text'].split('<|')[1:]
    for segment in segments:
        parts = segment.split('|>')
        if len(parts) != 2:
            continue
        start_time, text = parts
        if len(text) > max_length:
            print(f"Error: Text too long for ID {json_line['id']}. Max length: {max_length}. Audio duration: {json_line['audio_duration']}. Timestamped text: {json_line['timestamped_text']}")
            error_count += 1
    return error_count

def check_max_timespan(json_line, max_timespan):
    error_count = 0
    segments = json_line['timestamped_text'].split('<|')[1:]
    for i in range(1, len(segments)):
        parts1 = segments[i - 1].split('|>')
        parts2 = segments[i].split('|>')
        if len(parts1) != 2 or len(parts2) != 2:
            continue
        start_time1, _ = parts1
        start_time2, _ = parts2
        if float(start_time2) - float(start_time1) > max_timespan:
            print(f"Error: Timespan too long for ID {json_line['id']}. Max timespan: {max_timespan}. Timestamped text: {json_line['timestamped_text']}")
            error_count += 1
    return error_count

def main(input_file, max_length=84, max_timespan=6.0):
    error_counts = Counter()
    
    with open(input_file, 'r') as f:
        for line in f:
            json_line = json.loads(line.strip())
            if json_line['timestamped_text'] == "<|nocaptions|>":
                continue
            audio_duration = json_line['audio_duration']
            error_counts['check_timestamp_within_duration'] += check_timestamp_within_duration(json_line, audio_duration)
            error_counts['check_text_within_timestamp'] += check_text_within_timestamp(json_line)
            error_counts['check_text_length'] += check_text_length(json_line, max_length)
            error_counts['check_max_timespan'] += check_max_timespan(json_line, max_timespan)

    print("Summary of Errors:")
    for test, count in error_counts.items():
        print(f"{test}: {count} errors")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze the validity of timestamps in a json lines file.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input json lines file.')
    parser.add_argument('--max_length', type=int, default=84, help='Maximum text length per timestamp segment.')
    parser.add_argument('--max_timespan', type=float, default=6.0, help='Maximum timespan for a single subtitle.')
    args = parser.parse_args()

    main(args.input_file, args.max_length, args.max_timespan)

