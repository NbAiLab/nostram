import jsonlines
import os
import re

def contains_numeric_info(text):
    digits = re.search(r'\d+', text)
    words = re.search(r'\b(tjue(to|tre|fire|fem|seks|sju|åtte|ni)?|tretti|førti|femti|seksti|sytti|åtti|nitti|(tretti|førti|femti|seksti|sytti|åtti|nitti)(to|tre|fire|fem|seks|sju|åtte|ni)|hundre|tusen|million(er)?|milliard(er)?)\b', text, re.IGNORECASE)
    return bool(digits or words)

target_directory = '/Users/pere/Documents/NCC_speech_all_v5/data/train/'
output_file = '/Users/pere/Documents/numeric_output.json'

positive_count = 0
negative_count = 0

with jsonlines.open(output_file, mode='w') as outfile:
    for file_name in os.listdir(target_directory):
        if file_name.endswith('.json'):
            with jsonlines.open(os.path.join(target_directory, file_name)) as infile:
                for json_obj in infile:
                    if 'text' in json_obj and contains_numeric_info(json_obj['text']):
                        positive_count += 1
                        outfile.write({'id': json_obj['id'], 'text': json_obj['text']})
                    else:
                        negative_count += 1

print(f"Positive count: {positive_count}")
print(f"Negative count: {negative_count}")

