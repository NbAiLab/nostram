import requests
import pandas as pd
import argparse

def fetch_md_content(model_name, token):
    headers = {
        'Authorization': f'Bearer {token}'
    }
    url = f'https://huggingface.co/{model_name}/resolve/main/README.md'
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

def extract_table_from_md(md_content):
    lines = md_content.split('\n')
    table_start = None
    table_end = None
    for idx, line in enumerate(lines):
        if line.startswith('|'):
            if table_start is None:
                table_start = idx
            table_end = idx
    return lines[table_start:table_end+1]

def md_table_to_data(table_content, model_name):
    headers = table_content[0].strip('|').split('|')
    headers = [header.strip() for header in headers]
    
    data = []
    for row in table_content[2:]:
        values = row.strip('|').split('|')
        values = [value.strip() for value in values]
        
        # Check for missing values
        if "" in values:
            print(f"Error: Missing values in row for model {model_name}. This row will be skipped.")
            continue
        
        entry = {headers[i]: values[i] for i in range(len(headers))}
        entry['model_name'] = model_name
        data.append(entry)
    
    return headers, data

def main():
    parser = argparse.ArgumentParser(description="Fetch the training results table for a list of model names using the HuggingFace API and write it to a file.")
    parser.add_argument("--model_names", required=True, help="Comma-separated list of model names.")
    parser.add_argument("--output_file", required=True, help="File to save the output JSON lines.")
    parser.add_argument("--token", required=True, help="HuggingFace API token.")
    
    args = parser.parse_args()
    
    all_data = []
    first_headers = None
    for model_name in args.model_names.split(','):
        md_content = fetch_md_content(model_name.strip(), args.token)
        table_content = extract_table_from_md(md_content)
        headers, data = md_table_to_data(table_content, model_name)
        
        # Sanity check for column consistency
        if first_headers is None:
            first_headers = headers
        elif first_headers != headers:
            exit(f"Error: Inconsistent column names for model {model_name}. Exiting.")
        
        all_data.extend(data)
        print(f"Parsed model: {model_name}. Number of data lines: {len(data)}")

    df = pd.DataFrame(all_data)
    df.to_json(args.output_file, orient='records', lines=True)

if __name__ == "__main__":
    main()

