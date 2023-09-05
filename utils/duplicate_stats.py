import pandas as pd
import argparse

def main(input_file, min_duplicates):
    # Read JSON data into a Pandas DataFrame
    df = pd.read_json(input_file, lines=True)
    
    # Count duplicates for 'text' field
    duplicate_count = df['text'].value_counts()
    
    # Filter texts with at least min_duplicates occurrences
    filtered_count = duplicate_count[duplicate_count >= min_duplicates]
    
    # Generate Markdown Table
    print("| text | number_of_duplicates |") # id's |")
    print("|------|----------------------|") #------|")
    
    for text, num in filtered_count.items():
        # Find ids corresponding to the duplicate text
        #ids = ', '.join(map(str, df[df['text'] == text]['id'].tolist()))
        
        # HTML code for expandable id's
        #expandable_html = f"<details><summary>Show/Hide</summary><p>{ids}</p></details>"
        
        # Print entire sentence (text), number of duplicates and expandable ids in Markdown table format
        print(f"| {text} | {num} |") # {expandable_html} |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Input file path.')
    parser.add_argument('--min_duplicates', type=int, default=5, help='Minimum number of duplicates to include.')
    args = parser.parse_args()

    main(args.input_file, args.min_duplicates)

