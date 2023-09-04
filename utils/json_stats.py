import os
import argparse

def format_number(number):
    return '{:,}'.format(number).rjust(10)

def get_json_stats(target_dir):
    stats = {}
    for root, dirs, files in os.walk(target_dir):
        json_files = [f for f in files if f.endswith('.json')]
        if json_files:
            relative_root = os.path.relpath(root, target_dir)
            stats[relative_root] = {}
            for json_file in json_files:
                file_path = os.path.join(root, json_file)
                with open(file_path, 'r') as f:
                    line_count = sum(1 for _ in f)
                stats[relative_root][json_file] = line_count
    return stats

def generate_markdown(stats):
    print(f"## Target Directory: {args.target_dir}")
    print("| Directory | File | Lines     |")
    print("| --------- | ---- | ---------:|")
    
    for dir_path, json_stats in stats.items():
        indents = dir_path.count(os.sep)
        padding = "&nbsp;" * (indents * 8) 
        
        for json_file, line_count in json_stats.items():
            formatted_count = format_number(line_count)
            level = f"{padding}{dir_path}"
            print(f"| {level} | {json_file} | {formatted_count} |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Markdown file with JSON stats.")
    parser.add_argument("target_dir", help="Target directory to start the search.")
    args = parser.parse_args()
    
    stats = get_json_stats(args.target_dir)
    generate_markdown(stats)
 
