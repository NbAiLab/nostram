import os
import argparse

def format_number(number):
    """Format the number with commas and right-justify it."""
    return '{:,}'.format(number).rjust(10)

def get_json_stats(target_dir):
    """Walk through the target directory and collect stats about JSON files."""
    stats = {}
    for root, dirs, files in os.walk(target_dir):
        json_files = [f for f in files if f.endswith('.json')]
        relative_root = os.path.relpath(root, target_dir)
        stats[relative_root] = {}
        for json_file in json_files:
            file_path = os.path.join(root, json_file)
            with open(file_path, 'r') as f:
                line_count = sum(1 for _ in f)
            stats[relative_root][json_file] = line_count
    return stats

def generate_markdown(stats, target_dir):
    """Generate Markdown tables for the collected stats."""
    print(f"## Target Directory: {target_dir}")

    first_level_dirs = {k.split(os.sep)[0] for k in stats.keys() if k != '.'}

    for first_level_dir in sorted(first_level_dirs):
        print(f"### Directory: {first_level_dir}")
        print("| Directory | File | Lines     |")
        print("| --------- | ---- | ---------:|")
        
        total_lines = 0

        for dir_path, json_stats in sorted(stats.items()):
            if "mp3" in dir_path:
                continue
            if not dir_path.startswith(first_level_dir):
                continue

            if not json_stats:
                print(f"| {dir_path} | <empty> | <empty>    |")
                continue
            
            for json_file, line_count in json_stats.items():
                total_lines += line_count
                formatted_count = format_number(line_count)
                print(f"| {dir_path} | {json_file} | {formatted_count} |")

        print(f"| **Total** |      | **{format_number(total_lines)}** |")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Markdown file with JSON stats.")
    parser.add_argument("target_dir", help="Target directory to start the search.")
    args = parser.parse_args()

    stats = get_json_stats(args.target_dir)
    generate_markdown(stats, args.target_dir)

