import argparse

import numpy as np
import pandas as pd


def remove_double_speakers(uncleaned_df, cleaned_df):
    double_speakers = uncleaned_df.text.str.match(r".*(^|<p>)[\-–—][^<>]*(?<![\-–—])<br>[\-–—]")
    to_be_removed = []
    for i, row in double_speakers.iterrows():
        same_program = cleaned_df.program_id == row.program_id
        filtered = cleaned_df[same_program]
        start_included = filtered.start_time.between(row.start_time, row.end_time, inclusive="neither")
        end_included = filtered.end_time.between(row.start_time, row.end_time, inclusive="neither")
        filtered = filtered[start_included | end_included]
        to_be_removed.extend(filtered.id.to_list())
    to_be_removed = set(to_be_removed)
    cleaned_df.loc[cleaned_df.id.isin(to_be_removed), "timestamp_text"] = np.nan
    return cleaned_df


def main(uncleaned_json, cleaned_json, output_file_name):
    uncleaned_df = pd.read_json(uncleaned_json, lines=True)
    cleaned_df = pd.read_json(cleaned_json, lines=True)
    cleaned_df = remove_double_speakers(uncleaned_df, cleaned_df)
    cleaned_df.to_json(output_file_name, orient="records", lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_original_json", type=str, required=True)
    parser.add_argument("--input_cleaned_json", type=str, required=True)
    parser.add_argument("--output_file_name", type=str, required=True)
    args = parser.parse_args()

    main(
        uncleaned_json=args.input_original_json,
        cleaned_json=args.input_cleaned_json,
        output_file_name=args.output_file_name,
    )
