import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm


def remove_double_speakers(uncleaned_df, cleaned_df):
    to_be_removed = []
    timestamped = cleaned_df[~cleaned_df.timestamped_text.isna()].copy()
    timestamped = timestamped[timestamped.source.str.contains("nrk")]
    if len(timestamped) <= 0:
        print("Found no NRK samples")
        return cleaned_df
    timestamped[["program", "start_time", "end_time"]] = timestamped.id.str.split("_", expand=True)
    timestamped["start_time"] = timestamped["start_time"].astype(int)
    timestamped["end_time"] = timestamped["end_time"].astype(int)
    double_speakers = uncleaned_df.text.str.match(r".*(^|<p>)[\-–—][^<>]*(?<![\-–—])<br>[\-–—]")
    double_speakers = uncleaned_df[double_speakers]
    print("Found", len(double_speakers), "samples with multiple speakers")
    common_programs = set(double_speakers.program_id) & set(timestamped.group_id)
    it = tqdm(common_programs)
    for program in it:
        filtered = timestamped[timestamped.group_id == program]
        for i, row in double_speakers[double_speakers.program_id == program].iterrows():
            start_included = filtered.start_time.between(row.start_time, row.end_time, inclusive="neither")
            end_included = filtered.end_time.between(row.start_time, row.end_time, inclusive="neither")
            whole_included = (filtered.start_time <= row.start_time) & (row.end_time <= filtered.end_time)
            drop_these = filtered[start_included | end_included | whole_included]
            # breakpoint()
            if len(drop_these) > 0:
                # print("Filtering out", drop_these.id.to_list())
                to_be_removed.extend(drop_these.id.to_list())
        # it.update()
        it.set_postfix_str(f"removing={len(to_be_removed)}, current={row.program_id}")
    to_be_removed = set(to_be_removed)
    # cleaned_df = cleaned_df[cleaned_df.id.isin(to_be_removed)]
    cleaned_df.loc[cleaned_df.id.isin(to_be_removed), "timestamped_text"] = np.nan
    return cleaned_df


def main(uncleaned_json, cleaned_json, output_file_name):
    print("Loading files")
    uncleaned_df = pd.read_json(uncleaned_json, lines=True)
    cleaned_df = pd.read_json(cleaned_json, lines=True)
    print("Files loaded, starting filtering")
    cleaned_df = remove_double_speakers(uncleaned_df, cleaned_df)

    with open(output_file_name, 'w', encoding='utf-8') as file:
        cleaned_df.to_json(file, orient='records', lines=True, force_ascii=False)
    print("Saved to", output_file_name)


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
