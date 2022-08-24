
import json
import sys
import os
import jsonlines
from argparse import ArgumentParser
import pandas as pd
import glob



def main(args):
    json_pattern = os.path.join(args.directory,'*.jsonl')
    file_list = glob.glob(json_pattern)
    segments_list = [x for x in file_list if not '_subtitles' in x]
    subtitles_list = [x for x in file_list if '_subtitles' in x]
   

    for s in [segments_list,subtitles_list]:
        dfs = []
        for file in s:
            data = pd.read_json(file, lines=True) # read data frame from json file
            dfs.append(data) # append the data frame to the list
    
        df = pd.concat(dfs, ignore_index=True) # concatenate all the data frames in the list.
    
        programs = df.groupby(["title"])['duration'].agg(['sum','count']).reset_index()
        programs['hours'] = (programs['sum']/100/3600).round(1)
        programs = programs.drop(columns=['sum'])
        programs = programs.rename(columns={"count": "segments"})

        programs_detailed = df.groupby(["title","program_id","subtitle","category"])['duration'].agg(['sum','count']).reset_index()
        programs_detailed['hours'] = (programs_detailed['sum']/100/3600).round(1)
        programs_detailed = programs_detailed.drop(columns=['sum'])
        programs_detailed = programs_detailed.rename(columns={"count": "segments"})
       
        breakpoint()
        if s == segments_list:
            save_file = "stats.md"
            title="# NRK Programs Processed\n"
        else:
            save_file = "stats_subtitles.md"
            title="# NRK Subtitles Extracted\n"

        with open(save_file, 'w') as f:
            f.write(title)
            f.write(programs.to_markdown(index=False))
            f.write("\n\n")
            f.write(f"\n**A total of {round(df['duration'].sum()/100/3600,1)} hours in the dataset**")
            f.write("<details><summary>View detailed summary</summary>\n")
            f.write("## Detailed View\n")
            f.write(programs_detailed.to_markdown(index=False))
            f.write("</details>\n")
        print(save_file+" written to disk")
    

def parse_args():
    # Parse commandline
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", dest="directory", help="Directory to Json-lines file to analyse", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
