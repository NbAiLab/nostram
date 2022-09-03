
import json
import sys
import os
import jsonlines
from argparse import ArgumentParser
import pandas as pd
import glob
import urllib.request



def main(args):
    segments_pattern = os.path.join(args.directory,'segments/','*.json')
    segments_list = glob.glob(segments_pattern)
    
    subtitles_pattern = os.path.join(args.directory,'subtitles/','*.json')
    subtitles_list = glob.glob(subtitles_pattern)
    

    for s in [segments_list,subtitles_list]:
        dfs = []
        for file in s:
            data = pd.read_json(file, lines=True) # read data frame from json file
            dfs.append(data) # append the data frame to the list
    
        df = pd.concat(dfs, ignore_index=True) # concatenate all the data frames in the list.
        
        save_images(df['serieimageurl'].unique())
        save_images(df['programimageurl'].unique())
       
        # Create extra dataframes for each category
        categories = {}
        for cat in df.category.unique():
            categories[cat] = df[df['category'] == cat]

        breakpoint()

        programs = df.groupby(["title"])['duration'].agg(['sum','count']).reset_index()
        programs['hours'] = (programs['sum']/100/3600).round(1)
        programs = programs.drop(columns=['sum'])
        programs = programs.rename(columns={"count": "segments"})

        programs_detailed = df.groupby(["title","program_id","subtitle","category"])['duration'].agg(['sum','count']).reset_index()
        programs_detailed['hours'] = (programs_detailed['sum']/100/3600).round(1)
        programs_detailed = programs_detailed.drop(columns=['sum'])
        programs_detailed = programs_detailed.rename(columns={"count": "segments"})
       
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

def save_images(imagelist,save_dir="cachedimages"):
    for url in imagelist:
        image_name = url.split("/")[-1]+".jpg"
        image_path = os.path.join(save_dir,image_name)
        
        if not os.path.exists(image_path):
            print("Saving image "+ image_path)
            urllib.request.urlretrieve(url, image_path)




def parse_args():
    # Parse commandline
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", dest="directory", help="Directory to Json-lines file to analyse", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
