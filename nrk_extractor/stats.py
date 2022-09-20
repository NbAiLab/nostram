
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
    
    #subtitles_pattern = os.path.join(args.directory,'subtitles/','*.json')
    #subtitles_list = glob.glob(subtitles_pattern)
    

    for s in [segments_list]:
        dfs = []
        for file in s:
            data = pd.read_json(file, lines=True) # read data frame from json file
            dfs.append(data) # append the data frame to the list
    
        df = pd.concat(dfs, ignore_index=True) # concatenate all the data frames in the list.
        df['title'] = df['title'].astype(str)
        #df.loc[df['medium'] == 'tv', 'title'] = df['title'] + " (TV)"

        #save_images(df['serie_image_url'].unique())
        #save_images(df['program_image_url'].unique())
       
        # Create extra dataframes for each category
        categories = {}
        
        for cat in df.category.unique():
            
            categories[cat] = df[df['category'] == cat]

        
        if s == segments_list:
            save_file = "stats.md"
            title="# NRK Programs Processed\n"
        else:
            save_file = "stats_subtitles.md"
            title="# NRK Subtitles Extracted\n"

        with open(save_file, 'w') as f:
            f.write(title)
            f.write("## SUMMARY - hours\n")
            f.write("| category              | tv   | radio                    |   **total** |\n")   
            f.write("|:-------------------|-------------:|----------------------------:|---------------------------:|\n")
            for cat in categories:
                tv = "{:,.1f}".format(categories[cat].query("medium == 'tv'")['duration'].sum()/1000/3600,1)
                radio = "{:,.1f}".format(categories[cat].query("medium == 'radio'")['duration'].sum()/1000/3600,1)
                total = "{:,.1f}".format(categories[cat]['duration'].sum()/1000/3600,1)

                f.write("| "+cat+" | "+tv+" | "+radio+"                  |        **"+total+"** |\n")     
            
            tv = "{:,.1f}".format(df.query("medium == 'tv'")['duration'].sum()/1000/3600,1)
            radio = "{:,.1f}".format(df.query("medium == 'radio'")['duration'].sum()/1000/3600,1)
            total = "{:,.1f}".format(df['duration'].sum()/1000/3600,1)

            f.write("| **total** | **"+tv+"** | **"+radio+"**                  |        **"+total+"** |\n\n")  


            for cat in categories:
                programs = {}
                programs_detailed = {}

                programs[cat] = categories[cat].groupby(["serie_image_url","serie_title"])['duration'].agg(['sum','count']).reset_index()
                temp = categories[cat].groupby(["serie_title"])['episode_id'].agg(['nunique']).reset_index()
                programs[cat] = pd.merge(programs[cat],temp)
                programs[cat]['hours'] = (programs[cat]['sum']/1000/3600).round(1)
                programs[cat]['average(s)'] = ((programs[cat]['sum']/programs[cat]['count'])/1000).round(1)
                programs[cat]['serie_image_url'] = '<img src="'+programs[cat]['serie_image_url']+'" height="48">'

                programs[cat] = programs[cat].drop(columns=['sum'])
                programs[cat] = programs[cat].rename(columns={"count": "segments"})
                programs[cat] = programs[cat].rename(columns={"nunique": "programs"})
                programs[cat] = programs[cat][['serie_image_url', 'serie_title', 'programs', 'segments', 'average(s)','hours']]
                programs[cat] = programs[cat].rename(columns={"serie_image_url": " "})
                programs[cat]['serie_title'] = programs[cat]['serie_title'].str.replace('|','-')
                #Format
                programs[cat]['segments'] = programs[cat]['segments'].map('{:,d}'.format)
                #Detailed
                #programs_detailed[cat] = categories[cat].groupby(["program_image_url","title","episode_id","sub_title"])['duration'].agg(['sum','count']).reset_index()
                #programs_detailed[cat]['average(s)'] = ((programs_detailed[cat]['sum']/programs_detailed[cat]['count'])/1000).round(1)
                #programs_detailed[cat]['hours'] = (programs_detailed[cat]['sum']/1000/3600).round(1)
                #programs_detailed[cat] = programs_detailed[cat].drop(columns=['sum'])
                #programs_detailed[cat] = programs_detailed[cat].rename(columns={"count": "segments"})
                #programs_detailed[cat]['program_image_url'] = '<img src="'+programs_detailed[cat]['program_image_url']+'" height="24">'
                #programs_detailed[cat] = programs_detailed[cat].rename(columns={"program_image_url": " "})
                
                #Format
                #programs_detailed[cat]['segments'] = programs_detailed[cat]['segments'].map('{:,d}'.format)

                f.write("## "+cat+"\n") 
                f.write(programs[cat].to_markdown(index=False))
                f.write("\n\n")
                #f.write("<details><summary>More details</summary>\n\n")
                #f.write(programs_detailed[cat].to_markdown(index=False))
                #f.write("\n</details>\n\n")
            print(save_file+" written to disk")

#def save_images(imagelist,save_dir="cachedimages"):
#    for url in imagelist:
#        image_name = url.split("/")[-1]+".jpg"
#        image_path = os.path.join(save_dir,image_name)
#        
#        if not os.path.exists(image_path) and url  != "None":
#            print("Saving image "+ image_path)
#            urllib.request.urlretrieve(url, image_path)

def parse_args():
    # Parse commandline
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", dest="directory", help="Directory to Json-lines file to analyse", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
