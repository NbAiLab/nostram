import requests
import json
import re
import os
import subprocess
import time
import sys
import argparse
import isodate
import csv
import dateutil.parser

#################################################
# Get all podcast episode - prints a csv file
#################################################

def main(args):
    error = 0
    podcast_file = os.path.join(args.output_path,"podcast.csv")

    with open(podcast_file, 'w', encoding='UTF8', newline='') as writer:
        seconds = 0
        fieldnames = ['category', 'title', 'year','episode_id','duration']
        writer = csv.DictWriter(writer, fieldnames=fieldnames)
        
        base_url = "https://psapi.nrk.no" 
        all_url = base_url+"/radio/search/categories/podcast?take=1000"
        all_json = get_json(all_url)
        
        for n,serie in enumerate(all_json['series']):
            
            if 'podcast' in serie['_links']:
                print(f"\n-#{n} Podcast - {serie['title']}")
                serie_json = get_json(base_url+serie['_links']['podcast']['href'])
                
                if 'seasons' in serie_json['_embedded']:
                    for season in serie_json['_embedded']['seasons']:
                        print(f"--season - {season['titles']['title']} ({season['episodeCount']} episodes)")
                        season_json = get_json(base_url+season['_links']['self']['href'])
                        if not season_json: continue

                        for episode in season_json['_embedded']['episodes']['_embedded']['episodes']:
                            seconds += write_episode(episode, serie['title'],writer)
                else:
                    for episode in serie_json['_embedded']['episodes']['_embedded']['episodes']:
                        seconds += write_episode(episode, serie['title'],writer)

            if 'customSeason' in serie['_links']:
                print(f"\n-#{n} customSeason - {serie['title']}")
                serie_json = get_json(base_url+serie['_links']['customSeason']['href'])
                if not serie_json: continue

                for episode in serie_json['_embedded']['episodes']['_embedded']['episodes']:
                    seconds += write_episode(episode, serie['title'],writer)

        print(f"Total time: {round(seconds/3600)}") 
        print(f"Finished writing csv output file to {(podcast_file)}")

def write_episode(episode, serie_title, writer):
        print(f"---episode - {episode['titles']['title']} - {episode['episodeId']}")
                            
        year = dateutil.parser.isoparse(episode['date']).year
        duration = isodate.parse_duration(episode['duration']).total_seconds()
        row = {'category':'podcast', 'title': serie_title, 'year': year, 'episode_id': episode['episodeId'], 'duration': duration}
        writer.writerows([row])       
        return duration



def get_json(url):
    r = requests.get(url)
    if r.status_code != 200:
        return False   

    return json.loads(r.text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', help="Complete path to csv output file. The exact name will be given in the program.", required=True)

    args = parser.parse_args()
    main(args)








