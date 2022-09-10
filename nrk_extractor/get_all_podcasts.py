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

def main(args):
    error = 0
    podcast_file = os.path.join(args.output_path,"podcast.csv")

    with open(podcast_file, 'w', encoding='UTF8', newline='') as writer:
        seconds = 0
        fieldnames = ['id']
        writer = csv.DictWriter(writer, fieldnames=['id'])
        base_url = "https://psapi.nrk.no" 
        all_url = base_url+"/radio/search/categories/podcast?take=1000"
        all_json = get_json(all_url)
        
        for n,serie in enumerate(all_json['series']):
            
            if 'podcast' in serie['_links']:
                print(f"\n-#{n} Podcast - {serie['title']}")
                serie_url = base_url+serie['_links']['podcast']['href']
                serie_json = get_json(serie_url)
                
                if 'seasons' in serie_json['_embedded']:
                    for season in serie_json['_embedded']['seasons']:
                        print(f"--season - {season['titles']['title']} ({season['episodeCount']} episodes)")
                        season_url = base_url+season['_links']['self']['href']
                        season_json = get_json(season_url)
                        if not season_json: continue

                        for episode in season_json['_embedded']['episodes']['_embedded']['episodes']:
                            print(f"---episode - {episode['titles']['title']} - {episode['episodeId']}")
                            seconds += isodate.parse_duration(episode['duration']).total_seconds()

                else:
                    for episode in serie_json['_embedded']['episodes']['_embedded']['episodes']:
                        print(f"---episode - {episode['titles']['title']} - {episode['episodeId']}")
                        seconds += isodate.parse_duration(episode['duration']).total_seconds()

            if 'customSeason' in serie['_links']:
                print(f"\n-#{n} customSeason - {serie['title']}")
                serie_url = base_url+serie['_links']['customSeason']['href']
                serie_json = get_json(serie_url)
                if not serie_json: continue

                for episode in serie_json['_embedded']['episodes']['_embedded']['episodes']:
                    print(f"---episode - {episode['titles']['title']} - {episode['episodeId']}")
                    seconds += isodate.parse_duration(episode['duration']).total_seconds()

        ##Print line with podcast id
        id = 999
        row = {'id': id}
                            
        print('.', end='', flush=True)
                            
        writer.writerows([row])       
        

        print(f"Total time: {round(seconds/3600)}") 
        print(f"Finished writing csv output file to {(podcast_file)}")


def get_json(url):
    r = requests.get(url)
    if r.status_code != 200:
        #raise Exception("Failed to load metadata from '%s'" % url)
        return False   

    return json.loads(r.text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', help="Complete path to csv output file. The exact name will be given in the program.", required=True)


    args = parser.parse_args()
    main(args)








