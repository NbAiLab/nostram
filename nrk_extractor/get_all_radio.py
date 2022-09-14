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
import jsonlines

###################################################################################
# Get all radio episodes and series - from search interface - creates a json file #
###################################################################################

def main(args):
    error = 0
    podcast_file = os.path.join(args.output_path,"radio.json")

    with jsonlines.open(podcast_file, mode='w') as writer:
        seconds = 0
        base_url = "https://psapi.nrk.no" 
        
        for i in range(0,10000,100):
            print(f"\nProcessing url #{i}")
            all_url = base_url+"/radio/search/categories/alt-innhold?take=100&skip="+str(i)
            all_json = get_json(all_url)
            if not all_json:
                print("Finished processing the urls from the search!")
                break
        
            for n,item in enumerate(all_json['series']):
                itemtype = str(next(iter(item['_links'])))
                if 'series' == itemtype or 'podcast' == itemtype:
                    #print(f"-#{n} {itemtype} {item['title']}")
                    serie_json = get_json(base_url+item['_links'][str(next(iter(item['_links'])))]['href'])
                    if not serie_json:
                        print("\n\n***ERROR Serie Json\n")
                        continue

                    
                    if 'seasons' in serie_json['_embedded']:
                        for season in serie_json['_embedded']['seasons']:
                            #print(f"--season - {season['titles']['title']} ({season['episodeCount']} episodes)")
                            season_json = get_json(base_url+season['_links']['self']['href'])
                             
                            if not season_json: 
                                print("\n\n***ERROR Season Json\n")
                                continue
                            
                            for episode in season_json['_embedded']['episodes']['_embedded']['episodes']:
                                seconds += write_episode(episode,writer,season_json['image'][0]['url'])
                    else:
                        for episode in serie_json['_embedded']['episodes']['_embedded']['episodes']:
                            seconds += write_episode(episode, writer, serie_json['series']['image'][0]['url'])

                elif 'customSeason' == itemtype:
                    #print(f"-#{n} {itemtype} - {item['title']}")
                    serie_json = get_json(base_url+item['_links'][itemtype]['href'])
                    if not serie_json:
                        print("\n\n***ERROR Serie Json CustomSeason\n")
                        continue
                    
                    for episode in serie_json['_embedded']['episodes']['_embedded']['episodes']:
                        seconds += write_episode(episode,writer,serie_json['image'][0]['url'])

                elif 'singleProgram' == itemtype:
                    #print("singleProgram")
                    episode = get_json(base_url+item['_links'][itemtype]['href'])
                    seconds += write_episode(episode,writer)


                else:
                    print("This should not happen!")
                    print(item['_links'])
                    exit(-1)

        print(f"\nTotal time: {round(seconds/3600)} hours.") 
        print(f"\nFinished writing json output file to {(podcast_file)}")

def write_episode(episode,writer,serie_image_url="None"):
        base_url = "https://psapi.nrk.no"    
        episode_id = episode['episodeId']
        medium = episode['_links']['self']['href'].split("/")[1] 
        program_image_url = episode['image'][0]['url']

        title = episode['titles']['title']
        subtitle = episode['titles']['subtitle']
        year = dateutil.parser.isoparse(episode['date']).year
        
        #Availability - Take this from the manifest-file since it is more accurate
        # Get the playback file
        playback_json = get_json(base_url+episode['_links']['playback']['href'])

        #Get the manifest-file
        try:
            manifest_json = get_json(base_url+playback_json['_links']['manifests'][0]['href'])

            availability_information = manifest_json['availability']['information']
            is_geoblocked = manifest_json['availability']['isGeoBlocked']
        
            duration = round(isodate.parse_duration(manifest_json['playable']['duration']).total_seconds())
            audio_file = manifest_json['playable']['assets'][0]['url']
            audio_format = manifest_json['playable']['assets'][0]['format']
            audio_mime_type = manifest_json['playable']['assets'][0]['mimeType']
            manifest_exist = True
        except:
            #No manifest-file exists
            is_geoblocked = "unknown"
            availability_information = ""
            duration = 0
            audio_file = "unknown"
            audio_format = "unknown"
            audio_mime_type = "unknown"
            manifest_exist = False
            print('M', end='', flush=True)

        try:
            on_demand_from = manifest_json['availability']['onDemand']['from']
            on_demand_to  = manifest_json['availability']['onDemand']['to']        
        except:
            on_demand_from = "undefined"
            on_demand_to = "undefined"

        row = {'episode_id':episode_id, 
                'medium': medium, 
                'programi_image_url': program_image_url, 
                'serie_image_url':serie_image_url,
                'title':title,
                'subtitle': subtitle,
                'year':year,
                'duration': duration,
                'availability_information':availability_information,
                'is_geoblocked':is_geoblocked,
                'on_demand_from':on_demand_from,
                'on_demand_to':on_demand_to,
                'audio_file':audio_file,
                'audio_format':audio_format,
                'audio_mime_type':audio_mime_type}

        if manifest_exist:
            writer.write(row)
            print('.', end='', flush=True)
        
        return duration

def get_json(url):
    r = requests.get(url)
    if r.status_code != 200:
        return False   

    return json.loads(r.text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', help="Complete path to json output file. The exact name will be given in the program.", required=True)

    args = parser.parse_args()
    main(args)








