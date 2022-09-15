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

##################################################################################
# Get all tv episodes and series - from category interface - creates a json file #
##################################################################################

def main(args):
    error = 0
    valid_manifest = 0
    invalid_manifest = 0
    tv_file = os.path.join(args.output_path,"tv.json")

    with jsonlines.open(tv_file, mode='w') as writer:
        seconds = 0
        base_url = "https://psapi.nrk.no" 
       
        r = requests.get(base_url+"/tv/pages/")
        if r.status_code != 200:
            raise Exception("Failed to load metadata from '%s'" % murl)
            
        categories_json = json.loads(r.text)
            
        for category in categories_json['pageListItems']:
            print(f"\nProcessing Category {category['id']}")
            
            all_url = base_url+category['_links']['self']['href']
            all_json = get_json(all_url)
            
            for section in all_json['sections']:
                print(f"\n-Proceccing Section {section['included']['title']}")
                

                for n,item in enumerate(section['included']['plugs']):
                    itemtype = item['targetType']

                    if 'series' == itemtype:
                        print(f"\n--#{n} Processing {itemtype} {item['displayContractContent']['contentTitle'].strip()}")
                        
                        serie_json = get_json(base_url+item[item['targetType']]['_links']['self']['href'])
                        if not serie_json:
                            print("\n\n***ERROR Serie Json\n")
                            continue
                    
                        if 'seasons' in serie_json:
                            for season in serie_json['seasons']:
                                print(f"\n--season - {season['name']}")
                                season_json = get_json(base_url+'/tv/catalog/series/'+serie_json['id']+'/seasons/'+season['name'])
                                 
                                if not season_json: 
                                    print("\n\n***ERROR Season Json\n")
                                    continue
                               
                                if not 'episodes' in season_json['_embedded']:
                                    try:
                                        season_json['_embedded']['episodes'] = season_json['_embedded']['instalments']
                                    except:
                                        print("\n\n***ERROR Season Json Episodes\n")
                                        continue
                                
                                for episode in season_json['_embedded']['episodes']:
                                    episode_seconds = write_episode(episode,writer,season_json['image'][0]['url'])
                                    seconds += episode_seconds
                                    if episode_seconds:
                                        valid_manifest += 1
                                    else:
                                        invalid_manifest +=1
                                
                        else:
                            print("This happens for radio, but I have not found it in TV. It is therefore untested. If it crashes here, uncomment, and restart")
                            breakpoint()
                            
                            #for episode in serie_json['_embedded']['episodes']:
                            #    episode_seconds = write_episode(episode, writer, serie_json['series']['image'][0]['url'])
                            #    seconds += episode_seconds
                            #    if episode_seconds:
                            #        valid_manifest += 1
                            #    else:
                            #        invalid_manifest +=1

                    
                    elif 'episode' == itemtype or 'standaloneProgram' == itemtype:
                        #There seem to be very few like this, and the structure is very different. Therefore we simply drop them
                        print(f"\n{itemtype} that is dropped.\n")
                        continue
                        #episode = get_json(base_url+item['episode']['_links']['self']['href'])
                        #episode_seconds = write_episode(episode, writer)
                        #seconds += episode_seconds
                        #if episode_seconds:
                        #    valid_manifest += 1
                        #else:
                        #    invalid_manifest +=1
                    
                    elif 'channel' == itemtype:
                        print("\n\n***ERROR Channel\n")
                        continue

                    else:
                        print("This should not happen!")
                        print(item['_links'])
                        breakpoint()

        print(f"\nTotal time: {round(seconds/3600)} hours.") 
        print(f"\nThere were a total of {valid_manifest} episodes with valid manifest files, and {invalid_manifest} episodes with an invalid one.")
        print(f"\nFinished writing json output file to {(podcast_file)}")

def write_episode(episode,writer,serie_image_url="None"):
        base_url = "https://psapi.nrk.no"   
        episode_id = episode['prfId']

        medium = episode['_links']['self']['href'].split("/")[1] 
        program_image_url = episode['image'][0]['url']

        title = episode['titles']['title']
        subtitle = episode['titles']['subtitle']
        year = episode['productionYear']
        
        #Availability - Take this from the manifest-file since it is more accurate
        # Get the playback file
        playback_json = get_json(base_url+episode['_links']['playbackmetadata']['href'])

        #Get the manifest-file
        try:
            manifest_json = get_json(base_url+playback_json['_links']['manifests'][0]['href'])

            availability_information = manifest_json['availability']['information']
            is_geoblocked = manifest_json['availability']['isGeoBlocked']
            external_embedding_allowed = manifest_json['availability']['externalEmbeddingAllowed']
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
            external_embedding_allowed = "unknown"
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
                'program_image_url': program_image_url, 
                'serie_image_url':serie_image_url,
                'title':title,
                'subtitle': subtitle,
                'year':year,
                'duration': duration,
                'availability_information':availability_information,
                'is_geoblocked':is_geoblocked,
                'external_embedding_allowed':external_embedding_allowed,
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








