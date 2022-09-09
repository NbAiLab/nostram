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
    medium = args.medium
    series_file = os.path.join(args.output_file,args.medium+"_series.csv")
    singles_file = os.path.join(args.output_file,args.medium+"_singles.csv")

    with open(series_file, 'w', encoding='UTF8', newline='') as fse:
        with open(singles_file, 'w', encoding='UTF8', newline='') as fsi:
            fieldnames = ['category', 'title', 'first_prod_year', 'last_prod_year','num_episodes','first_episode_id','all_episodes_ids','total_seconds']
            
            fsi_writer = csv.DictWriter(fsi, fieldnames=fieldnames)
            fse_writer = csv.DictWriter(fse, fieldnames=fieldnames)
            fsi_writer.writeheader()
            fse_writer.writeheader()
            
            # Fetch the info blob
            murl = "https://psapi.nrk.no/"+medium+"/pages"
            r = requests.get(murl)
            if r.status_code != 200:
                raise Exception("Failed to load metadata from '%s'" % murl)
            
            res={}
            res["categories-"+medium] = json.loads(r.text)

            # Loop through categories to find series and standalones
            if medium == "radio":
                page_items = "pages"
                target_type = "type"
            else:
                page_items = "pageListItems"
                target_type ="targetType"

            for categories in res["categories-"+medium][page_items]:
                print(f'\n\n* Parsing {categories["_links"]["self"]["href"]}')

                murl = "https://psapi.nrk.no/"+categories["_links"]["self"]["href"]
                r = requests.get(murl)
                if r.status_code != 200:
                    raise Exception("Failed to load metadata from '%s'" % murl)
                res["series-"+medium] = json.loads(r.text)
                
                #Loop through series and standalones
                
                catcount = 0
                for series in res["series-"+medium]["sections"]:
                    
                    #For each of these check how long they are in total
                    for programs in series["included"]["plugs"]:
                        seconds = 0
                        num_episodes = 0
                        first_episode_id = 0
                        all_episodes_ids = []
                        first_prod_year = 2022
                        last_prod_year = 0

                        #print(programs[target_type])
                        if programs[target_type] == "series":
                            try:
                                theserie = get_serie_json(programs['series']['_links']['self']['href'], medium)
                                program_title = programs['displayContractContent']['contentTitle']
                            except:
                                theserie = get_serie_json(programs['_links']['series'],medium)
                                program_title = programs['series']['titles']['title']

                            
                            try:
                                for seasons in theserie['_embedded']['seasons']:
                                    #Loop through the episodes
                                    #The errors here seem to be stuff that are not episodes. Just ignore
                                    single = False

                                    try:
                                        for episodes in seasons['_embedded']['episodes']:
                                            num_episodes += 1
                                            all_episodes_ids.append(episodes['prfId'])
                                            if episodes['productionYear'] <= first_prod_year:
                                                first_prod_year = episodes['productionYear']
                                            if episodes['productionYear'] >= last_prod_year:
                                                last_prod_year = episodes['productionYear']
                                            if first_episode_id == 0:
                                                first_episode_id = episodes['prfId']
                                        
                                            seconds += isodate.parse_duration(episodes['duration']).total_seconds()
                                    except:
                                        #breakpoint()
                                        #print("Missing data. Skipping. Just ignore this if it is not too many.")
                                        error = 1
                                        continue
                            except:
                                #Runs only if there are no seasons
                                season = theserie['_embedded']['episodes']
                                for episodes in season['_embedded']['episodes']:
                                            num_episodes += 1
                                            all_episodes_ids.append(episodes['_links']['self']['href'].split('/')[-1])
                                            if episodes['productionYear'] <= first_prod_year:
                                                first_prod_year = episodes['productionYear']
                                            if episodes['productionYear'] >= last_prod_year:
                                                last_prod_year = episodes['productionYear']
                                            if first_episode_id == 0:
                                                first_episode_id = episodes['_links']['self']['href'].split('/')[-1]

                                            seconds += isodate.parse_duration(episodes['duration']).total_seconds()


                        elif (programs[target_type] == "standaloneProgram") or (programs[target_type] == "episode"):
                            
                            #Get meta information for this program
                            try:
                                theprogram = get_program_json(programs[programs[target_type]]['_links']['self']['href'])
                            except:
                                try:
                                    theprogram = get_program_json(programs['_links'][programs[target_type]])
                                except:
                                    theprogram = get_program_json(programs['_links']['program'])

                            first_prod_year = last_prod_year = theprogram['productionYear']
                            seconds += isodate.parse_duration(theprogram['duration']).total_seconds()
                            first_episode_id = theprogram['id']
                            all_episodes_ids.append(theprogram['id'])
                            
                            try:
                                program_title = programs['displayContractContent']['contentTitle']
                            except:
                                try:
                                    program_title = programs[programs[target_type]]['titles']['title']
                                except:
                                    program_title = programs['program']['titles']['title']

                            num_episodes = 1
                            single = True
                       
                        elif programs[target_type] == "podcastEpisode":
                            theprogram = get_program_json(programs['_links']['podcastEpisode'])
                            seconds += isodate.parse_duration(theprogram['duration']).total_seconds()
                            first_prod_year = last_prod_year = dateutil.parser.isoparse(theprogram['publishedAt']).year
                            first_episode_id = theprogram['_links']['self']['href'].split('/')[-1]
                            all_episodes_ids.append(first_episode_id)
                            program_title = programs['podcastEpisode']['titles']['title']
                            num_episodes = 1
                            single = True
                        
                        elif programs[target_type] == "podcast":
                            thepodcast = get_podcast_json(programs['_links']['podcast'])

                            for episode in thepodcast['_embedded']['episodes']['items']:
                                year = dateutil.parser.isoparse(episode['publishedAt']).year
                                if year <= first_prod_year:
                                            first_prod_year = year
                                if year >= last_prod_year:
                                            last_prod_year = year
                                seconds += isodate.parse_duration(episode['duration']).total_seconds()
                                if first_episode_id == 0:
                                    first_episode_id = episode['_links']['podcastEpisode']['href'].split('/')[-1]
                                all_episodes_ids.append(first_episode_id)
                                num_episodes = len(thepodcast['_embedded']['episodes']['items'])

                            program_title = programs['podcast']['titles']['title']
                            num_episodes = programs['podcast']['numberOfEpisodes']
                            single = False

                        elif programs[target_type] == "channel":
                            print("Channel - skipping!")
                            continue

                        else:
                            print("Unknown - skipping!")
                            continue

                        if error == 0:
                            #Clean up variables
                            if len(series['included']['title']) <= 1:
                                cat_title = "Undefined"
                            else:
                                cat_title = series['included']['title']
                            

                            #print(f"{cat_title} -  {program_title} - {first_prod_year} - {last_prod_year} - {nr_episodes} - {first_episode_id} - array({len(all_episodes_ids)}) - {round(seconds)}")

                            row = {'category': cat_title,'title': program_title,'first_prod_year':first_prod_year,'last_prod_year':last_prod_year,'num_episodes':num_episodes,'first_episode_id':first_episode_id,'all_episodes_ids':str(all_episodes_ids),'total_seconds':round(seconds)}
                            #print(".", end = '') 
                            print('.', end='', flush=True)
                            #print(row)
                            
                            if single == True:
                                fsi_writer.writerows([row])       
                            else:
                                fse_writer.writerows([row])

                        else:
                            error = 0


    print("Finished writing csv output file to "+str(args.output_file))

def get_serie_json(seriename,medium):
    surl = "https://psapi.nrk.no/"+medium+"/catalog"+seriename

    r = requests.get(surl)
    if r.status_code != 200:
        raise Exception("Failed to load metadata from '%s'" % surl)

    return(json.loads(r.text))

def get_podcast_json(podcast):
    surl = "https://psapi.nrk.no/"+podcast

    r = requests.get(surl)
    if r.status_code != 200:
        raise Exception("Failed to load metadata from '%s'" % surl)

    return(json.loads(r.text))


def get_program_json(programpath):
    surl = "https://psapi.nrk.no"+programpath

    r = requests.get(surl)
    if r.status_code != 200:
        raise Exception("Failed to load metadata from '%s'" % surl)

    return(json.loads(r.text))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--medium', help="Medium is either radio or tv", required=True)
    parser.add_argument('--output_file', help="Complete path to csv output file. Note that it will split this in two names with _singles and _series by itself", required=True)


    args = parser.parse_args()
    main(args)








