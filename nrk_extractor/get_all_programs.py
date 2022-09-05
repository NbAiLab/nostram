import requests
import json
import re
import os
import subprocess
import time
import sys
import argparse
import isodate
def main(args):

    # Fetch the info blob
    murl = "https://psapi.nrk.no/tv/pages"
    r = requests.get(murl)
    if r.status_code != 200:
        raise Exception("Failed to load metadata from '%s'" % murl)
    
    res={}
    res["categories-tv"] = json.loads(r.text)
    
    # Loop through categories to find series and standalones
    count = 0
    for categories in res["categories-tv"]["pageListItems"]:
        murl = "https://psapi.nrk.no/"+categories["_links"]["self"]["href"]
        print(murl)
        r = requests.get(murl)
        if r.status_code != 200:
            raise Exception("Failed to load metadata from '%s'" % murl)
        res["series-tv"] = json.loads(r.text)
        
        #Loop through series and standalones
        
        catcount = 0
        for series in res["series-tv"]["sections"]:
            print(series['included']['title']+' = '+str(series["included"]["count"]))
            count = count+int(series["included"]["count"])
            catcount = catcount+ +int(series["included"]["count"])
            
            #For each of these check how long they are in total
            for programs in series["included"]["plugs"]:
                seconds = 0

                if programs['targetType'] == "series":
                    e=99

                elif programs['targetType'] == "episode":
                    e = 1
                    seconds = isodate.parse_duration(programs['episode']['duration']).total_seconds()
                
                elif programs['targetType'] == "standaloneProgram":
                    e = 1
                    seconds = isodate.parse_duration(programs['episode']['duration']).total_seconds()
               
                print(programs["displayContractContent"]["contentTitle"]+" - " + str(e) + " - " +str(round(seconds/3600,1)))

        #print('##SUM## ' + categories['id']+' = '+str(catcount)+ "\n")

    #print("\n\n##Total## "+" = "+str(count))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', help="id")

    args = parser.parse_args()
    main(args)








