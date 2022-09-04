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
        r = requests.get(murl)
        if r.status_code != 200:
            raise Exception("Failed to load metadata from '%s'" % murl)
        res["series-tv"] = json.loads(r.text)
        
        #Loop through series and standalones
        
        catcount = 0
        for series in res["series-tv"]["sections"]:
            #print(series['included']['title']+' = '+str(series["included"]["count"]))
            count = count+int(series["included"]["count"])
            catcount = catcount+ +int(series["included"]["count"])
            
            #For each of these loop through programs to find out how long they are
            for programs in series["included"]["plugs"]:
                print(programs["targetType"])
                #isodate.parse_duration("PT4M13S").total_seconds()
        
        #print('##SUM## ' + categories['id']+' = '+str(catcount)+ "\n")

    #print("\n\n##Total## "+" = "+str(count))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', help="id")

    args = parser.parse_args()
    main(args)








