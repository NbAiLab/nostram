import sys
import os
#import numpy as np
import glob
import argparse
import requests
import time
from enum import Enum
import io
import json
import random
from random import randint
import shutil
from datetime import date,datetime

from PIL import Image
import requests
from joblib import Parallel, delayed


class episodefetcher:
    episodelist=[]


    def geturl(self,url):
        #maxreq = 4
        #cntreq = 0
        searchstr = url
        resp = requests.get(searchstr)
        if (resp.status_code !=200):
            print("something went wrong with url: "+ searchstr + " returned status code" + str(resp.status_code))

        return resp


    def __init__(self, ):
        self.episodelist=[]


    def getcurrentepisodeinfo(self):
        print("bla")
    def isseries(self,anepisodeidofsomething):
        programinforeq = "https://psapi.nrk.no/playback/metadata/program/" + anepisodeidofsomething
        resp = self.geturl(programinforeq)
        resultingjson = resp.json()
        if "series" not in resultingjson["_links"]:
            return False
        else:
            return True

    def getseries(self,anepisodeidofsomething):
        programinforeq="https://psapi.nrk.no/playback/metadata/program/"+anepisodeidofsomething
        resp=self.geturl(programinforeq)
        resultingjson=resp.json()
        #print(resp.json())
        #print(resultingjson["_links"]["series"]["href"])
        if "series" not in resultingjson["_links"]:
            return resultingjson["_links"]["self"]["href"].split("/")[-1]
        else:
            return resultingjson["_links"]["series"]["href"].split("/")[-1]

    def getmetadataforseries(self,serie):
        programinforeq = "https://psapi.nrk.no/series/" + serie
        resp = self.geturl(programinforeq)
        resultingjson = resp.json()
        
        resultlist=[]
        for ind,i in enumerate(resultingjson["seasons"]):
            resultlist.append(resultingjson["seasons"][ind]["name"])
        return resultlist

    def getmedium(self,serie):
        programinforeq = "https://psapi.nrk.no/series/" + serie
        resp = self.geturl(programinforeq)
        resultingjson = resp.json()
        
        href = resultingjson["_links"]["share"]["href"]
        if "radio.nrk.no" in href:
            medium = "radio"
        elif "tv.nrk.no" in href:
            medium = "tv"
        else:
            medium = "undefined"

        return medium
    
    def getserieimage(self,serie):
        programinforeq = "https://psapi.nrk.no/series/" + serie
        resp = self.geturl(programinforeq)
        resultingjson = resp.json()
        serieimage = resultingjson["image"]["webImages"][0]["imageUrl"]
        
        return serieimage

    def getprograms(self,medium,title,seasonsnr):
        programinforeq = "https://psapi.nrk.no/"+str(medium)+"/catalog/series/"+ str(title) + "/seasons/" + str(seasonsnr)
        resp = self.geturl(programinforeq)
        resultingjson =[]
        #print(medium + " "+ title + " " + seasonsnr)
        # print(resp.json())
        if resp.status_code != 200:
            return []

        if "episodes" in resp.json()["_embedded"]:
            resultingjson = resp.json()["_embedded"]["episodes"]
        
        else:
            resultingjson = resp.json()["_embedded"]["instalments"]
        
        resultlist=[]

        if not isinstance(resultingjson, list):
            for ind,i in enumerate(resultingjson["_embedded"]["episodes"]):
                id=resultingjson["_embedded"]["episodes"][ind]["_links"]["self"]["href"].split("/")[-1]
                resultlist.append(id) 
        else:
            for ind,i in enumerate(resultingjson):
                id=resultingjson[ind]["_links"]["self"]["href"].split("/")[-1]
                resultlist.append(id)
        

        return resultlist

    def episodebuilder(self,inputids):
        self.episodelist=[]
        inputlist=inputids.split(",")
        for programidentifier in inputlist:
            if self.isseries(programidentifier)== False:
                self.episodelist.append(programidentifier)
            else:
                currentserie = self.getseries(programidentifier)
                theseasons = self.getmetadataforseries(currentserie)

                #Figure out if this is radio or tv
                medium = self.getmedium(currentserie)

                for season in theseasons:
                    programlist = self.getprograms(medium,currentserie, season)
                    for program in programlist:
                        self.episodelist.append(program)
                        #print("serie" + currentserie + " sesong: " + str(i) + " id: " + str(r))

    def episodegenerator(self):
        for episode in self.episodelist:
            yield episode

    def getepisodemetadata(self,episode):
        data = {}
        programinforeq = "https://psapi.nrk.no/playback/metadata/program/" + episode
        resp = self.geturl(programinforeq)
        resultingjsonprogram = resp.json()

        currentserie=self.getseries(episode)
        seriesinforeq=""

        data["program_id"]=episode
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputids', help="list of id's")

    args = parser.parse_args()
    ef=episodefetcher()
    inputlist=args.inputids
    ef.episodebuilder(inputlist)
    for i in ef.episodegenerator():
        print(i)






