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

    firstepisode=""
    currentepisode=""


    def geturl(self,url):
        #maxreq = 4
        #cntreq = 0
        searchstr = url
        resp = requests.get(searchstr)
        return resp


    def __init__(self, ):
        self.firstepisode="firstepisode"
        self.currentepisode="firstepisode"

    def getnextepisode(self):
        print("bla")

    def getcurrentepisodeinfo(self):
        print("bla")

    def getseries(self,anepisodeidofsomething):

        programinforeq="https://psapi.nrk.no/playback/metadata/program/"+anepisodeidofsomething
        resp=self.geturl(programinforeq)
        resultingjson=resp.json()
        #print(resultingjson["_links"]["series"]["href"])
        return resultingjson["_links"]["series"]["href"].split("/")[-1]

    def getmetadataforseries(self,serie):
        programinforeq = "https://psapi.nrk.no/series/" + serie
        resp = self.geturl(programinforeq)
        resultingjson = resp.json()
        #print(resultingjson["seasons"][0]["name"])
        return int(resultingjson["seasons"][0]["name"]) #vi bør kanskje sortere her


    def getprograms(self,title,seasonsnr):
        programinforeq = "https://psapi.nrk.no/tv/catalog/series/kraakeklubben/seasons/1"
        resp = self.geturl(programinforeq)
        resultingjson = resp.json()["_embedded"]["episodes"]
        resultlist=[]
        for ind,i in enumerate(resultingjson):
            id=resultingjson[ind]["_links"]["self"]["href"].split("/")[-1]
            #print(resultingjson[ind]["_links"]["self"]["href"])
            resultlist.append(id)
        return resultlist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputids', help='list of id's'')

    args = parser.parse_args()

    ef=episodefetcher()
    inputlist=args.inputids
    for i in inputlist.split(","):
        currentserie=ef.getseries(i)
        noseasons=ef.getmetadataforseries(currentserie)
        for i in range(1,noseasons+1):
            result=ef.getprograms(currentserie, i)
            for r in result:
                print("serie" +currentserie + " sesong: " + str(i) + " id: " + str(r))

    #ef.getseries("DNPR63700111")
    #ef.getmetadataforseries("kraakeklubben")
    #ef.getprograms("kraakeklubben",1)