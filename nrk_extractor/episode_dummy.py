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
        resultlist=[]
        for ind,i in enumerate(resultingjson["seasons"]):
            resultlist.append(resultingjson["seasons"][ind]["name"])
        return resultlist



    def getprograms(self,title,seasonsnr):
        print(title)
        programinforeq = "https://psapi.nrk.no/tv/catalog/series/"+ str(title) + "/seasons/" + str(seasonsnr)
        resp = self.geturl(programinforeq)
        print(programinforeq)
        #exit(-1)
        #print(resp)
        #print(resp.json())
        resultingjson =[]
        if "episodes" in resp.json()["_embedded"]:
            resultingjson = resp.json()["_embedded"]["episodes"]
        else:
            resultingjson = resp.json()["_embedded"]["instalments"]

        resultlist=[]
        for ind,i in enumerate(resultingjson):
            id=resultingjson[ind]["_links"]["self"]["href"].split("/")[-1]
            #print(resultingjson[ind]["_links"]["self"]["href"])
            resultlist.append(id)
        return resultlist

    def get_all_programs(self,id):
        return ['MSUB22000113','MSUB22000215']

    def get_metadata(self,id):
        return {'id': 'kraakeklubben','title': 'Kråkeklubben'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputids', help="list of id's")

    args = parser.parse_args()

    ef=episodefetcher()
    inputlist=args.inputids
    for i in inputlist.split(","):
        currentserie=ef.getseries(i)
        noseasons=ef.getmetadataforseries(currentserie)
        for i in noseasons:
            result=ef.getprograms(currentserie, i)
            for r in result:
                print("serie" +currentserie + " sesong: " + str(i) + " id: " + str(r))

    #ef.getseries("DNPR63700111")
    #ef.getmetadataforseries("kraakeklubben")
    #ef.getprograms("kraakeklubben",1)

    # MSUB22000113
    # FBUA06000075
    # MSUS01004710
    # FBUA03003087
    # MSUB20002611
    # FBUB04000100
    # FBUA03001389
    # OBUB12000108
    # FSTL01000188
    # MKTV13100320
    # MSUE13000118
    # OBUB07000104
    # OBUS01000103
    # OBUB07000408
    # FBUA03002588
    # MSUS24000120
    # MSUB02000110
    # FBUA01007383
    # MSUS05001110
    # FALB60000192
    # FBUA03000179
    # DMND10005013
    # FBUA03001388
    # MSUB19120116
    # FALU07000191
    # MSUI40005120
    # DMYT24002818
    # DNPR63700110
    # DNPR63000116




