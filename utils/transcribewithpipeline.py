from transformers import pipeline
import glob
import jsonlines
import sys
import os
import numpy as np
import glob
import argparse
import requests
import time
import datetime
from datetime import date,datetime
from argparse import ArgumentParser

outputbasedir=""
#outputbasedir="/home/freddy/PycharmProjects/wav2vectranscription"

def fileexists(absfile):
    if os.path.isfile(absfile):
        return True
    else:
        return False

def makedirifnotexist(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def directoryexists(dir):
    if os.path.isdir(dir):
        return True
    else:
        return False

def log(oper,objekt,message):
    now = datetime.now()
    mystrdate=now.strftime("%Y/%m/%d %H:%M:%S")
    ostring=mystrdate + "\t\t" + str(oper) + "\t\t" + str(objekt) + "\t\t" + str(message) + "\n"
    print(ostring)
    logfp.write(ostring)
    logfp.flush()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-sourcename", "--source_name", dest="sourcename", help="Source directory with subtitle files",
                        required=True)
    parser.add_argument("-name", "--destination_name", dest="name", help="Source directory with subtitle files",
                        required=True)

    parser.add_argument("-device", "--deviceid", dest="deviceid", help="Source directory with subtitle files",
                        required=True)

    options = parser.parse_args()
    return options



if __name__ == "__main__":
    options = parse_args()
    outputdir=options.name
    basename=options.name.split("/")[-1]

    now = datetime.now()
    logdir="log"
    logfile=logdir + "/transcribe_"+ now.strftime("%Y%m%d-%H")+ ".log"
    makedirifnotexist(logdir)
    makedirifnotexist(outputdir)
    logfp=open(logfile, "a+")

    modelname="NbAiLab/nb-wav2vec2-1b-bokmaal"
    log("transcribe","Start",basename)
    if options.deviceid == "0":
        log("transcribe", "Start on device 0", basename)
        pipe = pipeline(model="NbAiLab/nb-wav2vec2-1b-bokmaal",device=0,return_timestamps="word")
    elif options.deviceid == "1":
        log("transcribe", "Start on device 1", basename)
        pipe = pipeline(model="NbAiLab/nb-wav2vec2-1b-bokmaal", device=1, return_timestamps="word")
    else:
        log("transcribe", "Start on cpu", basename)
        pipe = pipeline(model="NbAiLab/nb-wav2vec2-1b-bokmaal",  return_timestamps="word")
    #pipe = pipeline(model=modelname,return_timestamps="word")
    files = glob.glob(options.sourcename +'/*.mp3')
    output = pipe(files,chunk_length_s=10, stride_length_s=(2,2))

    #print(output)
    with jsonlines.open(outputdir+ "/" + "output.jsonl", mode='w') as writer:
        for cnt,o in enumerate(output):
            data={}
            data["file"]=files[cnt].split("/")[-1].split(".")[0]
            data["model"] = modelname
            data["text"]=o["text"]
            chunks=[]
            for c in o["chunks"]:
                chunk={}
                chunk["word"]=c["text"]
                chunk["timestamp"]=c["timestamp"]
                chunks.append(chunk)
            data["chunks"]=chunks

            #print(data)
            writer.write(data)


    log("transcribe","end",basename)
