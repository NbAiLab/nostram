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
import tarfile
import shutil

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
    parser.add_argument("-input_tarfile", "--input_tarfile", dest="input_tarfile", help="Source directory",
                        required=True)

    parser.add_argument("-master_savedir", "--master_savedir", dest="master_savedir", help="mastersave directory",
                        required=True)

    parser.add_argument("-device", "--device", dest="deviceid", help="Which device [0,1] (not required)",
                        required=False)

    options = parser.parse_args()
    return options

def buildtmpdir(pid, inputtarfile):
    savedir=str(pid)
    print(f"saving files to: {savedir}")
    makedirifnotexist(savedir)
    current_tar=tarfile.open(inputtarfile)

    current_tar.extractall(savedir)
    return savedir

if __name__ == "__main__":
    options = parse_args()
    pid=os.getpid()
    now = datetime.now()
    logdir="log"
    logfile=logdir + "/transcribe_"+ now.strftime("%Y%m%d-%H")+ ".log"
    makedirifnotexist(logdir)

    logfp=open(logfile, "a+")
    prefixoutputfilename = options.input_tarfile.split("/")[-1].split(".")[0]
    # modelname="NbAiLab/nb-wav2vec2-1b-bokmaal"
    modelname = "NbAiLab/nb-wav2vec2-300m-nynorsk"
    log("transcribe","Start",prefixoutputfilename)
    if options.deviceid is not None:
        if options.deviceid == "0":
            log("transcribe", "Start on device 0", prefixoutputfilename )
            pipe = pipeline(model= modelname, device=0, return_timestamps="word")
        elif options.deviceid == "1":
            log("transcribe", "Start on device 1", prefixoutputfilename )
            pipe = pipeline(model= modelname, device=1, return_timestamps="word")
        elif options.deviceid == "0,1":
            log("transcribe", "Start on device 0,1", prefixoutputfilename )
            pipe = pipeline(model= modelname, device="0,1", return_timestamps="word")
        else:
            log ("transcribe","Start", "Illegal device id specified")
            exit(-1)
    else:
        log("transcribe", "Start on cpu", prefixoutputfilename )
        pipe = pipeline(model= modelname, return_timestamps="word")

    #pipe = pipeline(model=modelname,return_timestamps="word")
    log("transcribe", "extracting files", prefixoutputfilename)
    tmpsavedir=buildtmpdir(pid, options.input_tarfile)

    log("transcribe", "finished extracting files", prefixoutputfilename)
    files = glob.glob(tmpsavedir +'/**/*.mp3', recursive=True)
    print("**************** Listing of files to transcribe")
    for f in files:
        print(f"file to be transcribed: {f}")
    print("**************** End listing of files to transcribe")
    output = pipe(files,chunk_length_s=10, stride_length_s=(2,2))


    outputdir=options.master_savedir+ "/"+ prefixoutputfilename
    makedirifnotexist(outputdir)
    #print(output)
    with jsonlines.open(outputdir+ "/" + prefixoutputfilename+"_output.jsonl", mode='w') as writer:
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

    shutil.rmtree(tmpsavedir)
    log("transcribe","end",prefixoutputfilename )
