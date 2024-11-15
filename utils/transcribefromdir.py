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
from tqdm import tqdm

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
    parser.add_argument("-input_dir", "--input_dir", dest="input_dir", help="Source dir",
                        required=True)
    parser.add_argument("-output_dir", "--output_dir", dest="output_dir", help="mastersave directory",
                        required=True)
    parser.add_argument("-device", "--device", dest="deviceid", help="Which device [0,1] (not required)",
                        required=False)
    parser.add_argument("-model", "--model", dest="model", help="which model to use", default="NbAiLab/nb-wav2vec2-1b-bokmaal",
                        required=False)
    parser.add_argument("-pattern", "--pattern", dest="pattern", help="filetype of files to process in input dir",
                        default="mp3",
                        required=False)
    parser.add_argument("-batchsize", "--batchsize", dest="batchsize", help="size of batch to process",type=int,
                        default=1000,
                        required=False)
    options = parser.parse_args()
    return options


if __name__ == "__main__":
    options = parse_args()
    pid=os.getpid()
    now = datetime.now()
    logdir="log"
    logfile=logdir + "/transcribe_"+ now.strftime("%Y%m%d-%H")+ ".log"
    makedirifnotexist(logdir)
    logfp=open(logfile, "a+")
    modelname=options.model
    log("transcribe","Start",options.input_dir)
    if options.deviceid is not None:
        if options.deviceid == "0":
            log("transcribe", "Start on device 0", options.input_dir )
            pipe = pipeline(model=modelname, device=0, return_timestamps="word")
        elif options.deviceid == "1":
            log("transcribe", "Start on device 1", options.input_dir )
            pipe = pipeline(model=modelname, device=1, return_timestamps="word")
        else:
            log ("transcribe","Start", "Illegal device id specified")
            exit(-1)
    else:
        log("transcribe", "Start on cpu", options.input_dir )
        pipe = pipeline(model=modelname, return_timestamps="word")

    files=glob.glob(options.input_dir+ "/*." + options.pattern)
    nofiles=len(files)
    print("*** Overall files to transcribe ***")

    outputdir = options.output_dir
    toprocesslist=[]
    for f in files:
        resultfile=outputdir + "/" +f.split("/")[-1].split(".")[0]+ ".jsonl"
        if not fileexists(resultfile):
            toprocesslist.append(f)

    files=toprocesslist
    #print(files)

    print("*** ")
    filechunks = [files[i:i + options.batchsize] for i in range(0, len(files), options.batchsize)]
    #print(filechunks)
    #print(len(filechunks))
    print(f"Dir {options.input_dir} Number of files in dir {nofiles} to process {len(files)} in  {len(filechunks)} chunks, batchsize {options.batchsize}")
    print("*** ")

    for cnt,filechunk in tqdm(enumerate(filechunks)):
        #output = pipe(files,chunk_length_s=25, stride_length_s=(2,2))
        output=""
        try:
            output = pipe(filechunk)
        except Exception as error:
            if options.batchsize != 1:
                errstr1=f"transcribe failed for {options.input_dir} batchsize {options.batchsize} batch number {cnt}"
                errstr2=f""
                log ("transcribe",str(error),errstr1)

            else:
                errstr1 = f"transcribe failed for {options.input_dir} file {filechunk[0]}"
                log("transcribe", str(error), errstr1)
            continue
        outputdir=options.output_dir
        makedirifnotexist(outputdir)
        for cnt, o in enumerate(output):
            currentfilename=f"{outputdir}/{filechunk[cnt].split('/')[-1].split('.')[0]}.jsonl"
            writer=jsonlines.open(currentfilename, mode='w')
            data = {}
            data["file"] = filechunk[cnt].split("/")[-1].split(".")[0]
            data["model"] = modelname
            data["text"] = o["text"]
            chunks = []
            for c in o["chunks"]:
                chunk = {}
                chunk["word"] = c["text"]
                chunk["timestamp"] = c["timestamp"]
                chunks.append(chunk)
            data["chunks"] = chunks
            writer.write(data)
            writer.close()

    log("transcribe","end",options.input_dir)
