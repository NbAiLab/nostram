
import json
import re
import os
import subprocess
import time
import sys
import random
import glob
import tarfile
import gzip
import math
import shutil
import time
from datetime import date,datetime
import pandas as pd
import copy
import librosa

from argparse import ArgumentParser

nameprefix = "NCC_S_no"

accumulations=[0.0] * 15
sourcelist=[]
sourceaccumulators= [0.0] * 1000


logfp=None


def calculate_duration(mp3):
    try:
        duration = int(librosa.get_duration(filename=mp3) * 1000)
    except:
        return 0
    return duration

def log(message):
    now = datetime.now()
    mystrdate=now.strftime("%Y/%m/%d %H:%M:%S")
    ostring=mystrdate + "\t"  + str(message) + "\n"
    #print(ostring)
    logfp.write(ostring)
    logfp.flush()


def partition(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i+size]

def checkfilesizelargerthan(absfile,sizelimit):
    sz=os.path.getsize(absfile)
    #print("----->"+str(sz))
    if sz >= sizelimit:
        return True
    else:
        return False

def directoryexists(dir):
    if os.path.isdir(dir):
        return True
    else:
        return False

def countfiles(dir,ext=None):
    if (ext == None):
        return len(os.listdir(dir))
    else:
        return len(glob.glob(dir+"/*." + ext))

def fileexists(absfile):
    if os.path.isfile(absfile):
        return True
    else:
        return False

def mkdirifnotexists(dir):
    if directoryexists(dir) == False:
        os.mkdir(dir)
    return directoryexists(dir)

def removefilesfromdir(dir):
    files = glob.glob(dir+"/*")
    for f in files:
        try:
           os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

def checkjson(options,currentjson):
    mp3file=options.src + "/audio/" + currentjson["audio"]
    if fileexists(mp3file) == False:
        message=f'mp3file {mp3file} does not exist for id {currentjson["id"]}'
        print(message)
        log(message)
        return False
    if checkfilesizelargerthan(mp3file,500) == False:
        message = f'mp3file {mp3file} has size less than 500 for id {currentjson["id"]}'
        print(message)
        log(message)
        return False


    return True

def accumulateprsource(currentjson):
    currentsource=currentjson["source"]
    currentstartindex=-1
    if currentsource in sourcelist:
        currentstartindex=sourcelist.index(currentsource) * 5
    else:
        sourcelist.append(currentsource)
        currentstartindex=sourcelist.index(currentsource) * 5

    sourceaccumulators[currentstartindex] += 1
    sourceaccumulators[currentstartindex+1]+=float(currentjson["duration"])
    sourceaccumulators[currentstartindex+2]+=len(currentjson["text"])
    if (sourceaccumulators[currentstartindex+3] ==0 or float(currentjson["duration"]) < sourceaccumulators[currentstartindex+3]):
         sourceaccumulators[currentstartindex+3] = float(currentjson["duration"])
    if  float(currentjson["duration"]) > sourceaccumulators[currentstartindex+4]:
         sourceaccumulators[currentstartindex+4]=  float(currentjson["duration"])


def accumulate(split,currentjson):

    accumulateprsource(currentjson)
    #print(split)
    #print(currentjson)
    if split == "train":
        startindex=0
    elif split=="validation":
        startindex=5
    else:
        startindex=10

    accumulations[startindex]+=1
    accumulations[startindex+1]+=float(currentjson["duration"])
    accumulations[startindex+2]+=len(currentjson["text"])
    if (accumulations[startindex+3] ==0 or float(currentjson["duration"]) < accumulations[startindex+3]):
         accumulations[startindex+3] = float(currentjson["duration"])
    if  float(currentjson["duration"]) > accumulations[startindex+4]:
         accumulations[startindex+4]=  float(currentjson["duration"])

def printaccumulators(options):
    print("generating stats.md")
    data = {'Split': ["Train","Validation", "Test"],
            'Counter': [accumulations[0], accumulations[5],accumulations[10]],
            'Sum duration': [accumulations[1], accumulations[6],accumulations[11]],
            'Sum words':[accumulations[2], accumulations[7],accumulations[12]],
            'Min duration': [accumulations[3], accumulations[8], accumulations[13]],
            'Max duration': [accumulations[4], accumulations[9], accumulations[14]],

            }
    df = pd.DataFrame(data)
    #print(data)
    pd.set_option("display.float_format", "{:.2f}".format)
    # Convert the dataframe to markdown format
    markdown = df.to_markdown(index=False,tablefmt="grid")
    fp = open(options.destination + "/stats.md", "w+")
    fp.write(markdown)
    fp.close()

    cnt=0
    data={'Source':sourcelist}
    counterlist=[]
    sumlist=[]
    sumwordslist=[]
    summaxlist=[]
    summinlist=[]
    for s in sourcelist:
        counterlist.append(str(sourceaccumulators[cnt]))
        sumlist.append(str(sourceaccumulators[cnt+1]))
        sumwordslist.append(str(sourceaccumulators[cnt+2]))
        summinlist.append(str(sourceaccumulators[cnt+3]))
        summaxlist.append(str(sourceaccumulators[cnt+4]))
        cnt+=5


    data["Counter"]=counterlist
    data["Sum duration"] =sumlist
    data["Sum words"] = sumwordslist
    data["Min duration"] = summinlist
    data["Max duration"] = summaxlist

    df = pd.DataFrame(data)
    pd.set_option("display.float_format", "{:.2f}".format)
    markdown = df.to_markdown(index=False, tablefmt="grid")
    fp = open(options.destination + "/stats.md", "a")
    fp.write("\n")
    fp.write(markdown)
    fp.close()

    # print(accumulations)
    # print(sourcelist)
    # print(sourceaccumulators)

def buildtest(options):
    testjsonlines = []
    testjsonlinesfortar = []
    listfiles = glob.glob(options.src + "/test/*.json")
    for l in listfiles:
        with open(l) as f:
            for line in f:
                testjsonlines.append(json.loads(line))


    print(f'Read {len(testjsonlines)} test json lines')

    random.shuffle(testjsonlines)
    for c in testjsonlines:
        testjsonlinesfortar.append(copy.deepcopy(c))
    mkdirifnotexists(options.destination)
    mkdirifnotexists(options.destination + "/data")
    mkdirifnotexists(options.destination + "/data/test")
    removefilesfromdir(options.destination + "/data/test")
    cnt = 1
    # nameprefix="NCC_S_no"
    nopartitions =  1
    destinationtest = options.destination + "/data/test"
    jsoncnt = 0
    jsonname = ""

    for p in testjsonlines:
        s=copy.deepcopy(p)
        p["audio"] = p["audio"].split("/")[-1]
        if ((jsoncnt == 0) ):
            if (jsoncnt != 0):
                outfp.close()
            jsonname = f'{destinationtest}/{nameprefix}-{cnt:04d}-{nopartitions:04d}.json'
            #outfp = open(jsonname, "w+", encoding='utf8')

            outfp = open(jsonname, "w+", encoding='utf8')
            cnt += 1
            jsoncnt = 0

        accumulate("test", p)

        if (checkjson(options, s) == True):
            json.dump(p,outfp,ensure_ascii=False)
            outfp.write("\n")
        jsoncnt += 1
    outfp.close()
    jsoncnt = 0
    cnt = 1
    currenttarfile = None
    for p in testjsonlinesfortar:
        #print(p["audio"])
        if ((jsoncnt == 0) ):
            if (jsoncnt != 0):
                #print("close")
                currenttarfile.close()
            jsonname = f'{destinationtest}/{nameprefix}-{cnt:04d}-{nopartitions:04d}'
            currenttarfile = tarfile.open(jsonname + ".tar.gz", 'w:gz')
            cnt += 1
            jsoncnt = 0
        #shutil.copy("/home/freddy/PycharmProjects/dataset_sound/mini.mp3", options.src + "/audio/" + p["audio"])
        #print("writing")
        if (checkjson(options, p) == True):
            filename=options.src + "/audio/" + p["audio"]
            currenttarfile.add(filename,arcname=filename.split("/")[-1])
        jsoncnt += 1

    currenttarfile.close()


def buildvalidate(options):
    validationjsonlines = []
    validationjsonlinesfortar = []
    listfiles = glob.glob(options.src + "/validation/*.json")
    for l in listfiles:
        with open(l) as f:
            for line in f:
                validationjsonlines.append(json.loads(line))


    print(f'Read {len(validationjsonlines)} validation json lines')
    random.shuffle(validationjsonlines)
    for c in validationjsonlines:
        validationjsonlinesfortar.append(copy.deepcopy(c))
    #partitionedtrainlist = partition(validationjsonlines, options.maxlines)
    mkdirifnotexists(options.destination)
    mkdirifnotexists(options.destination + "/data")
    mkdirifnotexists(options.destination + "/data/validation")
    removefilesfromdir(options.destination + "/data/validation")
    cnt = 1
    # nameprefix="NCC_S_no"
    nopartitions =  1
    destinationvalidation = options.destination + "/data/validation"
    jsoncnt = 0
    jsonname = ""

    for p in validationjsonlines:
        s = copy.deepcopy(p)
        p["audio"] = p["audio"].split("/")[-1]
        if ((jsoncnt == 0) ):
            if (jsoncnt != 0):
                outfp.close()
            jsonname = f'{destinationvalidation}/{nameprefix}-{cnt:04d}-{nopartitions:04d}.json'
            outfp = open(jsonname, "w+", encoding='utf8')
            cnt += 1
            jsoncnt = 0
        accumulate("validation", p)
        if (checkjson(options, s) == True):
            json.dump(p, outfp, ensure_ascii=False)
            outfp.write("\n")
        jsoncnt += 1
    outfp.close()
    jsoncnt = 0
    cnt = 1
    currenttarfile = None
    for p in validationjsonlinesfortar:
        #print(p["audio"])
        if ((jsoncnt == 0) ):
            if (jsoncnt != 0):
                #print("close")
                currenttarfile.close()
            jsonname = f'{destinationvalidation}/{nameprefix}-{cnt:04d}-{nopartitions:04d}'
            currenttarfile = tarfile.open(jsonname + ".tar.gz", 'w:gz')
            cnt += 1
            jsoncnt = 0
        #shutil.copy("/home/freddy/PycharmProjects/dataset_sound/mini.mp3", options.src + "/audio/" + p["audio"])
        #print("writing")
        if (checkjson(options, p) == True):
            filename = options.src + "/audio/" + p["audio"]
            currenttarfile.add(filename, arcname=filename.split("/")[-1])
        jsoncnt += 1

    currenttarfile.close()


def buildtrain(options):
    globaljsonlines = []
    globaljsonlinesfortar = []
    listfiles = glob.glob(options.src + "/train/*.json")
    for l in listfiles:
        with open(l) as f:
            for line in f:
                globaljsonlines.append(json.loads(line))


    print(f'Read {len(globaljsonlines)} train json lines')
    random.shuffle(globaljsonlines)
    for c in globaljsonlines:
        globaljsonlinesfortar.append(copy.deepcopy(c))

    #partitionedtrainlist = partition(globaljsonlines, options.maxlines)
    mkdirifnotexists(options.destination)
    mkdirifnotexists(options.destination + "/data")
    mkdirifnotexists(options.destination + "/data/train")
    removefilesfromdir(options.destination + "/data/train")
    cnt = 1
    # nameprefix="NCC_S_no"
    nopartitions = math.ceil(len(globaljsonlines) / int(options.maxlines))
    destinationtrain = options.destination + "/data/train"
    jsoncnt = 0
    jsonname = ""
    for p in globaljsonlines:
        #audioname=p["audio"].split("/")[-1]
        s = copy.deepcopy(p)
        p["audio"] = p["audio"].split("/")[-1]
        if ((jsoncnt == 0) or (jsoncnt == int(options.maxlines))):
            if (jsoncnt != 0):
                outfp.close()
            jsonname = f'{destinationtrain}/{nameprefix}-{cnt:04d}-{nopartitions:04d}.json'
            outfp = open(jsonname, "w+", encoding='utf8')
            cnt += 1
            jsoncnt = 0

        accumulate("train", p)
        if (checkjson(options, s) == True):
            json.dump(p, outfp, ensure_ascii=False)
            outfp.write("\n")
        outfp.write("\n")
        jsoncnt += 1
    outfp.close()
    jsoncnt = 0
    cnt = 1
    currenttarfile = None
    for p in globaljsonlinesfortar:
        #print(p["audio"])
        if ((jsoncnt == 0) or (jsoncnt == int(options.maxlines))):
            if (jsoncnt != 0):
                #print("close")
                currenttarfile.close()
            jsonname = f'{destinationtrain}/{nameprefix}-{cnt:04d}-{nopartitions:04d}'
            currenttarfile = tarfile.open(jsonname + ".tar.gz", 'w:gz')
            cnt += 1
            jsoncnt = 0
        #shutil.copy("/home/freddy/PycharmProjects/dataset_sound/mini.mp3", options.src + "/audio/" + p["audio"])
     #   print("writing")
        if (checkjson(options, p) == True):
            filename = options.src + "/audio/" + p["audio"]
            currenttarfile.add(filename, arcname=filename.split("/")[-1])

        jsoncnt += 1

    currenttarfile.close()
    return nopartitions

def generatedatasetgeneratorfile(options,numberofpartitions):

    print("generating dataloader")
    loadercmd = './generatedataloader.sh' + " " + "template_dataloader.py" + " " + str(numberofpartitions) + " " + options.destination
    process = subprocess.Popen(loadercmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    return

def copysomefiles(options):
    shutil.copy("desc.md",options.destination+ "/.")
    shutil.copy("README.md", options.destination + "/.")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-sub", "--source_subtitledir", dest="src", help="Source directory with subtitle files",
                        required=True)

    parser.add_argument("-dest", "--destination", dest="destination", required=True)
    parser.add_argument("-maxlines", "--maxlines", dest="maxlines", help="max size of partition with subtitles",
                        default=10000, required=False)

    parser.add_argument("-split", "--buildsplit", dest="split", help="Which split to generate",
                        default="all", required=False)

    options = parser.parse_args()
    return options



if __name__ == "__main__":
    #print(accumulations)
    #exit(-1)

    options = parse_args()
    now = datetime.now()
    logdir = options.src+"/log"
    mkdirifnotexists(logdir)
    logfile = logdir + "/datasetbuilder_" + now.strftime("%Y%m%d-%H") + ".log"
    logfp = open(logfile, "a+")
    if options.split.lower() == "all":
        nopartitions=buildtrain(options)
        buildvalidate(options)
        buildtest(options)
        copysomefiles(options)
        generatedatasetgeneratorfile(options,nopartitions)
        printaccumulators(options)
        logfp.close()
    elif options.split.lower() == "train":
        nopartitions = buildtrain(options)
    elif options.split.lower() == "validation":
        buildvalidate(options)
    else:
        buildtest(options)







