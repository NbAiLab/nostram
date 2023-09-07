
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
import numpy as np

from argparse import ArgumentParser

nameprefix = "NCC_S_no"


train_source=[]
validation_source=[]
test_source=[]
countersprsource={}

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

audiofiles={}


def buildaudiopaths(audiopathoption):
    global audiofiles
    audiofiles = {}
    audiopaths=audiopathoption.split(",")

    for p in audiopaths:
        print(p)

        currentfiles=glob.glob(p.strip() + "/**/*.mp3", recursive=True)
        for c in currentfiles:
            key=c.split("/")[-1].strip()
            value=c.strip()
            audiofiles[key]=value
            print(str(key)+ ":::" + str(value))


def findaudiofilepath(relativepath):
    global audiofiles
    relativepath=relativepath.strip()
    #print("searching for: "+ str(relativepath))
    if relativepath in audiofiles:
        #print(audiofiles[relativepath])
        return audiofiles[relativepath]

    print("Error: missing audio file for " + relativepath )
    exit(-1)


def checkjson(options,currentjson):
    print(currentjson)
    mp3file= findaudiofilepath(str(currentjson["id"])+ str(".mp3"))


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


def accumulate(options,split,currentjson):
    global train_source
    global validation_source
    global test_source

    if (split == "train"):
        train_source.append((currentjson["source"],currentjson["audio_duration"]))
    elif (split=="validation"):
        validation_source.append((currentjson["source"],currentjson["audio_duration"]))
    else:
        test_source.append((currentjson["source"],currentjson["audio_duration"]))

def histogram_sources(options):
    srclist=[]
    for i in train_source:
        if str(i[0]) not in srclist:
            srclist.append(i[0])
    for i in test_source:
        if str(i[0]) not in srclist:
            srclist.append(i[0])
    for i in validation_source:
        if str(i[0]) not in srclist:
            srclist.append(i[0])
    #print(srclist)

def countsources(options):
    global countersprsource
    for i in train_source:
        key=str(i[0]).replace(" "," ")
        if key in countersprsource:
            countersprsource[key]+=1
            countersprsource[key+"_duration"] += int(i[1])
        else:
            countersprsource[key] = 1
            countersprsource[key + "_duration"] = int(i[1])

def histogram_validation(options):
    validationduration=[]
    for i in validation_source:
        validationduration.append(int(i[1]/1000))
    validationdict = {}
    validationdict["Duration"]=validationduration
    #print(traindict)

    df = pd.DataFrame(validationdict)
    pd.options.display.float_format = '{:.2f}'.format
    # Calculate the histogram
    min_value = int(np.floor(df['Duration'].min()))
    max_value = int(np.ceil(df['Duration'].max()))
    bins = np.arange(min_value, max_value + 1,1) - 0.5

    hist, edges = np.histogram(df['Duration'], bins=bins)

    # Create the histogram in Markdown format
    markdown_histogram = "**Validation duration distribution**\n\n| Duration | Frequency |\n|--------|-----------|\n"

    for i in range(len(hist)):
        markdown_histogram += f"| {int(edges[i] + 0.5)} | {hist[i]} |\n"

    print(markdown_histogram)

    destination = options.destination + "/" + options.name
    fp = open(destination + "/stats.md", "a")

    fp.write(markdown_histogram)
    fp.close()

def histogram_test(options):
    testduration=[]
    for i in test_source:
        testduration.append(int(i[1]/1000))
    testdict = {}
    testdict["Duration"]=testduration
    #print(traindict)

    df = pd.DataFrame(testdict)
    pd.options.display.float_format = '{:.2f}'.format
    # Calculate the histogram
    min_value = int(np.floor(df['Duration'].min()))
    max_value = int(np.ceil(df['Duration'].max()))
    bins = np.arange(min_value, max_value + 1,1) - 0.5

    hist, edges = np.histogram(df['Duration'], bins=bins)

    # Create the histogram in Markdown format
    markdown_histogram = "**Test duration distribution**\n\n| Duration | Frequency |\n|--------|-----------|\n"

    for i in range(len(hist)):
        markdown_histogram += f"| {int(edges[i] + 0.5)} | {hist[i]} |\n"

    print(markdown_histogram)

    destination = options.destination + "/" + options.name
    fp = open(destination + "/stats.md", "a")
    #fp.write("**Test duration distribution**\n")
    fp.write(markdown_histogram)
    fp.close()


def histogram_train(options):
    trainduration=[]
    for i in train_source:
        trainduration.append(int(i[1]/1000))
    traindict = {}
    traindict["Duration"]=trainduration
    #print(traindict)

    df = pd.DataFrame(traindict)
    pd.options.display.float_format = '{:.2f}'.format
    # Calculate the histogram
    min_value = int(np.floor(df['Duration'].min()))
    max_value = int(np.ceil(df['Duration'].max()))
    bins = np.arange(min_value, max_value + 1,1) - 0.5

    hist, edges = np.histogram(df['Duration'], bins=bins)

    # Create the histogram in Markdown format
    markdown_histogram = "**Training duration distribution**\n\n| Duration | Frequency |\n|--------|-----------|\n"

    for i in range(len(hist)):
        markdown_histogram += f"| {int(edges[i] + 0.5)} | {hist[i]} |\n"

    print(markdown_histogram)

    destination = options.destination + "/" + options.name
    fp = open(destination + "/stats.md", "w+")
    #fp.write("**Training duration distribution**\n")
    fp.write(markdown_histogram)
    fp.close()


def buildtest(options):
    testjsonlines = []
    testjsonlinesfortar = []
    listfiles = glob.glob(options.src + "/test/*.json")
    linecnt=0
    for l in listfiles:
        with open(l) as f:
            for line in f:
                if int(options.maxtestlines) == -1:
                    testjsonlines.append(json.loads(line))
                elif linecnt < int(options.maxtestlines):
                    testjsonlines.append(json.loads(line))
                linecnt+=1

    print(f'Read {len(testjsonlines)} test json lines')

    random.shuffle(testjsonlines)
    for c in testjsonlines:
        testjsonlinesfortar.append(copy.deepcopy(c))

    destination = options.destination + "/" + options.name
    mkdirifnotexists(destination)
    mkdirifnotexists(destination + "/data")
    mkdirifnotexists(destination + "/data/test")
    removefilesfromdir(destination + "/data/test")
    cnt = 1
    # nameprefix="NCC_S_no"
    nopartitions =  1
    destinationtest = destination + "/data/test"
    jsoncnt = 0
    jsonname = ""

    for p in testjsonlines:
        s=copy.deepcopy(p)
        #p["audio"] = p["audio"].split("/")[-1]
        if ((jsoncnt == 0) ):
            jsonname = f'{destinationtest}/{options.name}-no-{cnt:04d}-{nopartitions:04d}.json'
            #outfp = open(jsonname, "w+", encoding='utf8')

            outfp = open(jsonname, "w+", encoding='utf8')
            cnt += 1
            jsoncnt = 0

        if (checkjson(options, s) == True):
            accumulate(options,"test", p)
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
            jsonname = f'{destinationtest}/{options.name}-no-{cnt:04d}-{nopartitions:04d}'
            currenttarfile = tarfile.open(jsonname + ".tar.gz", 'w:gz')
            cnt += 1
            jsoncnt = 0
        #shutil.copy("/home/freddy/PycharmProjects/dataset_sound/mini.mp3", options.src + "/audio/" + p["audio"])
        #print("writing")
        if (checkjson(options, p) == True):
            absfilename = findaudiofilepath(p["id"].strip()+".mp3")
            currenttarfile.add(absfilename, arcname=absfilename.split("/")[-1])

        jsoncnt += 1

    currenttarfile.close()
#######
def buildextravalidatedir(extramasterdir,extradir):
    print("Building extra validation:" + str(extradir))
    extravalidationjsonlines = []
    extravalidationjsonlinesfortar = []
    listfiles = glob.glob(extramasterdir + "/validation/" +extradir + "/*.json*")
    for l in listfiles:
        with open(l) as f:
            for line in f:
                extravalidationjsonlines.append(json.loads(line))

    print(f'Read {len(extravalidationjsonlines)} extra validation json lines from '+ str(extradir) )
    random.shuffle(extravalidationjsonlines)
    for c in extravalidationjsonlines:
        extravalidationjsonlinesfortar.append(copy.deepcopy(c))
    extradir="validation_"+extradir
    destination = options.destination + "/" + options.name
    mkdirifnotexists(destination)
    mkdirifnotexists(destination + "/data")
    mkdirifnotexists(destination + "/data/"+extradir)
    removefilesfromdir(destination + "/data/"+extradir)
    destinationdir=destination + "/data/"+extradir
    cnt = 1
    jsoncnt = 0
    # nameprefix="NCC_S_no"
    nopartitions = 1
    for p in extravalidationjsonlines:
        s = copy.deepcopy(p)
        #print(jsoncnt)
        #p["audio"] = p["audio"].split("/")[-1]
        if (jsoncnt == 0):
            jsonname = f'{destinationdir}/{options.name}-no-{cnt:04d}-{nopartitions:04d}.json'
            outfp = open(jsonname, "w+", encoding='utf8')
            cnt += 1
            jsoncnt = 0

        if (checkjson(options, s) == True):
            json.dump(p, outfp, ensure_ascii=False)
            outfp.write("\n")
        jsoncnt += 1
    outfp.close()

    jsoncnt = 0
    cnt = 1
    currenttarfile = None
    for p in extravalidationjsonlinesfortar:
        # print(p["audio"])
        if ((jsoncnt == 0)):
            if (jsoncnt != 0):
                # print("close")
                currenttarfile.close()
            jsonname = f'{destinationdir}/{options.name}-no-{cnt:04d}-{nopartitions:04d}'
            currenttarfile = tarfile.open(jsonname + ".tar.gz", 'w:gz')
            cnt += 1
            jsoncnt = 0
        if (checkjson(options, p) == True):
            absfilename = findaudiofilepath(p["id"].strip() + ".mp3")
            currenttarfile.add(absfilename, arcname=absfilename.split("/")[-1])

        jsoncnt += 1

    currenttarfile.close()


def buildextratestdir(extramasterdir,extradir):
    print("Building extra test:" + str(extradir))
    extratestjsonlines = []
    extratestjsonlinesfortar = []
    listfiles = glob.glob(extramasterdir + "/test/" +extradir + "/*.json*")
    for l in listfiles:
        with open(l) as f:
            for line in f:
                extratestjsonlines.append(json.loads(line))

    print(f'Read {len(extratestjsonlines)} extra test json lines from '+ str(extradir) )
    random.shuffle(extratestjsonlines)
    for c in extratestjsonlines:
        extratestjsonlinesfortar.append(copy.deepcopy(c))
    extradir = "test_" + extradir
    destination = options.destination + "/" + options.name
    mkdirifnotexists(destination)
    mkdirifnotexists(destination + "/data")
    mkdirifnotexists(destination + "/data/"+extradir)
    removefilesfromdir(destination + "/data/"+extradir)
    destinationdir=destination + "/data/"+extradir
    cnt = 1
    jsoncnt = 0
    # nameprefix="NCC_S_no"
    nopartitions = 1
    for p in extratestjsonlines:
        s = copy.deepcopy(p)
        #print(jsoncnt)
        #p["audio"] = p["audio"].split("/")[-1]
        if (jsoncnt == 0):
            jsonname = f'{destinationdir}/{options.name}-no-{cnt:04d}-{nopartitions:04d}.json'
            outfp = open(jsonname, "w+", encoding='utf8')
            cnt += 1
            jsoncnt = 0

        if (checkjson(options, s) == True):
            json.dump(p, outfp, ensure_ascii=False)
            outfp.write("\n")
        jsoncnt += 1
    outfp.close()

    jsoncnt = 0
    cnt = 1
    currenttarfile = None
    for p in extratestjsonlinesfortar:
        # print(p["audio"])
        if ((jsoncnt == 0)):
            if (jsoncnt != 0):
                # print("close")
                currenttarfile.close()
            jsonname = f'{destinationdir}/{options.name}-no-{cnt:04d}-{nopartitions:04d}'
            currenttarfile = tarfile.open(jsonname + ".tar.gz", 'w:gz')
            cnt += 1
            jsoncnt = 0
        if (checkjson(options, p) == True):
            absfilename = findaudiofilepath(p["id"].strip() + ".mp3")
            currenttarfile.add(absfilename, arcname=absfilename.split("/")[-1])

        jsoncnt += 1

    currenttarfile.close()


#######

def buildvalidate(options):
    validationjsonlines = []
    validationjsonlinesfortar = []
    listfiles = glob.glob(options.src + "/validation/*.json")
    linecnt=0
    for l in listfiles:
        with open(l) as f:
            for line in f:
                if int(options.maxvalidationlines) == -1:
                    validationjsonlines.append(json.loads(line))
                elif linecnt < int(options.maxvalidationlines):
                    validationjsonlines.append(json.loads(line))
                linecnt += 1


    print(f'Read {len(validationjsonlines)} validation json lines')
    random.shuffle(validationjsonlines)
    for c in validationjsonlines:
        validationjsonlinesfortar.append(copy.deepcopy(c))
    #partitionedtrainlist = partition(validationjsonlines, options.maxlines)
    destination = options.destination + "/" + options.name
    mkdirifnotexists(destination)
    mkdirifnotexists(destination + "/data")
    mkdirifnotexists(destination + "/data/validation")
    removefilesfromdir(destination + "/data/validation")
    cnt = 1
    # nameprefix="NCC_S_no"
    nopartitions =  1
    destinationvalidation = destination + "/data/validation"
    jsoncnt = 0
    jsonname = ""

    for p in validationjsonlines:
        s = copy.deepcopy(p)
        #print(jsoncnt)
        #p["audio"] = p["audio"].split("/")[-1]
        if (jsoncnt == 0):
            jsonname = f'{destinationvalidation}/{options.name}-no-{cnt:04d}-{nopartitions:04d}.json'
            outfp = open(jsonname, "w+", encoding='utf8')
            cnt += 1
            jsoncnt = 0

        if (checkjson(options, s) == True):
            accumulate(options,"validation", p)
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
            jsonname = f'{destinationvalidation}/{options.name}-no-{cnt:04d}-{nopartitions:04d}'
            currenttarfile = tarfile.open(jsonname + ".tar.gz", 'w:gz')
            cnt += 1
            jsoncnt = 0
        #shutil.copy("/home/freddy/PycharmProjects/dataset_sound/mini.mp3", options.src + "/audio/" + p["audio"])
        #print("writing")
        if (checkjson(options, p) == True):
            absfilename=findaudiofilepath(p["id"].strip()+".mp3")
            currenttarfile.add(absfilename, arcname=absfilename.split("/")[-1])


        jsoncnt += 1

    currenttarfile.close()

def checktrainjsonlines(options,globaljsonlines):
    validtrainjsonlines=[]
    for linespec in globaljsonlines:
        #print(linespec)
        if (checkjson(options, linespec) == True):
            validtrainjsonlines.append(linespec)

    return validtrainjsonlines


def buildtrain(options):
    globaljsonlines = []
    globaljsonlinesfortar = []
    listfiles = glob.glob(options.src + "/train/*.json")
    if options.maxtrainlines != -1:
        listfiles = listfiles[:100 * int(options.maxtrainlines) + 1]
    linecnt=0
    for l in listfiles:
        with open(l) as f:
            for line in f:
                globaljsonlines.append(json.loads(line))

    print(f'Read {len(globaljsonlines)} train json lines')
    verifiedtrainjsonlines = checktrainjsonlines(options,globaljsonlines)
    print(f'{len(verifiedtrainjsonlines)} verified train json lines')
    globaljsonlines =verifiedtrainjsonlines

    if options.maxtrainlines != -1:
        globaljsonlines =globaljsonlines[:256*int(options.maxtrainlines)+1]
    #print(f'Read {len(globaljsonlines)} train json lines')

    random.shuffle(globaljsonlines)
    for c in globaljsonlines:
        globaljsonlinesfortar.append(copy.deepcopy(c))


    #partitionedtrainlist = partition(globaljsonlines, options.maxlines)
    destination=options.destination + "/" +options.name
    mkdirifnotexists(destination)
    mkdirifnotexists(destination + "/data")
    mkdirifnotexists(destination + "/data/train")
    removefilesfromdir(destination + "/data/train")
    cnt = 1

    nopartitions = options.numberofsplits
    nolinesprpartition=math.floor(len(globaljsonlines) / nopartitions)
    print(str(nolinesprpartition) + " lines pr partition")
    destinationtrain = destination + "/data/train"
    jsoncnt = 0
    #jsonname = ""

    for p in globaljsonlines:
        audiopath=p["id"].strip()+".mp3"
        #p["audio"] = p["audio"].split("/")[-1]
        #del p["lang_voice_confidence"]
        if ((jsoncnt == 0) or (jsoncnt == nolinesprpartition)):
            if (jsoncnt != 0):
                outfp.close()
                currenttarfile.close()
            jsonname = f'{destinationtrain}/{options.name}-no-{cnt:04d}-{nopartitions:04d}.json'
            outfp = open(jsonname, "w+", encoding='utf8')

            jsonnametar = f'{destinationtrain}/{options.name}-no-{cnt:04d}-{nopartitions:04d}'
            currenttarfile = tarfile.open(jsonnametar + ".tar.gz", 'w:gz')
            cnt += 1
            jsoncnt = 0


        accumulate(options, "train", p)
        json.dump(p, outfp, ensure_ascii=False)
        outfp.write("\n")
        #filename = options.src + "/audio/" + audiopath
        absfilename=findaudiofilepath(audiopath)
        #print("Reference:" +str(audiopath))
        if (fileexists(absfilename) == False):
            print("str('Serious error: ')" + str(absfilename) + " does not exist,exiting!!!")
            exit(-1)
        currenttarfile.add(absfilename, arcname=absfilename.split("/")[-1])
        jsoncnt += 1

    outfp.close()
    currenttarfile.close()
    lastfileno=nopartitions+1
    lastfilename = f'{destinationtrain}/{options.name}-no-{lastfileno:04d}-{nopartitions:04d}.json'
    while fileexists(lastfilename) == True:
        os.remove(lastfilename)
        lastfileno = nopartitions + 1
        lastfilename = f'{destinationtrain}/{options.name}-no-{lastfileno:04d}-{nopartitions:04d}.json'

    lastfileno = nopartitions + 1
    lastfilename = f'{destinationtrain}/{options.name}-no-{lastfileno:04d}-{nopartitions:04d}.tar.gz'
    while fileexists(lastfilename) == True:
        os.remove(lastfilename)
        lastfileno += 1
        lastfilename = f'{destinationtrain}/{options.name}-no-{lastfileno:04d}-{nopartitions:04d}.tar.gz'

    return nopartitions

def generatedatasetgeneratorfile(options,numberofpartitions):

    print("generating dataloader")
    dest=options.destination+"/"+options.name
    loadercmd = './generatedataloader.sh' + " " + "template_dataloader.py" + " " + str(numberofpartitions) + " " +options.name + " " + dest
    process = subprocess.Popen(loadercmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    return

def generatereadmefile(options):
    global countersprsource
    print("generating readme")
    version=4
    nrkduration=0
    nrkcount=0
    nstduration = 0
    nstcount = 0
    npscduration = 0
    npsccount = 0
    for key, value in countersprsource.items():
        if key.startswith("NRK") and not key.endswith("tion"):
            nrkcount=int(value)
        elif key.startswith("NRK") and key.endswith("tion"):
            nrkduration=int(value)
        if key.startswith("NST") and not key.endswith("tion"):
            nstcount=int(value)
        elif key.startswith("NST") and key.endswith("tion"):
            nstduration=int(value)
        if key.startswith("NPSC") and not key.endswith("tion"):
            npsccount=int(value)
        elif key.startswith("NPSC") and key.endswith("tion"):
            npscduration=int(value)

    counterparameterstring=str(nrkcount) + " " + str(nrkduration) + " " + str(nstcount) + " " + str(nstduration) + " " + str(npsccount) + " " + str(npscduration)
    dest = options.destination + "/" + options.name + "/README.md"
    loadercmd = './buildreadme.sh' + " " + "template_readme.txt" + " " + str(options.name) + " " + str(version) + " " + str(counterparameterstring) + " " + dest
    process = subprocess.Popen(loadercmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    return

def copysomefiles(options):
    shutil.copy("desc.md",options.destination+"/"+options.name+ "/.")
    shutil.copy("README.md", options.destination +"/"+options.name + "/.")
    shutil.copy("description.md", options.destination + "/" + options.name + "/.")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-source", "--source", dest="src", help="Source directory with subtitle files",
                        required=True)

    parser.add_argument("-dest", "--destination", dest="destination", required=True)
    parser.add_argument("-name", "--name", dest="name", required=True)
    parser.add_argument("-audiopaths", "--audiopaths", dest="audiopaths", help="comma separated list of master audio paths", required=True)

    parser.add_argument("-maxlines", "--maxlines", dest="maxlines", help="max size of partition with subtitles",
                        default=-1, required=False)
    parser.add_argument("-numberofsplits", "--numberofsplits", dest="numberofsplits", help="train numberofsplits",
                        default=256, required=False,type=int)

    parser.add_argument("-split", "--buildsplit", dest="split", help="Which split to generate",
                        default="all", required=False)

    parser.add_argument("-maxtrainlines","--maxtrainlines",help="maxlines in train partition",default=-1,required=False)
    parser.add_argument("-maxvalidationlines", "--maxvalidationlines", help="maxlines in validate partition", default=-1,
                        required=False)
    parser.add_argument("-maxtestlines", "--maxtestlines", help="maxlines in test partition", default=-1,
                        required=False)

    parser.add_argument("-masterextradir", "--masterextradir", help="masterextradir", type=str, default="/mnt/lv_ai_1_ficino/ml/ncc_speech_corpus2/transcribed_json_4/",
                        required=False)

    parser.add_argument("-extratestdir", "--extratestdir", help="extratestdir", type=str,
                        default=None,
                        required=False)

    parser.add_argument("-extravalidationdir", "--extravalidationdir", help="extravalidationdir", type=str,
                        default=None,
                        required=False)

    options = parser.parse_args()
    return options



if __name__ == "__main__":
    #print(:s)
    #exit(-1)
    # buildaudiopaths("/nfsmounts/datastore/ncc_speech_corpus/transcribed_json_4/NPSC/audio/")
    # print(findaudiofilepath("2022/Stortinget-20220512-095500_9624600_9652300.mp3"))
    #

    options = parse_args()
    buildaudiopaths(options.audiopaths)

    now = datetime.now()
    logdir = options.src+"/log"
    mkdirifnotexists(logdir)
    logfile = logdir + "/datasetbuilder_" + now.strftime("%Y%m%d-%H") + ".log"
    logfp = open(logfile, "a+")

    if options.extratestdir != None:
        extratestdirstobuild = options.extratestdir.split(",")
        for extra in extratestdirstobuild:
            buildextratestdir(options.masterextradir, extra)
    if options.extravalidationdir != None:
        extravalidationdirstobuild = options.extravalidationdir.split(",")
        for extra in extravalidationdirstobuild:
            buildextravalidatedir(options.masterextradir, extra)
    exit(-1)

    if options.split.lower() == "all":
        nopartitions=buildtrain(options)
        buildvalidate(options)
        buildtest(options)
        copysomefiles(options)
        generatedatasetgeneratorfile(options,nopartitions)
        countsources(options)
        generatereadmefile(options)

        histogram_train(options)
        histogram_validation(options)
        histogram_test(options)

    elif options.split.lower() == "train":

        nopartitions = buildtrain(options)
        histogram_train(options)
    elif options.split.lower() == "validation":

        buildvalidate(options)
        histogram_validation(options)


    else:
        buildtest(options)
        histogram_test(options)



    if options.extratestdir != None:
        extratestdirstobuild= options.extratestdir.split(",")
        for extra in extratestdirstobuild:
            buildextratestdir(options.masterextradir,extra)
    if options.extravalidationdir != None:
        extravalidationdirstobuild= options.extravalidationdir.split(",")
        for extra in extravalidationdirstobuild:
            buildextravalidatedir(options.masterextradir,extra)

    logfp.close()
    #histogram_sources(options)
    #histogram_train(options)

    #histogram_validation(options)
    #histogram_test(options)





