
import json
import jsonlines
import re
import os

import time
import sys
import random
import glob

import math
import shutil
import time
from datetime import date,datetime

import copy
import numpy as np


from argparse import ArgumentParser
options=None

#"train:test:validate"
jsonratio="6:1:1"


def copysoundfiles(options):
    destinationdir=options.soundoutputdir
    #print(options.masterinputdir)
    directories = [d for d in os.listdir(options.masterinputdir+ "/") ]
    #print(directories)
    for d in directories:
        basedir= d.split("/")[-1]
        #print(basedir)
        #mkdirifnotexists(destinationdir+"/")
        mkdirifnotexists(destinationdir + "/audio_books")
        mkdirifnotexists(destinationdir + "/audio_books/"+basedir)
        files=[]
        if directoryexists(options.masterinputdir+ "/"+ d + "/mp3"):
            files= [f for f in os.listdir(options.masterinputdir+ "/"+ d + "/mp3")]


        if len(files) == 0:
            print("empty book:"+ options.masterinputdir+ "/"+ d)
        else:
            for f in files:
                print(options.masterinputdir+ "/"+ d+ "/mp3/" +str(f) + "::::"+ destinationdir+ "/audio_books/"+ basedir+ "/.")
                shutil.copy(options.masterinputdir+ "/"+ d+ "/mp3/" +f,destinationdir+ "/audio_books/"+ basedir+ "/"+ str(f))


def directoryexists(dir):
    if os.path.isdir(dir):
        return True
    else:
        return False

def fileexists(absfile):
    if os.path.isfile(absfile):
        return True
    else:
        return False

def mkdirifnotexists(dir):
    if directoryexists(dir) == False:
        os.mkdir(dir)
    return directoryexists(dir)

def parse_args():
    global options
    parser = ArgumentParser()


    parser.add_argument("-masteroutputdir", "--masteroutputdir", dest="masteroutputdir", help="masteroutputdir",
                        required=True)
    parser.add_argument("-masterinputdir", "--masterinputdir", dest="masterinputdir", help="masterinputdir",
                        required=True)
    parser.add_argument("-soundoutputdir", "--soundoutputdir", dest="soundoutputdir", help="soundoutputdir",
                        required=True)
    parser.add_argument("-distributionratio", "--distributionratio", dest="distributionratio", help="distributionratio",
                        required=False,default=jsonratio)

    parser.add_argument("-maxtestlines", "--maxtestlines", dest="maxtestlines", help="maxtestlines",
                        required=False, default=-1,type=int)
    parser.add_argument("-maxvalidationlines", "--maxvalidationlines", dest="maxvalidationlines", help="maxvalidationlines",
                        required=False,default=-1,type=int)

    options = parser.parse_args()
    return options

def validateitem(item):
    if item["audio_duration"] > 30000:
        return False
    if item["audio_duration"] < 1000:
        return False
    return True

def makefiles(options):
    distibutionratio=options.distributionratio
    trainparts=int(distibutionratio.split(":")[0])
    testparts=int(distibutionratio.split(":")[1])
    validationparts=int(distibutionratio.split(":")[2])
    nodistparts=trainparts+testparts+validationparts

    traindir= options.masteroutputdir+ "/train"
    testdir = options.masteroutputdir + "/test"
    validationdir = options.masteroutputdir + "/validation"
    mkdirifnotexists(options.masteroutputdir)
    mkdirifnotexists(traindir)
    mkdirifnotexists(testdir)
    mkdirifnotexists(validationdir)

    trainfilewriter=jsonlines.open(traindir+"/audio_books_train.json", mode='w')
    testfilewriter = jsonlines.open(testdir + "/audio_books_test.json", mode='w')
    validationfilewriter = jsonlines.open(validationdir + "/audio_books_validation.json", mode='w')

    jsonfilelist=glob.glob(options.masterinputdir+ "/**/*.json",recursive=True)
    jsoncounter=1
    jsonerror=0
    cntvalidationlines=0
    cnttestlines=0
    for jsonfile in jsonfilelist:
        print(jsonfile)
        jsongroup=jsonfile.split("/")[-1].split("_")[0]

        with open(jsonfile, 'r') as f:
            reader = jsonlines.Reader(f)
            cntno=0
            cntnn = 0
            cnten = 0
            cntnull = 0
            for line in reader.iter():
                item = line
                if item["text_language"] == "no":
                    cntno+=1
                elif item["text_language"] == "nn":
                    cntnn+=1
                elif item["text_language"] == "en":
                    cnten += 1
                elif item["text_language"] == null:
                    cntnull += 1
            lang=""
            max=max(cntno,cntnn,cnten,cntnull)
            if cntno == max:
                lang="no"
            elif cntnn == max:
                lang="nn"
            elif cnten == max:
                lang="en"
            else:
                lang=null

            for line in reader.iter():
                item=line
                if validateitem(item)== False:
                    jsonerror += 1
                else:
                    item["group_id"]=jsongroup
                    item["text_language"] = lang
                    if (jsoncounter % nodistparts) < trainparts:
                        print("writing train")
                        trainfilewriter.write(item)
                    elif (jsoncounter % nodistparts)  < (trainparts +testparts):
                        if options.maxtestlines != -1 and cnttestlines >= options.maxtestlines:
                            print("writing train")
                            trainfilewriter.write(item)
                        else:
                            print("writing test")
                            testfilewriter.write(item)
                            cnttestlines += 1
                    else:
                        if options.maxvalidationlines != -1 and cntvalidationlines >= options.maxvalidationlines:
                            print("writing train")
                            trainfilewriter.write(item)

                        else:
                            print("writing validation")
                            validationfilewriter.write(item)
                            cntvalidationlines += 1

                    jsoncounter+=1

    print("jsonok: " + str(jsoncounter))
    print("jsonerrors: "+ str(jsonerror))
    trainfilewriter.close()
    testfilewriter.close()
    validationfilewriter.close()


if __name__ == "__main__":
    # extractionlist=(("ff",10,20),("gg",30,40))
    # extractfromsoundfile("9789180515016_content.mp3","tmpmp3",extractionlist)
    # exit(-1)
    options = parse_args()
    copysoundfiles(options)
    makefiles(options)
    #copysoundfiles(options)