import sys
import os
import math
import numpy as np
import glob
import argparse
import subprocess
import string

from os import path
import unicodedata
from string import printable
import ftfy
import jsonlines
import json
import logging
import re
import nltk.data

from pathlib import Path
from utils.matching import Matcher, load_segments
import re
from Levenshtein import distance


splittimelimit=1.2


localcorpus=[]
words=[]
origwords=[]
wordpos=[]
v2wpos=[]
globalstring=""
globalwordsandtime=[]
globalv2wstring=""

resultstrings=[]
globalendtimestamp=0

def fileexists(absfile):
    if os.path.isfile(absfile):
        return True
    else:
        return False



def find_all(sub):

    return [(a.start(), a.end()) for a in list(re.finditer(sub, globalv2wstring))]


def buildglobalwordsandtime(input_wav2vec):
    global globalendtimestamp
    global globalwordsandtime
    global v2wpos
    global globalv2wstring
    overallcnt = 0
    globalv2wstring = ""
    with jsonlines.open(input_wav2vec) as reader:
        print("mm2")
        for obj in reader:
            print("mm1")
            for cnt, chunk in enumerate(obj["chunks"]):
                print(chunk)
                #print("mm")
                res = isinstance(chunk, list)
                if (res == True):
                    for c in chunk:
                        wordtuple = (c["word"], c["timestamp"])
                        globalwordsandtime.append(wordtuple)
                        for ch in c["word"]:
                            v2wpos.append(overallcnt)
                        overallcnt += 1
                        globalv2wstring += str(c["word"].strip())
                        globalendtimestamp = c["timestamp"][1]
                else:
                    wordtuple = (chunk["word"], chunk["timestamp"])
                    globalwordsandtime.append(wordtuple)
                    for ch in chunk["word"]:
                        v2wpos.append(overallcnt)
                    overallcnt += 1
                    globalv2wstring += str(chunk["word"].strip())
                    globalendtimestamp = chunk["timestamp"][1]
                    # print(str(chunk["word"].strip()))


def existinv2wasstart(word,approxstart):
    global v2wpos
    matchlist = find_all(word.strip().lower())
    if matchlist == []:
        return -1
    matchlist.reverse()
    selectedwordno=-1
    for wordindex in matchlist:
        wordno=v2wpos[wordindex[0]]
        localstartime = globalwordsandtime[wordno][1][0]
        if float(localstartime) <= float(approxstart) :
            selectedwordno = wordno
    return selectedwordno

def existinv2wasstop(word,approxstop):
    global v2wpos
    matchlist = find_all(word.strip().lower())
    if matchlist == []:
        return -1
    selectedwordno=-1
    for wordindex in matchlist:
        wordno=v2wpos[wordindex[0]]
        localstoptime = globalwordsandtime[wordno][1][1]
        if float(localstoptime) >= float(approxstop) :
            selectedwordno = wordno
            break
    return selectedwordno

def findstartwordinv2w(inputstring,approxstart,approxstop,startwordoffset,endwordoffset):
    teststring=re.sub(r'[^a-zæøåA-ZÆØÅ0-9 ]', '', str(inputstring)).strip().lower()
    currenttestword = teststring.strip().split()[0]
    selectedwordno= existinv2wasstart(currenttestword,approxstart)
    if selectedwordno != -1:
        print("found start" + str(globalwordsandtime[selectedwordno][0]) + " for " + str(currenttestword))
        startwordtime=globalwordsandtime[selectedwordno][1]
    else:
        print("not found start "  " for " + str(currenttestword))

    currenttestword = teststring.strip().split()[-1]
    selectedwordno = existinv2wasstop(currenttestword, approxstop)

    if selectedwordno != -1:
        print("found stop" + str(globalwordsandtime[selectedwordno][0]) + " for " + str(currenttestword))
        stopwordtime = globalwordsandtime[selectedwordno][1]
    else:
        print("not found stop"  " for " + str(currenttestword))

def findstartwordinv2w_prev(inputstring,approxstart,approxstop,startwordoffset,endwordoffset):
    print("blabla--------->" +str(inputstring))
    limitedteststring=inputstring.strip().split()[0]
    teststring=re.sub(r'[^a-zæøåA-ZÆØÅ0-9]', '', str(limitedteststring))
    print("blabla 2--------->" + str(teststring.strip().lower()))
    print("blabla 3--------->"+str(find_all(teststring.strip().lower())))
    matchlist=find_all(teststring.strip().lower())
    selectedwordno=-1
    for j in matchlist:
        wordno=v2wpos[j[0]]
        localstartime=globalwordsandtime[wordno][1][0]
        #print(str(globalwordsandtime[wordno]) + "  ÅÅÅ   " + str(limitedteststring) + "####" + str(j[0]) + " #### " + str(approxstart) )
        if float(localstartime) <= float(approxstart):
            selectedwordno=wordno

    print("selected: " + str(distance(str(globalwordsandtime[selectedwordno][0]), str(teststring.strip().lower()))))       #(str(globalwordsandtime[selectedwordno]) + "  ÅÅÅ   " + str(limitedteststring))
    print(globalwordsandtime[selectedwordno][0] + "  ÅÅÅ   " + str(teststring.strip().lower()))
    for cnt in range(selectedwordno-startwordoffset-12,selectedwordno+12):
        print("selected distance:" + str(distance(str(globalwordsandtime[cnt][0]),str(teststring.strip().lower()))))
        print("selected: "+ str(cnt)+ " "+ str(globalwordsandtime[cnt][0]) + "  ÅÅÅ   " + str(teststring))

    index=globalv2wstring.find(teststring.strip().lower())


def only_numbers(inputString):
    return all(char.isdigit() for char in inputString)

def formataudiotext(input_file,tmpdir,outputfile):

    global words
    global origwords
    global wordspos
    global globalstring

    tokenizer = nltk.data.load('tokenizers/punkt/norwegian.pickle')
    fp = open(input_file)

    data = fp.readlines()
    cleandata = []
    strdata = ""
    for d in data:
        # print(d)
        if len(d.strip()) > 3 or only_numbers(d.strip()) == False:
            cleandata.append(d)
            strdata += " " + d
    data = strdata

    #arr='\n'.join(tokenizer.tokenize(data))
    fp=open(tmpdir + "/"+ input_file.split("/")[-1] +"_sentences.txt","w+")
    arr = '\n'.join(tokenizer.tokenize(data))
    arr.replace(".","\n")
    arr.replace("-", "")

    fp.write(arr)
    fp.close()
    fp = open(outputfile, "w+")
    overallcnt=0
    with open(tmpdir + "/"+ input_file.split("/")[-1] +"_sentences.txt","r") as textreader:
        for line in textreader:
            firstchar=line.strip()[0]
            if len(line.strip()) <2 and firstchar not in  ['I','i','Å','å']:
                print("Skipping line:" + str(line))
                continue
            l=re.sub(r'[^a-zæøåA-ZÆØÅ0-9 .]', '', line).strip()
            if not(len(l) <= 3 ):
                #print("--->" + str(line))
                if (l[-1] != '.'):
                    l=l+"."
                fp.write(l.lower() + "\n")
                wordsfromline=line.split(" ")
                for cnt,w in enumerate(wordsfromline):
                    #print(w)
                    clean_word=re.sub(r'[^a-zæøåA-ZÆØÅ0-9 ]', '',w).strip()
                    #print(w + str("--->")+ clean_word)
                    words.append(clean_word.lower())
                    origwords.append(w)
                    for ch in clean_word:
                        wordpos.append(overallcnt)
                    overallcnt += 1



    globalstring=''.join(words)
    fp.close()
    #os.remove(tmpdir + "/"+ input_file.split("/")[-1] +"_sentences.txt")

def getwordwithstarttime(starttime):
    for cnt, wordtuple in enumerate(globalwordsandtime):
        if float(wordtuple[1][0]) == starttime:
            return wordtuple[0]

    return " "

def getindexwithstarttime(starttime):
    for cnt, wordtuple in enumerate(globalwordsandtime):
        if float(wordtuple[1][0]) == starttime:
            return cnt

    return -1

def getindexwithstarttime(starttime):
    for cnt, wordtuple in enumerate(globalwordsandtime):
        if float(wordtuple[1][0]) == starttime:
            return cnt

    return -1

def getwordwithendtime(endtime):
    for cnt, wordtuple in enumerate(globalwordsandtime):
        if float(wordtuple[1][1]) == endtime:
            return wordtuple[0]

    return " "

def printmatchingresult(textfile,wav2vecfile,results):
    print("Audiobook: " + textfile + ",wav2vecfile:" + wav2vecfile)

    for data in results:
        #print("XXXXX")
        json_object = json.loads(data)
        # print(str("audiobook text:") + str(json_object["text"]))
        # print(str("wav2vec text:") + str(json_object["wav2vectext"]))
        # print(str("soundfile:") + str(json_object["soundfile"]))
        # print(str("soundstart:") + str(json_object["soundstart"]))
        # print(str("soundend:") + str(json_object["soundend"]))
        # print(str("segstart:") + str(json_object["segstart"]))
        # print(str("segend:") + str(json_object["segend"]))
        # print(str("ratio:") + str(json_object["ratio"]))
        textarray=json_object["text"].split(" ")
        realtext=textarray[int(json_object["segstart"]):int(json_object["segend"])]
        realtextasstring=' '.join(realtext)
        #print(realtextasstring + ":" +str(json_object["soundstart"]) + ":" +str(json_object["soundend"])+ ":" + str(json_object["ratio"]) )
        # print(str(data["text"]))
        # print(data["soundfile"] )
        # print(data["soundstart"] )
        # print(data["soundend"] )
        # print(data["segstart"] )
        # print(data["segend"] )
        # print (data["wav2vectext"])
        # print (data["ratio"] )

def sortresult(result):
    return float(result.split(",")[1])
def result2file(textfile,wav2vecfile,results):
    print("Audiobook: " + textfile + ", wav2vecfile:" + wav2vecfile)

    for data in results:
        #print("XXXXX")
        json_object = json.loads(data)
        # print(str("audiobook text:") + str(json_object["text"]))
        # print(str("wav2vec text:") + str(json_object["wav2vectext"]))
        # print(str("soundfile:") + str(json_object["soundfile"]))
        # print(str("soundstart:") + str(json_object["soundstart"]))
        # print(str("soundend:") + str(json_object["soundend"]))
        # print(str("segstart:") + str(json_object["segstart"]))
        # print(str("segend:") + str(json_object["segend"]))
        # print(str("ratio:") + str(json_object["ratio"]))
        textarray=json_object["text"]
        # print(textarray)
        #
        # print(localcorpus[int(json_object["corp_start"]):int(json_object["corp_end"])])
        # print()

        #realtextarray=localcorpus[int(json_object["corp_start"]):int(json_object["corp_end"])]
        #print(str(textarray) + "%%%%%%"+ str(realtextarray))
       # realtext= realtextarray[int(json_object["segstart"]):int(json_object["segend"])]
        realtext=textarray[int(json_object["segstart"]):int(json_object["segend"])]
        realtextasstring= ' '.join(realtext)
        #print(realtextasstring + ":" +str(json_object["soundstart"]) + ":" +str(json_object["soundend"])+ ":" + str(json_object["ratio"]) )
        #print("ratio:" + str(json_object["ratio"]))
        pos=0

        originaltext=findstringinlocalcorpus(realtextasstring)
        if (originaltext == None):
            continue
        print("FFFZ0:" + originaltext)
        print("FFFZ:" + realtextasstring)
        print("FFFZ2:" + originaltext)

        #print("SSSSSSSS"+ originaltext)
        cleanorgtext=originaltext[originaltext.index("@")+1 :]
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXX   >" + cleanorgtext)

        #findstartwordinv2w(cleanorgtext,float(json_object["soundstart"]),float(json_object["soundend"]))
        startwordoffset=originaltext.split(",")[0]
        endwordoffset=originaltext.split(",")[1].split("@")[0]



        #findstartwordinv2w(cleanorgtext, float(json_object["soundstart"]), float(json_object["soundend"]),int(startwordoffset),int(endwordoffset))
        #print(str(startwordoffset) + "  " + str(json_object["soundstart"]))
        #print(":::::" +str(float(json_object["soundend"]) ))
        #print("[" + realtextasstring + " ///////" + originaltext + "]")
        firstword=str(cleanorgtext).strip().split(" ")[0].lower()
        secondword=str(cleanorgtext).strip().split(" ")[1].lower()
        lastword=str(cleanorgtext).strip().split(" ")[-1].lower()
        secondlastword=str(cleanorgtext).strip().split(" ")[-2].lower()
        print("FFFZ3:" + firstword + ":"+ lastword)

        #start=findwav2vecstarttime(int(startwordoffset),float(json_object["soundstart"] ))
        #end=findwav2vecendtime(int(endwordoffset), float(json_object["soundend"]),lastword)
        #print("abcxxstart" + str(start) + " endwordbefore " + str(findwav2vecendtime_wordbefore(int(startwordoffset),float(json_object["soundstart"]))))
        #print("abcxxend" + str(end)+ " start next word"+ str(findwav2vecstarttime_wordafter(int(endwordoffset), float(json_object["soundend"]),lastword)))

        start = findwav2vecstarttime(int(startwordoffset), float(json_object["soundstart"]))
        end = findwav2vecendtime(int(endwordoffset), float(json_object["soundend"]), secondlastword,lastword)

        #exit(-1)



        ostring="["+str(start) + "----->" +str(end) + ":" +str(startwordoffset)+ ":" +str(endwordoffset)+ "]" + str(originaltext.split(",")[2:])
        print("xzcd---------------------------------->",ostring)
        #exit(-1)
        #print(json_object["corp_start"])
        #print(localcorpus[json_object["corp_start"]])
        if len(ostring)>0 and json_object["ratio"] >0:
            print("FFFZ5:" + str(getwordwithstarttime(start))+ " " + firstword)
            print("FFFZ5:" + str(getwordwithendtime(end)) + " " + lastword)
            if (comparewords(str(getwordwithstarttime(start)),firstword) == True) and (comparewords(str(getwordwithendtime(end)),lastword) == True):
                insertionresult=insertsentencetimestamps(cleanorgtext,start,end)
                #print("tulletull:"+ insertionresult)
                endwordbefore = findwav2vecendtime_wordbefore(int(startwordoffset), float(json_object["soundstart"]))
                startnextword = findwav2vecstarttime_wordafter(int(endwordoffset), float(json_object["soundend"]),
                                                               lastword)
                #ostring = "<M," + str(start) + "," + str(end) + ">"  + str(insertionresult) + "</>"
                if endwordbefore != -1:
                    start = start - ((start - endwordbefore) / 2)
                if startnextword != -1 and end != -1:
                    end = end + ((startnextword - end) / 2)

                ostring = "<M," + str(start) + "," + str(end) + ">" + str(insertionresult) + "</>"

                if end != -1:
                    resultstrings.append(ostring)


        # print(str(data["text"]))
        # print(data["soundfile"] )
        # print(data["soundstart"] )
        # print(data["soundend"] )
        # print(data["segstart"] )
        # print(data["segend"] )
        # print (data["wav2vectext"])
        # print (data["ratio"] )


def  insertsentencetimestamps(cleanorgtext,start,end):
    print("insertsentencetimestamps:" + str(cleanorgtext) + str(start) + ":" +str(end))
    starttimestamp=float(start)
    stoptimestamp = float(end)
    nosentences=countsentences(cleanorgtext)

    words=cleanorgtext.strip().split(" ")
    #for cnt,w in enumerate(words):
        #print(str(cnt) + ":" + str(w))
    #exit(-1)
    #print("ZZZZZZZ"+cleanorgtext)
    newstring=""
    for cnt,w in enumerate(words):
        #print(cnt)
        if len(w.strip()) > 0:
            #newstring+=" " + w
            #print("inside::"+ newstring)
            w=w.strip()
            if (checkiflastletterispunct(w) == True and (cnt != len(words)-1 )) and (stoptimestamp != -1.0):
                 print("insertsentencex----------------------------------------->>>>" + str(w) +str(findwordbetween(w,starttimestamp,stoptimestamp,cnt)))
                 wordbetween=findwordbetween(w,starttimestamp,stoptimestamp,cnt)
                 if len(str(wordbetween)) >= 3:
                    newstring+=" " + w+"<P," + str(wordbetween) + "/>"
                    starttimestamp=float(wordbetween.split(",")[1])
                 else:
                     newstring += " " + w
            else:
                newstring +=" " + w
        #print("inside")
   # print("returnx")
    return  newstring.strip()

def cleanandlowerword(word):
   return re.sub(r'[^a-zæøåA-ZÆØÅ0-9]', '', word).strip().lower()


def findwordbetween(word, starttime, stoptime, approxpos):
    #print("in findbetween" + str(word) + str(":")+ str(starttime) + str(":") + str(stoptime)+ str(" ")+str(approxpos))
    wordclean = cleanandlowerword(word)
    matchlist = find_all(wordclean)
    if (len(matchlist) == 0):
        print("hey empty matchlist")
    # print(matchlist)
    uselaterhit = False
    intervall = ""
    tmpwordno=0
    for matchcount, wordindex in enumerate(matchlist):
        print(str(wordindex[0]) + "####SS" + str(len(v2wpos)))
        if wordindex[0] >= len(v2wpos):
            continue
        wordno = v2wpos[wordindex[0]]


        if (len(globalwordsandtime[wordno][0]) != len(wordclean)):
            continue

        #print(str(globalwordsandtime[wordno][0] + "######%%%%%%" + str(word) + "#######/////" + str(wordclean)))
        #print("in findbetween" + str(word) + str(":") + str(starttime) + str(":") + str(stoptime) + str(" ") + str(approxpos)+ str(" ") + str(wordno))
        # print("jauX" + str(wordno)+ " " + wordclean + str(starttime) +"::" + str(stoptime) + ":" + globalwordsandtime[wordno][0])
        if (globalwordsandtime[wordno][1][0] > starttime) and (globalwordsandtime[wordno][1][1] <= stoptime):
            if matchcount < len(matchlist) - 1:
                tmpindex = matchlist[matchcount+1]
                tmpwordno = v2wpos[tmpindex[0]]
                print("in findbetween" + str(word) + str(":") + str(starttime) + str(":") + str(stoptime) + str( " ") + str(approxpos) + str(" ") + str(wordno)+ str(" ") + str(tmpwordno))
                if (globalwordsandtime[tmpwordno][1][0] > starttime) and (globalwordsandtime[tmpwordno][1][1] <= stoptime) and tmpwordno - wordno < 12:
                    print("in findbetween using later hit")
                    uselaterhit = True

            if uselaterhit == True:
                intervall = str(globalwordsandtime[tmpwordno][1][1]) + "," + str(globalwordsandtime[tmpwordno+1][1][0])
            else:
                intervall = str(globalwordsandtime[wordno][1][1]) + "," + str(globalwordsandtime[wordno+1][1][0])
            print("return findbetween 1")
            return intervall
    print("return findbetween 2")
    return " "

def countsentences(words):
    return words.count('.')+1

def comparewords(w1,w2):
    word1 = re.sub(r'[^a-zæøåA-ZÆØÅ0-9]', '', w1).strip().lower()
    word2 = re.sub(r'[^a-zæøåA-ZÆØÅ0-9]', '', w2).strip().lower()
    if (word1 == word2):
        return True
    else:
        return False
def findwav2vecstarttime(offset, currentstarttime):
    global globalwordsandtime
    index=-1
    for cnt,wordtuple in enumerate(globalwordsandtime):
        if float(wordtuple[1][0]) == currentstarttime:
            index=cnt
            break
            #print("ppppp"+ str(wordtuple[1][0]))

    if (index - offset) <= 0:
        return globalwordsandtime[0][1][0]
    else:
        return globalwordsandtime[index-offset][1][0]

def findwav2vecstarttime_todelete(offset, currentstarttime,firstword,secondword):
    global globalwordsandtime
    index=-1
    for cnt,wordtuple in enumerate(globalwordsandtime):
        if float(wordtuple[1][0]) == currentstarttime:
            index=cnt
            break
            #print("ppppp"+ str(wordtuple[1][0]))

    firstwordclean = cleanandlowerword(firstword)
    secondwordclean= cleanandlowerword(secondword)
    while index > 0 and firstwordclean != globalwordsandtime[index][0] and secondwordclean != globalwordsandtime[index + 1][0]:
        # print("@@@@"+word+ " " +globalwordsandtime[index][0]+ " " +str(currentendtime))
        index -= 1

    if (index < 0):
        return -1

    elif currentstarttime - globalwordsandtime[index][1][1]  > 100:
        return -1
        # index = startindex + offset + 1
        # print("VVVVVVVVVVVVVVVVVVVVVVV"+ str(wordin) + " " + str(globalwordsandtime[index][1][1]))
        # return globalwordsandtime[index][1][1]
    else:
        return globalwordsandtime[index][1][1]
  #  if (index - offset) <= 0:
   #     return globalwordsandtime[0][1][0]
    #else:
        #return globalwordsandtime[index-offset][1][0]

def findwav2vecendtime_wordbefore(offset, currentstarttime):
    global globalwordsandtime
    index=-1
    for cnt,wordtuple in enumerate(globalwordsandtime):
        if float(wordtuple[1][0]) == currentstarttime:
            index=cnt-1
            break
            #print("ppppp"+ str(wordtuple[1][0]))

    if (index - offset) <= 0:
        return -1
    else:
        return globalwordsandtime[index-offset][1][1]

def findwav2vecstarttime_wordafter(offset, currentendtime,wordin):
    global globalwordsandtime
    index = -1
    # word = re.sub(r'[^a-zæøåA-ZÆØÅ]', '', wordin).strip().lower()
    word = cleanandlowerword(wordin)
    # print("xxxx"+ str(currentendtime))
    for cnt, wordtuple in enumerate(globalwordsandtime):
        # print(str(currentendtime) + " xxxxy "+ str(wordtuple[1][1]))
        if float(wordtuple[1][1]) == float(currentendtime):
            index = cnt -1
            # print("ppppp" + str(wordtuple[1][1]))
            break
            # print("ppppp"+ str(wordtuple[1][0]))
    startindex = index
    while index < len(globalwordsandtime) and word != globalwordsandtime[index][0]:
        # print("@@@@"+word+ " " +globalwordsandtime[index][0]+ " " +str(currentendtime))
        index += 1
    if (index >= len(globalwordsandtime)-1):
        return -1

    elif globalwordsandtime[index][1][1] - currentendtime > 100:
        return -1
        # index = startindex + offset + 1
        # print("VVVVVVVVVVVVVVVVVVVVVVV"+ str(wordin) + " " + str(globalwordsandtime[index][1][1]))
        # return globalwordsandtime[index][1][1]
    else:
        return globalwordsandtime[index+1][1][0]

def findwav2vecendtime(offset, currentendtime,secondlastwordin,wordin):
    global globalwordsandtime
    index = -1
    #word = re.sub(r'[^a-zæøåA-ZÆØÅ]', '', wordin).strip().lower()
    word=cleanandlowerword(wordin)
    secondlastword=cleanandlowerword(secondlastwordin)
    #print("xxxx"+ str(currentendtime))
    for cnt, wordtuple in enumerate(globalwordsandtime):
        #print(str(currentendtime) + " xxxxy "+ str(wordtuple[1][1]))
        if float(wordtuple[1][1]) == float(currentendtime):
            index = cnt-1
            #print("ppppp" + str(wordtuple[1][1]))
            break
            # print("ppppp"+ str(wordtuple[1][0]))
    startindex=index
    while index < len(globalwordsandtime) and word != globalwordsandtime[index][0] and secondlastword != globalwordsandtime[index-1][0]:
        #print("@@@@"+word+ " " +globalwordsandtime[index][0]+ " " +str(currentendtime))
        index+=1
    if (index >= len(globalwordsandtime)):
        return -1

    elif globalwordsandtime[index][1][1] -currentendtime > 100:
        return -1
        # index = startindex + offset + 1
        # print("VVVVVVVVVVVVVVVVVVVVVVV"+ str(wordin) + " " + str(globalwordsandtime[index][1][1]))
        # return globalwordsandtime[index][1][1]
    else:
        return globalwordsandtime[index][1][1]


def findwav2vecendtime_off(offset, currentendtime,wordin):
    global globalwordsandtime
    index=-1
    for cnt,wordtuple in enumerate(globalwordsandtime):
        if float(wordtuple[1][1]) == currentendtime:
            index=cnt
            break
            #print("ppppp"+ str(wordtuple[1][0]))

    if (index + offset) <= 0:
        return globalwordsandtime[0][1][1]
    else:
        return globalwordsandtime[index-offset][1][1]



def checkifwordcapitalized(word):
    w = re.sub(r'[^a-zæøåA-ZÆØÅ0-9]', '', word).strip()
    if (len(w) == 0):
        return False
    # If the first character is not a quote, check if it's capitalized
    return w[0].isupper()

def checkiflastletterispunct(word):
    w = re.sub(r'[^a-zæøåA-ZÆØÅ0-9.»?!]', '', word).strip()
    wclean=w.strip()
    if (len(wclean) == 0):
        return False
    last_char = wclean[-1]
    if (last_char in ".?!»"):
        return True
    else:
        return False


def findstringinlocalcorpus(inputstring):
    global words
    global origwords
    global globalstring
    global wordpos
    global globalwordsandtime

    if len(inputstring.strip()) < 1:
        return
    nowords=len(inputstring.strip().split(" "))
    if (nowords < 1):
        return

    barestring=inputstring.replace(" ","").strip()
    startword = 0
    initpoint=0
    startenhance=0
    #retstring="0,0,\"\""
    retstring=None
    print("### x----------------------------------------------------------------------------------------------------------------------->"+barestring)
    if barestring in globalstring:
        startword=wordpos[globalstring.find(barestring)]
        #initpoint=startword
        print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"+str(origwords[startword]) + str("&&&&&&") + str(origwords[startword+1]) + str("%%%%%%%%")+ str(inputstring.split(" ")[0]))
        limit=5
        #print("XZ:" +origwords[startword] +"0")
        while startword > 1 and checkifwordcapitalized(origwords[startword]) == False:
            startword-=1
            startenhance+=1
            nowords+=1
            limit-=1
            #print("XZ:" + origwords[startword] + str(startenhance))
       # startword+=1
        #nowords -= 1
        endenhance=0
        #print("XyZ:" + origwords[startword+nowords+endenhance] + "0")
        while startword+nowords<len(origwords) and checkiflastletterispunct(origwords[startword+nowords].strip()) == False:
            #print("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW" + str(nowords))
            endenhance+=1
            nowords +=1
            #print("XyZ:" + origwords[startword + nowords + endenhance] + str(endenhance))

    #here
    #print("x" +origwords[startword+nowords].strip())
    #print("y" +origwords[startword + nowords+1].strip())
    #print("["+ str(' '.join(origwords[startword:startword+nowords]) + "????" + inputstring+ "]"))
    #print(startword)
    #print(str(origwords[0]) + "$$" + str(globalwordsandtime[0]))
    #findinglobalvectime(initpoint,startword,startword+nowords+1)
        retstring=str(startenhance)+","+str(endenhance)+"@"+str(' '.join(origwords[startword:startword+nowords+1]).replace("\n"," "))
        #retstring = str(' '.join(origwords[startword:startword + nowords + 1]).replace("\n", " "))
    return retstring








def buildwav2vecformat(input_wav2vec,output_wav2vec):
    #global globalwordsandtime
    accumulatedtime=0
    jsonfilewriter=open(output_wav2vec, "w+")
    cnt=0
    with jsonlines.open(input_wav2vec) as reader:
        for obj in reader:
            #print(obj["text"])
            filename=obj["file"]
            accumulatedstart = 0
            words=[]
            timestamps=[]
            splitnumber=0
            for cnt,chunk in enumerate(obj["chunks"]):
                res = isinstance(chunk, list)
                if (res == True):
                    # print("islist")
                    chunk = chunk[0]

                if accumulatedstart == 0:
                    accumulatedstart=chunk["timestamp"][0]


                #print(chunk["word"])
                #currentduration=(chunk["timestamp"][1] - chunk["timestamp"][0])
                currentstop=float(chunk["timestamp"][1])
                nextseq=cnt+1

                if (nextseq >= len(obj["chunks"])):
                    data = {}
                    data["file"] = input_wav2vec + "_" + str(splitnumber)
                    data["start"] = accumulatedstart
                    data["end"] = currentstop
                    data["text"] = ' '.join(words)
                    wordlist = []
                    for cnt, word in enumerate(words):
                        worddata = {}
                        worddata["word"] = word
                        worddata["timestamp"] = timestamps[cnt]
                        wordlist.append(worddata)
                        #print(str("------------------------------------->")+ str(word)+ ":::" +str(timestamps[cnt]))

                    data["chunks"] = wordlist
                    #print(data)
                    jsonfilewriter.write(json.dumps(data, ensure_ascii=False))
                    jsonfilewriter.write("\n")
                    jsonfilewriter.close()
                    # print(globalwordsandtime)
                    # for i in globalwordsandtime:
                    #     print(i)
                    return

                nextchunk = obj["chunks"][nextseq]
                res = isinstance(nextchunk, list)
                if (res == True):
                    # print("islist")
                    nextchunk = nextchunk[0]



                #print(currentduration)
                if ((nextchunk["timestamp"][0]- currentstop) > splittimelimit):
                    data={}
                    data["file"]=input_wav2vec+"_" + str(splitnumber)
                    data["start"]=accumulatedstart
                    data["end"] = currentstop
                    data["text"] = ' '.join(words)

                    wordlist = []
                    for cnt,word in enumerate(words):

                        worddata={}
                        worddata["word"]=word
                        worddata["timestamp"] = timestamps[cnt]
                        wordlist.append(worddata)
                        #print(str("------------------------------------->") + str(word) + ":::" + str(timestamps[cnt]))

                    data["chunks"]=wordlist
                    accumulatedstart = 0
                    words = []
                    timestamps=[]
                    #print(data)
                    jsonfilewriter.write(json.dumps(data, ensure_ascii=False))
                    jsonfilewriter.write("\n")

                    splitnumber+=1
                else:
                    words.append(chunk["word"])
                    timestamps.append(chunk["timestamp"])


    jsonfilewriter.close()


def matchsoundandtext(textfile,wav2vecfile):
    matcher = Matcher(Path(textfile))
    asr_results = load_segments(Path(wav2vecfile))
    positions = matcher.match(asr_results)
    matchingresults=matcher.resultingjson(positions)
    result2file(textfile,wav2vecfile,matchingresults)
    os.remove(textfile)
    #os.remove(wav2vecfile)
    #extcorp=matcher.getCorpus()
    #print(extcorp[100:240])
    #print(localcorpus[100:240])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_audiobook', help='File ordinary audiobook', required=True, type=str)
    parser.add_argument('--input_wav2vecfile', help='File ordinary audiobook', required=True, type=str)
    parser.add_argument('--tmpdir', help='temporary dir', required=True, type=str)
    parser.add_argument('--resultfile', help='file with results', required=True, type=str)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    if fileexists(args.resultfile):
        print ("File already exists " + args.resultfile)
        exit(0)
    formataudiotext(args.input_audiobook,args.tmpdir,args.tmpdir+"/"+args.input_audiobook.split("/")[-1])
    #print("formatting done")
    inputfile=args.input_wav2vecfile
    v2wfile = args.tmpdir + "/" + inputfile.split("/")[-1].split(".")[0] + "_prepared.jsonl"
    #print(v2wfile)
    buildglobalwordsandtime(args.input_wav2vecfile)
    buildwav2vecformat(args.input_wav2vecfile, v2wfile)
    #print(v2wfile)
    matchsoundandtext(args.tmpdir+"/"+args.input_audiobook.split("/")[-1],v2wfile)
    fp=open(args.resultfile,"w+")
    resultstrings.sort(key=sortresult)
    sumduration=0
    uniqueresults=list(set(resultstrings)).sort(key=sortresult)
    prev=""
    lastendpoint=0
    for r in resultstrings:
        thisendpoint=float(r.split(",")[2].split(">")[0])
        if r != prev and lastendpoint!=thisendpoint:
            sumduration+= (float(r.split(",")[2].split(">")[0]) -float(r.split(",")[1]))
            fp.write(r)
            fp.write("\n")
            lastendpoint=thisendpoint
        prev=r
    fp.close()
    percentage=round(sumduration/globalendtimestamp *100,2)
    #print("Input tekst:" + str(args.input_audiobook.split("/")[-1]) + " w2v file: " + str(args.input_wav2vecfile.split("/")[-1]))
    print("Extracted "+ str(round(sumduration,2)) + " seconds of " + str(round(globalendtimestamp,2)) + " seconds (" + str(percentage) + ")" + " from " + str(args.input_audiobook.split("/")[-1]) )
    #findstartwordinv2w("Knut! jubler")
