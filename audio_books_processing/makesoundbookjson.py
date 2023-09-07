
import json
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
from pydub import AudioSegment,silence
import fasttext
from datasets.utils.download_manager import DownloadManager
from typing import Optional, List, Set, Union, Tuple
import subprocess


NORDIC_LID_URL = "https://huggingface.co/NbAiLab/nb-nordic-lid/resolve/main/"
model_name = "nb-nordic-lid.bin"
print("started processing")
model = fasttext.load_model(DownloadManager().download(NORDIC_LID_URL + model_name))
model_labels = set(label[-3:] for label in model.get_labels())
print("end")
#nno,nob
def detect_lang(
    text: str,
    langs: Optional[Union[List, Set]]=None,
    threshold: float=-1.0,
    return_proba: bool=False
) -> Union[str, Tuple[str, float]]:
    """
    This function takes in a text string and optional arguments for a list or
    set of languages to detect, a threshold for minimum probability of language
    detection, and a boolean for returning the probability of detected language.
    It uses a pre-defined model to predict the language of the text and returns
    the detected ISO-639-3 language code as a string. If the return_proba
    argument is set to True, it will also return a tuple with the language code
    and the probability of detection. If no language is detected, it will
    return "und" as the language code.
    Args:
    - text (str): The text to detect the language of.
    - langs (List or Set, optional): The list or set of languages to detect in
        the text. Defaults to all languages in the model's labels.
    - threshold (float, optional): The minimum probability for a language to be
        considered detected. Defaults to `-1.0`.
    - return_proba (bool, optional): Whether to return the language code and
        probability of detection as a tuple. Defaults to `False`.
    Returns:
    str or Tuple[str, float]: The detected language code as a string, or a
        tuple with the language code and probability of detection if
        return_proba is set to True.
    """
    if langs:
        langs = set(langs)
    else:
        langs = model_labels
    raw_prediction = model.predict(text, threshold=threshold, k=-1)
    predictions = [
        (label[-3:], min(probability, 1.0))
        for label, probability in zip(*raw_prediction)
        if label[-3:] in langs
    ]
    if not predictions:
        return ("und", 1.0) if return_proba else "und"
    else:
        return predictions[0] if return_proba else predictions[0][0]







#df["word_count_subtitles"] = df["text"].str.split().apply(len)    df["word_count_transcription"] = df["text_transcription"].str.split().apply(len)    df["verbosity_score"] = np.where(        df["word_count_transcription"] != 0, df["word_count_subtitles"]/df["word_count_transcription"], 0)
    # Calculate Verbosity    df["verbosity"] = np.choose(np.searchsorted(np.array(        [0.50, 0.60, 0.70, 0.80, 0.90]), df["verbosity_score"]), [1, 2, 3, 4, 5, 6])

offsetstartmillisec=0
from argparse import ArgumentParser
options=None
null=None
def parse_args():
    global options
    parser = ArgumentParser()


    parser.add_argument("-masteraudiooutputdir", "--masteraudiooutputdir", dest="masteraudiooutputdir", help="masteraudiooutputdir",
                        required=True)

    parser.add_argument("-masteraudioinputdir", "--masteraudioinputdir", dest="masteraudioinputdir", help="masteraudiodir",
                        required=True)
    parser.add_argument("-masterextractinputdir", "--masterextractinputdir", dest="masterextractinputdir",
                        help="masterextractinputdir",
                        required=True)

    options = parser.parse_args()
    return options

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


def extractfromsoundfile_ffmpeg(inputmp3,outputdir,extractionlist):
    mkdirifnotexists(outputdir)
    print("reading sound file")
    for cnt,e in enumerate(extractionlist):
        outputfilename = outputdir+"/"+e["filename"]
        print("Extracting: " + str(outputfilename))
        frommillisec=int(float(e["start"]) * 1000)
        frommillisec += offsetstartmillisec
        tomillisec=int(float(e["stop"]) * 1000)
        subprocess.run(['ffmpeg', '-i', inputmp3, '-ss', str(frommillisec / 1000), '-to', str(tomillisec / 1000), '-ac', '1', '-ab', '16000', '-c:a', 'libmp3lame', '-q:a', '4', '-c','copy','-y',outputfilename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def extractfromsoundfile(inputmp3,outputdir,extractionlist):
    mkdirifnotexists(outputdir)
    print("reading sound file")
    soundfile = AudioSegment.from_mp3(inputmp3.strip())
    soundfile.set_frame_rate(16000).set_channels(1)

    print("end reading sound file")
    for cnt,e in enumerate(extractionlist):

        outputfilename = outputdir+"/"+e["filename"]
        print("Extracting: " + str(outputfilename))
        frommillisec=int(float(e["start"]) * 1000)

        frommillisec += offsetstartmillisec
        tomillisec=int(float(e["stop"]) * 1000)
        extractedfile = soundfile[frommillisec:tomillisec]
        extractedfile.set_frame_rate(16000).set_channels(1).export(outputfilename, format="mp3", bitrate="16000")

def filesizelargerthanzero(inpfile):
    fsize=os.path.getsize(inpfile)
    if fsize > 0:
        return True
    else:
        return False

def processfile(extractedsoundbookfile):
    return True

def  calculateverbosity(a,b):
    return 1,2


if __name__ == "__main__":
    # extractionlist=(("ff",10,20),("gg",30,40))
    # extractfromsoundfile("9789180515016_content.mp3","tmpmp3",extractionlist)
    # exit(-1)
    lastjson={}
    options = parse_args()

    masteraudiooutputdir = options.masteraudiooutputdir
    masteraudioinputdir = options.masteraudioinputdir
    masterextractinputdir = options.masterextractinputdir
    splitlist = []
    for extractfile in glob.glob(masterextractinputdir + "/*.txt"):
        if filesizelargerthanzero(extractfile) == False:
            print("Zero file " + str(extractfile))
            continue

        outputdir = masteraudiooutputdir + "/" + extractfile.split("/")[-1].split("-")[-1].split(".")[0]
        mkdirifnotexists(outputdir)
        outputfilejson = outputdir + "/" + extractfile.split("/")[-1].split("-")[-1].split(".")[0] + ".json"
        if fileexists(outputfilejson):
            print("Skipping " + str(extractfile) + " because file " + str(outputfilejson) + " exists")
            continue

        outfp = open(outputfilejson, "w+", encoding='utf8')

        with open(extractfile) as f:
            # print(f)
            soundfile = extractfile.split("/")[-1].split("-")[-1].split(".")[0] + "_content.mp3"
            soundbookid = extractfile.split("/")[-1].split("-")[-1].split(".")[0]
            splitlist = []
            for line in f:
                print(line.strip())
                splitcodes = re.findall(r'\<.*?\>', line)
                line=line.strip().replace('\r', ' ').replace('\n', ' ').replace('  ',' ')

                print(splitcodes)

                startsecond = float(splitcodes[0].split(",")[1])
                endsecond = float(splitcodes[0].split(",")[2].split(">")[0])
                duration = endsecond - startsecond
                cleanline = re.sub("[\<].*?[\>]", "", line)
                print(cleanline)
                textlanguage = detect_lang(cleanline.strip(), langs=["nob", "nno", "eng"])
                textlanguageconverged = ""
                if textlanguage == "nob":
                    textlanguageconverged = str("no")
                elif textlanguage == "nno":
                    textlanguageconverged = str("nn")
                elif textlanguage == "eng":
                    textlanguageconverged = str("en")
                else:
                    textlanguageconverged = null
                # audiofilename = soundbookid + "/" + currentid + ".mp3"
                if duration > 30 and len(splitcodes) > 3:
                    stringparts = list(filter(None, re.sub("[\<].*?[\>]", "<sub>", line).split("<sub>")))
                    for cnt, part in enumerate(stringparts):
                        if (cnt == 0):
                            currentstartsecond = float(startsecond)
                            currentendsecond = float(splitcodes[1].split(",")[1])
                            currentdurationseconds = float(currentendsecond) - float(currentstartsecond)
                            print(str(currentstartsecond) + ":" + str(currentendsecond) + ":" + part)
                            currentid = soundbookid + "_" + str(int(float(currentstartsecond) * 1000)) + "_" + str(int(float(currentendsecond) * 1000))
                            audiofilename = soundbookid + "/" + currentid + ".mp3"

                            ourjson = {
                                "id": str(currentid),
                                "group_id": null,
                                "source": str("audio_books"),
                                "audio_language": str("no"),
                                "audio_duration": int(currentdurationseconds * 1000),
                                "previous_text": null,
                                "text_language": textlanguageconverged,
                                "text": str(part),
                                "translated_text_no": null,
                                "translated_text_nn": null,
                                "translated_text_en": null,
                                "translated_text_es": null,
                                "timestamped_text": null,
                                "wav2vec_wer": null,
                                "whisper_wer": null,
                                "verbosity_level": 6,
                            }
                            if ourjson != lastjson:
                                if currentdurationseconds <= (len(ourjson["text"].split()) *2):
                                    json.dump(ourjson, outfp, ensure_ascii=False)
                                    outfp.write("\n")
                                    splitelement = {}
                                    splitelement["filename"] = audiofilename.split("/")[-1]
                                    splitelement["start"] = currentstartsecond
                                    splitelement["stop"] = currentendsecond
                                    if splitlist == []:
                                        splitlist.append(splitelement)
                                    else:
                                        splitlist.append(splitelement)
                                    lastjson=ourjson


                        elif cnt == (len(stringparts) - 1):
                            print(splitcodes[cnt].split(",")[2].split("/")[0] + ":" + str(endsecond) + ":" + part)

                            currentstartsecond = float(splitcodes[cnt].split(",")[2].split("/")[0])
                            currentendsecond = float(endsecond)
                            currentdurationseconds = float(currentendsecond) - float(currentstartsecond)
                            # print(str(currentstartsecond) + ":" + str(currentendsecond) + ":" + part)
                            currentid = soundbookid + "_" + str(int(float(currentstartsecond) * 1000)) + "_" + str(
                                int(float(currentendsecond) * 1000))
                            audiofilename = soundbookid + "/" + currentid + ".mp3"
                            ourjson = {
                                "id": str(currentid),
                                "group_id": null,
                                "source": str("audio_books"),
                                "audio_language": str("no"),
                                "audio_duration": int(currentdurationseconds * 1000),
                                "previous_text": null,
                                "text_language": textlanguageconverged,
                                "text": str(part),
                                "translated_text_no": null,
                                "translated_text_nn": null,
                                "translated_text_en": null,
                                "translated_text_es": null,
                                "timestamped_text": null,
                                "wav2vec_wer": null,
                                "whisper_wer": null,
                                "verbosity_level": 6,
                            }
                            if ourjson != lastjson:
                                if currentdurationseconds <= (len(ourjson["text"].split()) * 2):
                                    json.dump(ourjson, outfp, ensure_ascii=False)
                                    outfp.write("\n")
                                    splitelement = {}
                                    splitelement["filename"] = audiofilename.split("/")[-1]
                                    splitelement["start"] = currentstartsecond
                                    splitelement["stop"] = currentendsecond
                                    if splitlist == []:
                                        splitlist.append(splitelement)
                                    else:
                                        splitlist.append(splitelement)

                                    break
                                    lastjson=ourjson
                        else:
                            print(splitcodes)
                            print(splitcodes[cnt].split(",")[2].split("/")[0] + ":" + str(splitcodes[cnt + 1].split(",")[1]) + ":" + part)
                            print(splitcodes)
                            currentstartsecond = float(splitcodes[cnt].split(",")[2].split("/")[0])
                            currentendsecond = float(splitcodes[cnt + 1].split(",")[1])
                            currentdurationseconds = float(currentendsecond) - float(currentstartsecond)
                            # print(str(currentstartsecond) + ":" + str(currentendsecond) + ":" + part)
                            currentid = soundbookid + "_" + str(int(float(currentstartsecond) * 1000)) + "_" + str(
                                int(float(currentendsecond) * 1000))
                            audiofilename = soundbookid + "/" + currentid + ".mp3"
                            ourjson = {
                                "id": str(currentid),
                                "group_id": null,
                                "source": str("audio_books"),
                                "audio_language": str("no"),
                                "audio_duration": int(currentdurationseconds * 1000),
                                "previous_text": null,
                                "text_language": textlanguageconverged,
                                "text": str(part),
                                "translated_text_no": null,
                                "translated_text_nn": null,
                                "translated_text_en": null,
                                "translated_text_es": null,
                                "timestamped_text": null,
                                "wav2vec_wer": null,
                                "whisper_wer": null,
                                "verbosity_level": 6,
                            }
                            if ourjson != lastjson:
                                if currentdurationseconds <= (len(ourjson["text"].split()) * 2):
                                    json.dump(ourjson, outfp, ensure_ascii=False)
                                    outfp.write("\n")
                                    splitelement = {}
                                    splitelement["filename"] = audiofilename.split("/")[-1]
                                    splitelement["start"] = currentstartsecond
                                    splitelement["stop"] = currentendsecond
                                    if splitlist == []:
                                        splitlist.append(splitelement)
                                    else:
                                        splitlist.append(splitelement)
                                    lastjson = ourjson


                else:
                    currentid = soundbookid + "_" + str(int(startsecond * 1000)) + "_" + str(int(endsecond * 1000))
                    audiofilename = soundbookid + "/" + currentid + ".mp3"
                    print("#" + str(startsecond) + ":" + str(endsecond) + ":" + str(cleanline))

                    ourjson = {
                        "id": str(currentid),
                        "group_id": null,
                        "source": str("audio_books"),
                        "audio_language": str("no"),
                        "audio_duration": int(duration * 1000),
                        "previous_text": null,
                        "text_language": textlanguageconverged,
                        "text": str(cleanline),
                        "translated_text_no": null,
                        "translated_text_nn": null,
                        "translated_text_en": null,
                        "translated_text_es": null,
                        "timestamped_text": null,
                        "wav2vec_wer": null,
                        "whisper_wer": null,
                        "verbosity_level": 6,
                    }
                    if ourjson != lastjson:
                        currentdurationseconds = float(endsecond) - float(startsecond)
                        if currentdurationseconds <= (len(ourjson["text"].split()) * 2):
                            json.dump(ourjson, outfp, ensure_ascii=False)
                            outfp.write("\n")
                            splitelement = {}
                            splitelement["filename"] = audiofilename.split("/")[-1]
                            splitelement["start"] = startsecond
                            splitelement["stop"] = endsecond
                            if splitlist == []:
                                splitlist.append(splitelement)
                            else:
                                splitlist.append(splitelement)
                            lastjson=ourjson

        outfp.close()
        size=os.path.getsize(masteraudioinputdir + "/" + soundfile)
        print(size)
        #4294967296
        if size >= 20000:
            extractfromsoundfile_ffmpeg(masteraudioinputdir + "/" + soundfile, outputdir + "/mp3", splitlist)
        else:
            extractfromsoundfile(masteraudioinputdir + "/" + soundfile, outputdir + "/mp3", splitlist)
