import requests
import glob
import json
import re
import os
import subprocess
import time
import sys
import argparse
import isodate
import csv
import dateutil.parser
import jsonlines

##################################################################################
# Try to get all the VTT files so they do not disappear
##################################################################################

def main(args):
    vtt_path = args.path+"VTT_test"
    audio_path = args.path+"audio"
    
    filelist = glob.glob(audio_path+"/*.*")
    for f in filelist[0:10]:
        print(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="Complete path to where both the audio and VTT are stored.", required=True)

    args = parser.parse_args()
    main(args)








