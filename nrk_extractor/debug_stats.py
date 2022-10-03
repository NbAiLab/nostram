
import json
import sys
import os
import jsonlines
from argparse import ArgumentParser
import pandas as pd
import glob
import urllib.request

###################################################
# Debug script made for detecting missing subtitles
###################################################

def main(args):
    target_file = os.path.join(args.directory,'process_list/','tv.json')
    
    data = pd.read_json(target_file, lines=True) # read data frame from json file
    
    ## Lets just work with part of the data
    data = data.sample(n=100)

    breakpoint()


def parse_args():
    # Parse commandline
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", dest="directory", help="Directory to Json-lines file to analyse", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
