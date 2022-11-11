import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import glob
import os
import re 

def main(args):


    filelist = glob.glob(os.path.join(args.directory,"*.json"))
    for f in filelist:
        data = pd.DataFrame()
        data = pd.read_json(f, lines=True,orient='records')
        
        breakpoint()

        output_filename = f.replace(args.directory,args.output_directory)
        
        print(f"Starting to write: {output_filename}")

        #with open(output_filename, 'w', encoding='utf-8') as file:
        #    data.to_json(output_filename,force_ascii=False,orient="records",lines=True)

def parse_args():
    # Parse commandline
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", dest="directory", help="Directory to Json-lines file to annotate.", required=True)
    parser.add_argument("-o", "--output_directory", dest="output_directory", help="Directory to store the processed Json-lines file.", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
