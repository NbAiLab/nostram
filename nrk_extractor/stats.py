
import json
import sys
import jsonlines
from argparse import ArgumentParser
import pandas as pd

def main(args):
    df = pd.read_json(args.jsonl, lines=True)
    programs_detailed = df.groupby(["title","program_id","subtitle","category"])['duration'].agg(['sum','count'])
    programs_detailed['hours'] = (programs_detailed['sum']/100/3600).round(1)
    programs_detailed = programs_detailed.drop(columns=['sum'])

    programs = df.groupby(["title"])['duration'].agg(['sum','count'])
    programs['hours'] = (programs['sum']/100/3600).round(1)
    programs = programs.drop(columns=['sum'])

    breakpoint()
    #with jsonlines.open(args.jsonl) as reader:
    #   for line in reader:
    #        print(line)

def parse_args():
    # Parse commandline
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="jsonl", help="Json-lines file to analyse", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
