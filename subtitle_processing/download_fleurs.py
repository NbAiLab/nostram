from datasets import load_dataset
import os
import argparse

def main(args):
    if not os.path.exists(args.output_folder+"/audio"):
            os.makedirs(args.output_folder+"/audio") 
    
    norwegian_fleurs  = load_dataset("google/fleurs", "nb_no")

    norwegian_fleurs = norwegian_fleurs.map(remove_columns=["audio"])
    print(norwegian_fleurs)

    for split,dataset in norwegian_fleurs.items():
            dataset.to_json(f"{args.output_folder}/norwegian_fleurs-{split}.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', required=True, help='Path to output folder.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
