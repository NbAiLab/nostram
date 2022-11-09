import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import glob
import os
import re 
from transformers import pipeline, T5ForConditionalGeneration, AutoTokenizer

def main(args):
    tokenizer = AutoTokenizer.from_pretrained("north/fine_North_large")
    model = T5ForConditionalGeneration.from_pretrained("north/fine_North_large")
    text2text_generator = pipeline("text2text-generation",model=model,tokenizer=tokenizer,batch_size=256)



    print("Model should load here")
    
    def remove_brackets(text):
        text = re.sub("[\<].*?[\>]", "", text)
        #output = re.sub("[\(\[].*?[\)\]]", "", str(text))
        return text

    def remove_duplicate_words(text):
        input = text.split()
        output = []
        prev = ""
        dup = False
        
        for i in input:
            if i != prev:
                output.append(i)
            else:
                dup = True
            prev = i
        
        final = ' '.join(output)
        return final


    filelist = glob.glob(os.path.join(args.directory,"*.json"))
    for f in filelist:
        data = pd.DataFrame()
        data = pd.read_json(f, lines=True,orient='records')
        
        data['clean'] = data['normsentence_text']
        data['clean'] = data['clean'].apply(remove_brackets)
        data['clean'] = data['clean'].apply(remove_duplicate_words)
        data['sentence_nbo'] = "..."
        data['sentence_nno'] = "..."


        #The ortographic clean text is not in clean
        #For Norwegian Bokmål
        print("Correcting Norwegian Bokmål")
        sentences = ["correct: " + s for s in data[data['sentence_language_code'] == 'nb-NO']['clean']]
        output = text2text_generator(sentences)
        data.loc[data['sentence_language_code'] == "nb-NO",'sentence_nob'] = [o['generated_text'] for o in output]
        
        print("Translating Norwegian Bokmål to Nynorsk")
        sentences = ["nyn: " + s for s in data[data['sentence_language_code'] == 'nb-NO']['sentence_nob']]
        output = text2text_generator(sentences)
        data.loc[data['sentence_language_code'] == "nb-NO",'sentence_nno'] = [o['generated_text'] for o in output]
         
        print("Correcting Norwegian Nynorsk")
        sentences = ["correct: " + s for s in data[data['sentence_language_code'] == 'nn-NO']['clean']]
        output = text2text_generator(sentences)
        data.loc[data['sentence_language_code'] == "nn-NO",'sentence_nno'] = [o['generated_text'] for o in output]
        
        print("Translating Norwegian Nynorsk to Bokmål")
        sentences = ["nbo: " + s for s in data[data['sentence_language_code'] == 'nn-NO']['sentence_nno']]
        output = text2text_generator(sentences)
        data.loc[data['sentence_language_code'] == "nn-NO",'sentence_nbo'] = [o['generated_text'] for o in output]
         
        data = data.drop(['clean'], axis=1)
       
        output_filename = f.replace(args.directory,args.output_directory)
        
        print(f"Starting to write: {output_filename}")

        with open(output_filename, 'w', encoding='utf-8') as file:
            data.to_json(output_filename,force_ascii=False,orient="records",lines=True)

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
