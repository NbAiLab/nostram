import pandas as pd
import jiwer

def normalizer(text):
    # Normalization steps go here
    # For example:
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveWhiteSpace(replace_by_space=True)
    ])
    return transformation(text)

def calculate_wer(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Normalize raw_text and predictions
    df['raw_text'] = df.raw_text.apply(normalizer)
    df['predictions'] = df.predictions.apply(normalizer)
    
    # Filter out rows where raw_text has less than 6 words
    df = df[df['raw_text'].apply(lambda x: len(x.split()) >= 6)]

    # Get the raw_text and predictions as lists
    references = df['raw_text'].tolist()
    predictions = df['predictions'].tolist()

    # Compute WER using the compute_measures method
    measures = jiwer.compute_measures(references, predictions)

    # Return the WER score
    return measures['wer']

# Use the function
wer_score = calculate_wer("test.csv")
print("The WER score is:", wer_score)

