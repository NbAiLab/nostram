import json
from sacrebleu import corpus_bleu
from argparse import ArgumentParser

def calculate_bleu(predictions, targets):
    bleu = corpus_bleu(predictions, [targets])
    return bleu.score

def main(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    predictions = []
    targets = []

    for line in lines:
        data = json.loads(line)
        predictions.append(data['prediction'])
        targets.append(data['target'])

    bleu_score = calculate_bleu(predictions, targets)
    print(f"SacreBLEU Score: {bleu_score:.2f}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Calculate SacreBLEU score between predictions and targets in a JSON Lines file.")
    parser.add_argument("input_file", help="Path to the input JSON Lines file.")
    args = parser.parse_args()
    main(args.input_file)

