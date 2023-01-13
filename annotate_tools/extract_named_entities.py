import argparse
import time

import pandas as pd
import torch.cuda
from transformers import pipeline


def main(input_file, output_file, model, batch_size, separator):
    print("Starting...")
    df = pd.read_json(input_file, lines=True, nrows=10_000)

    model = pipeline("ner", model,
                     device="cuda" if torch.cuda.is_available() else "cpu",
                     aggregation_strategy="average",
                     max_seq_len=512)

    entity_groups = set(label.replace("B-", "").replace("I-", "") for label in model.model.config.label2id) - {"O"}
    df[[f"named-entities_{eg}" for eg in sorted(entity_groups)]] = ""

    print("Adding NER predictions...")
    t0 = time.perf_counter()
    for i in range(0, len(df), batch_size):
        texts = df.iloc[i:i + batch_size].text.tolist()
        results = model(texts)
        for j, res in enumerate(results):
            group_words = {}
            for entry in res:
                group = entry["entity_group"]
                word = entry["word"]
                group_words.setdefault(group, set()).add(word)

            for group, words in group_words.items():
                df.iloc[i + j, df.columns.get_loc(f"named-entities_{group}")] = separator.join(sorted(words))

    t1 = time.perf_counter()
    df.to_json(output_file, lines=True, orient="records")
    print(f"Done! Took {t1 - t0} seconds ({len(df) / (t1 - t0):.3g} samples/second)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--separator", type=str, default="|")
    args = parser.parse_args()

    main(
        input_file=args.input_file,
        output_file=args.output_file,
        model=args.model,
        batch_size=args.batch_size,
        separator=args.separator
    )
