import argparse
import os

from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch

from transformers import pipeline

model_id = "facebook/mms-lid-2048"

classifier = pipeline("audio-classification", model=model_id, device="cuda" if torch.cuda.is_available() else "cpu")


def main(audio_folder, output_file):
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(audio_folder) for f in fn]

    with open(output_file, "w") as writer:
        writer.write("file,lang,confidence\n")
        for file in files:
            print(file, end="")
            detected_langs = classifier(file, chunk_length_s=30, stride_length_s=5)
            detected_lang = max(detected_langs, key=lambda x: x["score"])
            confidence = detected_lang["score"]
            label = detected_lang["label"]
            print(",", detected_langs)

            writer.write(f"{os.path.basename(file)},{label},{confidence}\n")
            writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_folder")
    parser.add_argument("--output_file")

    args = parser.parse_args()

    main(
        audio_folder=args.audio_folder,
        output_file=args.output_file
    )
