import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torchaudio
import tqdm
from torch.utils.data import IterableDataset
from transformers import pipeline

model_id = "facebook/mms-lid-2048"

classifier = pipeline("audio-classification", model=model_id, device="cuda:1" if torch.cuda.is_available() else "cpu")


class AudioDataset(IterableDataset):
    def __init__(self, audio_files):
        self.audio_files = audio_files

    def __iter__(self):
        def load(audio_file):
            # Load the audio file
            audio, orig_freq = torchaudio.load(audio_file)

            # Convert the audio to a 16kHz sample rate
            audio = torchaudio.transforms.Resample(orig_freq, 16000)(audio)

            # Convert the audio to mono
            audio = audio.mean(dim=0)

            # Yield the processed audio
            return audio.numpy()

        with ThreadPoolExecutor() as ex:
            yield from ex.map(load, self.audio_files)


def main(audio_folder, output_file):
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(audio_folder) for f in fn
             if f.endswith(".mp3") or f.endswith(".wav") or f.endswith(".mp4")]

    dataset = AudioDataset(files)

    it = tqdm.tqdm(files)
    with open(output_file, "w") as writer:
        writer.write("file,lang,confidence\n")

        predictions = classifier(dataset, chunk_length_s=30, stride_length_s=5, batch_size=16)

        for file, detected_langs in zip(it, predictions):
            detected_lang = max(detected_langs, key=lambda x: x["score"])
            confidence = detected_lang["score"]
            label = detected_lang["label"]

            fname = os.path.basename(file)
            writer.write(f"{fname},{label},{confidence:.6f}\n")
            writer.flush()
            it.set_postfix_str(f"{fname}, {label}, {confidence:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_folder")
    parser.add_argument("--output_file")

    args = parser.parse_args()

    main(
        audio_folder=args.audio_folder,
        output_file=args.output_file
    )
