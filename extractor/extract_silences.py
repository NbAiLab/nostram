import argparse
import json
import os

from tqdm import tqdm

from extractor.detect_voice import VoiceDetector


def main(input_folder, out_file, min_duration):
    voice_detector = VoiceDetector(method="silero")

    with open(out_file, "w") as writer:
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(input_folder) for f in fn]

        for i, file in enumerate(files):
            if file.split(".")[-1] not in ("mp3", "mp4"):
                continue
            voice_detector.select_sourcefile(os.path.join(input_folder, file))
            voice_segments = voice_detector.analyze()
            start = 0
            end = voice_segments[-1]["end"]  # TODO use end of actual file?

            silence_segments = []

            last_voice = start
            for segment in voice_segments:
                if segment["start"] - last_voice > min_duration:
                    silence_segments.append({"start": last_voice, "end": segment["start"]})
                last_voice = segment["end"]

            for segment in silence_segments:
                writer.write(json.dumps({"file": file} | segment) + "\n")

            print(f"{i + 1} / {len(files)} files processed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--min_duration", default=0, type=int)
    args = parser.parse_args()

    main(args.input_folder, args.out_file, args.min_duration)
