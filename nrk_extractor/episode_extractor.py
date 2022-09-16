#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
NORCE Research Institute 2022, Njaal Borch <njbo@norceresearch.no>
Licensed under Apache 2.0

Modified by Freddy Wetjen, Per Egil Kummervold, and Javier de la Rosa
National Library of Norway
"""
import json
import os
import subprocess
from argparse import ArgumentParser

import jsonlines
from detect_voice import VoiceDetector
from tqdm import tqdm


class EpisodeExtractor():

    def extract_audio(self, url, target, start=None, end=None):
        if os.path.exists(target) and os.stat(target).st_size > 0:
            if start is None or end is None:
                print("Audio already present at '%s'" % target)
            return

        if start is None or end is None:
            cmd = ["ffmpeg", "-y", "-i", url, "-vn", "-c:a", "copy", target]
            print(f"Extracting audio to %s" % target)
        else:
            cmd = ["ffmpeg", "-y", "-i", url, "-ss", f"{start}ms", "-to", f"{end}ms", "-vn", "-c:a", "copy", target]

        p = subprocess.run(cmd,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)

        if p.returncode != 0:
            raise Exception("Extraction failed '%s', code %s" % (cmd, p))

    def extract_segment(self, entry, audio, audio_segments_dir):
        audio_segment_filename = f"{entry['episode_id']}_{entry['start_time']}_{entry['end_time']}_{entry['duration_ms']}.wav"
        audio_segment_dir = audio_segments_dir + f"/{entry['episode_id']}"

        if not os.path.exists(audio_segment_dir):
            os.makedirs(audio_segment_dir)

        audio_segment_destinaton = os.path.join(
            audio_segment_dir, audio_segment_filename
        )
        self.extract_audio(
            audio,
            audio_segment_destinaton,
            start=entry['start_time'],
            end=entry['end_time']
        )

    def dump_at(self, info, target_dir):
        id = info['episode_id']

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        audio_dir = target_dir + "/audio"
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)

        segments_dir = target_dir + "/segments"
        if not os.path.exists(segments_dir):
            os.makedirs(segments_dir)

        audio_segments_dir = target_dir + "/audio_segments"
        if not os.path.exists(audio_segments_dir):
            os.makedirs(audio_segments_dir)

        if info['audio_format'] == "HLS":
            audio_file_name = f"{id}.mp4"
        elif info['audio_format'] == "MP3":
            audio_file_name = f"{id}.mp3"
        else:
            print("Error: Unknown Audio Format")
            exit(-1)

        target_audio = os.path.join(target_dir, "audio", audio_file_name)
        target_segment = os.path.join(target_dir, 'segments', id + '.json')

        self.extract_audio(info["audio_file"], target_audio)

        subtitles = self.resync(target_audio,  options)
        self.save_jsonlines(subtitles, target_segment, info, target_audio, audio_segments_dir)

        return info

    def resync(self, audiofile, options, max_gap_s=0.5):
        detector = VoiceDetector(audiofile)
        segments = detector.analyze(aggressive=options.aggressive,
                                    max_pause=float(options.max_pause),
                                    max_segment_length=float(options.max_segment_length))

        return segments

    def save_jsonlines(self, segments, destination, info, audio, audio_segments_dir):
        # Save segments
        with open(destination, "w") as destination_file:
            # Write a single line pr entry that's good
            segments_iter = tqdm(
                segments, desc="Saving audio segments", total=len(segments)
            )
            for item in segments_iter:
                entry = {
                    "id": info["episode_id"] + "_" + str(int(item["start"] * 100)) + "_" + str(int(item["end"] * 100)),
                    "start_time": int(item["start"]*100),
                    "end_time": int(item["end"]*100),
                    "duration_ms": int((item["end"] - item["start"])*100),
                    'episode_id': info['episode_id'],
                    'medium': info['medium'],
                    'program_image_url': info['program_image_url'],
                    'serie_image_url': info['serie_image_url'],
                    'title': info['title'],
                    'sub_title': info['subtitle'],
                    'year': info['year'],
                    'availability_information': info['availability_information'],
                    'is_geoblocked': info['is_geoblocked'],
                    'external_embedding_alowed': info['external_embedding_allowed'],
                    'on_demand_from': info['on_demand_from'],
                    'on_demand_to': info['on_demand_to'],
                    'audio_file': info['audio_file'],
                    'audio_format': info['audio_format'],
                    'audio_mime_type': info['audio_mime_type']
                }
                destination_file.write(json.dumps(entry) + "\n")
                self.extract_segment(entry, audio, audio_segments_dir)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-s", "--source", dest="src", help="Json-file with episodes and meta-data", required=True)

    parser.add_argument("-d", "--destination", dest="dst",
                        help="Destination folder (new subfolder will be made)",
                        default="/dump/")

    parser.add_argument("-i", "--info", dest="infoonly", help="Only show info",
                        action="store_true", default=False)

    parser.add_argument("-a", "--aggressive", dest="aggressive", help="How aggressive (0-3, 3 is most aggressive), default 1", default=1)
    parser.add_argument("--min_cps", dest="min_cps", help="Minimum CPS", default=0)
    parser.add_argument("--max_cps", dest="max_cps", help="Maximum CPS", default=0)
    parser.add_argument("--max_pause", dest="max_pause", help="Merge if closer than this (if >0s)", default=0)
    parser.add_argument("--max_segment_length", dest="max_segment_length", help="Max segment length", default=30)
    parser.add_argument("--max_adjust", dest="max_adjust", help="Maximum adjustment (s)", default=1.0)
    parser.add_argument("--min_time", dest="min_time", help="Minimium time for a sub (s)", default=1.2)

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except Exception:
        pass  # Ignore

    options = parser.parse_args()

    options.min_cps = float(options.min_cps)
    options.max_cps = float(options.max_cps)
    options.max_adjust = float(options.max_adjust)
    options.min_time = float(options.min_time)

    extractor = EpisodeExtractor()

    print("\n\n* Starting processing " + options.src)

    with jsonlines.open(options.src) as reader:
        for obj in reader:
            info = extractor.dump_at(obj, options.dst)


