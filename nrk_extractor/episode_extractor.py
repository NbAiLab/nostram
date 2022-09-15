#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

"""
NORCE Research Institute 2022, Njaal Borch <njbo@norceresearch.no>
Licensed under Apache 2.0

Modified by Freddy Wetjen and Per Egil Kummervold
National Library of Norway
"""

import requests
import json
import re
import os
import subprocess
import time
import sys

from argparse import ArgumentParser
from detect_voice import VoiceDetector
from sub_parser import SubParser
from fetch_episodes import episodefetcher
import jsonlines


class EpisodeExtractor():

    def extract_audio(self, info, target):
        if os.path.exists(target) and os.stat(target).st_size > 0:
            print("Audio already present at '%s'" % target)
            return
        
        url = info["audio_file"]
        cmd = ["ffmpeg", "-y", "-i", url, "-vn", "-c:a", "copy", target]

        print("Extracting audio to %s" % target)
        p = subprocess.run(cmd,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)

        if p.returncode != 0:
            raise Exception("Extraction failed '%s', code %s" % (cmd, p))


    def dump_at(self, info, target_dir):
        id = info['episode_id']

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        if not os.path.exists(target_dir+"/audio"):
            os.makedirs(target_dir+"/audio")
        
        if not os.path.exists(target_dir+"/segments"):
            os.makedirs(target_dir+"/segments")
        
        if info['audio_format'] == "HLS":
            audio_file_name = f"{id}.mp4"
        elif info['audio_format'] == "MP3":
            audio_file_name = f"{id}.mp3"
        else:
            print("Error: Unknown Audio Format")
            exit(-1)

        target_audio = os.path.join(target_dir+"/audio", audio_file_name)
        
        self.extract_audio(info, target_audio)
        
        destination = os.path.join(target_dir, 'segments', id+'.json')
       
        subtitles = self.resync(target_audio,  options)
        self.save_jsonlines(subtitles, destination, info)

        return info

    def resync(self, audiofile, options, max_gap_s=0.5):
        detector = VoiceDetector(audiofile)
        segments = detector.analyze(aggressive=options.aggressive,
                                    max_pause=float(options.max_pause),
                                    max_segment_length=float(options.max_segment_length))
        
        return segments
    
    def save_jsonlines(self, segments, destination, info):
        def build_entry(item, info):
            entry = {
                    "id": info["episode_id"]+"_"+str(int(item["start"]*100))+"_"+str(int(item["end"]*100)),
                    "start_time": int(item["start"]*100),
                    "end_time": int(item["end"]*100),
                    "duration_ms": int((item["end"] - item["start"])*100),
                    'episode_id':info['episode_id'], 
                    'medium': info['medium'], 
                    'program_image_url': info['program_image_url'], 
                    'serie_image_url':info['serie_image_url'],
                    'title':info['title'],
                    'sub_title': info['subtitle'], 
                    'year':info['year'],
                    'availability_information':info['availability_information'],
                    'is_geoblocked':info['is_geoblocked'],
                    'external_embedding_alowed':info['external_embedding_allowed'],
                    'on_demand_from':info['on_demand_from'],
                    'on_demand_to':info['on_demand_to'],
                    'audio_file':info['audio_file'],
                    'audio_format':info['audio_format'],
                    'audio_mime_type':info['audio_mime_type']}

            return entry

        # Save segments
        with open(destination, "w") as f:
            # Write a single line pr entry that's good
            for idx, item in enumerate(segments):
                entry = build_entry(item,info)
                f.write(json.dumps(entry) + "\n")

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

    print("\n\n* Starting processing "+options.src)

    with jsonlines.open(options.src) as reader:
        for obj in reader:
            info = extractor.dump_at(obj, options.dst)
              

