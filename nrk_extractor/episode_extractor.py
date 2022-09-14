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

"""
META: https://psapi.nrk.no/programs/PRHO04004903

Playlist:https://nrk-od-51.akamaized.net/world/23451/3/hls/prho04004903/playlist.m3u8?bw_low=262&bw_high=2399&bw_start=886&no_iframes&no_audio_only&no_subtitles






https://psapi.nrk.no/playback/metadata/program/PRHO04004903?eea-portability=true

"""


class EpisodeExtractor():
    def resolve_urls(self, url_or_id):
        """
        Resolve NRK urls based on a url or an ID
        """
        print("url:" + str(url_or_id))
        if url_or_id.startswith("http"):
            r = re.match("https://.*/(.*)/avspiller", url_or_id)
            if r:
                id = r.groups()[0]
            else:
                raise Exception("Unknown URL, expect 'avspiller' URL")
        else:
            id = url_or_id

        res = {
            "id": id
        }

        # Fetch the info blob
        murl = "https://psapi.nrk.no/playback/metadata/program/%s?eea-portability=true" % id
        r = requests.get(murl)
        if r.status_code != 200:
            raise Exception("Failed to load metadata from '%s'" % murl)
        #print(r.json()["vtt"])
        res["info"] = json.loads(r.text)
        
        #Set medium
        ef=episodefetcher()
        #Try first for series, but if this does not exist, try for program
        try:
            res["info"]["medium"] = ef.getmedium(ef.getseries(id))
        except:
            res["info"]["medium"] = ef.getmediumprogram(id)

        #Set serie image
        try:
            res["info"]["serieimageurl"] = ef.getserieimage(ef.getseries(id))
        except:
            res["info"]["serieimageurl"] = "placeholder"

        if not bool(res["info"]["playable"]):
            res["info"] = "Unknown"

        if "playable" not in res["info"] or "resolve" not in res["info"]["playable"]:
            return None
            #raise Exception("Bad info block from '%s'" % murl)

        murl = "https://psapi.nrk.no" + res["info"]["playable"]["resolve"]
        r = requests.get(murl)
        if r.status_code != 200:
            raise Exception("Failed to load manifest from '%s'" % murl)

        res["manifest"] = json.loads(r.text)

        # Now we find the core playlist URL
        purl = res["manifest"]["playable"]["assets"][0]["url"]
        r = requests.get(purl)
        if r.status_code != 200:
            raise Exception("Failed to get playlist from '%s'" % purl)

        spec = None  # We take the first valid line
        for line in r.text.split("\n"):
            if line.startswith("#"):
                continue
            spec = line
            break

        if not spec:
            raise Exception("Failed to find valid specification in core m3u8")

        # We now have the correct url for downloading
        res["m3u8"] = os.path.split(purl)[0] + "/" + spec

        # We need the subtitles too
        if not res["manifest"]["playable"]["subtitles"]:
            # raise Exception("Missing subtitles")
            print("Missing subtitles for", id)
            res["vtt"] = None
        else:
            res["vtt"] = res["manifest"]["playable"]["subtitles"][0]["webVtt"]

        return res


    def extract_audio(self, info, target):

        if os.path.exists(target) and os.stat(target).st_size > 0:
            print("  Audio already present at '%s'" % target)
            return
        
        #if "m3u8" not in info:
        #    print("No playlist in info", info)
        #    raise Exception("Missing HLS playlist for '%s'" % target)

        #url = info["m3u8"].replace("no_audio_only", "audio_only")
        
        url = info["audio_file"]
        cmd = ["ffmpeg", "-y", "-i", url, "-vn", "-c:a", "copy", target]

        print("  Extracting audio to %s" % target)
        p = subprocess.run(cmd,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)

        if p.returncode != 0:
            raise Exception("Extraction failed '%s', code %s" % (cmd, p))


    def dump_at(self, info, target_dir):
        id = info['episode_id']

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        if not os.path.exists(target_dir+"/mp4"):
            os.makedirs(target_dir+"/mp4")
        
        if not os.path.exists(target_dir+"/mp3"):
            os.makedirs(target_dir+"/mp3")
        
        if not os.path.exists(target_dir+"/segments"):
            os.makedirs(target_dir+"/segments")

        target_audio = os.path.join(target_dir+"/mp4", "%s.mp4" % id)
        
        self.extract_audio(info, target_audio)
        
        destination = os.path.join(target_dir, 'segments', id+'.json')
       

        #subtitles = self.load_subtitles(info["subtitles"])
        subtitles = self.resync(target_audio,  options)
        #print(subtitles)
        #self.find_good_areas(subtitles)

        print("*** SAVING")
        self.save_jsonlines(subtitles, destination, info)

        return info

    @staticmethod
    def time_to_string(seconds):
        """
        Convert a number of minutes to days, hours, minutes, seconds
        """

        days = hours = minutes = secs = 0
        ret = ""

        if seconds == 0:
            return "0 seconds"

        secs = seconds % 60
        if secs:
            ret = "%d sec" % secs

        if seconds <= 60:
            return ret

        tmp = (seconds - secs) / 60
        minutes = tmp % 60

        if minutes > 0:
            ret = "%d min " % minutes + ret

        if tmp <= 60:
            return ret.strip()

        tmp = tmp / 60
        hours = tmp % 24
        if hours > 0:
            ret = "%d hours " % hours + ret

        if tmp <= 24:
            return ret.strip()

        days = tmp / 24

        return ("%d days " % days + ret).strip()

    def find_good_areas(self, subtitles):

        bad = 0
        playtime = 0
        for sub in subtitles.items:
            # If sub is two lines and BOTH start with "-" it's two different
            # people and hence the timing is shit
            lines = sub["text"].split("<br>")
            if len(lines) > 1:
                if lines[0].strip().startswith("—") and \
                   lines[1].strip().startswith("—"):
                    # print("BAD SUB", sub)
                    bad += 1
                    continue

            playtime += sub["end"] - sub["start"]

        print("  -- %d subtitles, %d are bad" % (len(subtitles.items), bad))
        print("  Playtime: %s" % NRKExtractor.time_to_string(playtime))

    def load_subtitles(self, subtitlefile, max_gap_s=0.4):
        subs = SubParser()
        subs.load_srt(subtitlefile)

        # We merge those with continuation mark '—'
        updated = []
        for item in subs.items:
            lines = item["text"].split("<br>")
            # If item has *3* lines, the first one is (likely) a person's name - remove it for now
            if len(lines) == 3:
                item["text"] = "<br>".join(lines[1:])

            # If both lines start with "—" it's two people - ignore
            if len(lines) == 2 and lines[0].startswith("—") and lines[1].startswith("—"):
                print("Two people, skipping")
                continue

            if len(updated) == 0:
                updated.append(item)
                continue

            if updated[-1]["text"][-1] == "—" and item["text"][0] == "—":
                # Merge
                updated[-1]["text"] = updated[-1]["text"][:-1] + item["text"][2:]
                updated[-1]["end"] = item["end"]
            elif 0 and item["start"] - updated[-1]["end"] < max_gap_s:  # Need to sync them first
                updated[-1]["text"] = updated[-1]["text"] + "<br>" + item["text"]
                updated[-1]["end"] = item["end"]
            else:
                updated.append(item)

        subs.items = updated
        return subs

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
                    "duration": int((item["end"] - item["start"])*100),
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

    print("\n\n* Preparing to process "+options.src)

    with jsonlines.open(options.src) as reader:
        for obj in reader:
            info = extractor.dump_at(obj, options.dst)
              

