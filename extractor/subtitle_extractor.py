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
import glob
from tqdm import tqdm
import random

"""
META: https://psapi.nrk.no/programs/PRHO04004903

Playlist:https://nrk-od-51.akamaized.net/world/23451/3/hls/prho04004903/playlist.m3u8?bw_low=262&bw_high=2399&bw_start=886&no_iframes&no_audio_only&no_subtitles






https://psapi.nrk.no/playback/metadata/program/PRHO04004903?eea-portability=true

"""


class NRKExtractor():
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

        # Read info file
        if os.path.isfile(options.dst+"/info/"+id+"_info.json"):
            with open(options.dst+"/info/"+id+"_info.json", 'r') as openfile:
                res["info"] = json.load(openfile)
            
        else:
            print(f'  Fetching info-file from NRK.')
            # Fetch the info blob
            murl = "https://psapi.nrk.no/playback/metadata/program/%s?eea-portability=true" % id
            r = requests.get(murl)
            
            if r.status_code != 200:
                raise Exception("Failed to load metadata from '%s'" % murl)
            #print(r.json()["vtt"])
            res["info"] = json.loads(r.text)
                    
            # Save
            with open(options.dst+"/info/"+id+"_info.json", "w") as outfile:
                json.dump(res["info"], outfile)
            
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

        # Handle non-playable files
        if not bool(res["info"]["playable"]):
            res["info"] = res["info"]["nonPlayable"]["endUserMessage"]
            with open(options.dst+"/unavailable/"+id+".json", "w") as creating_empty_json_file: 
                pass 

        if "playable" not in res["info"] or "resolve" not in res["info"]["playable"]:
            print(res["info"])
            return None
            #raise Exception("Bad info block from '%s'" % murl)
        

        # Read manifest file
        if os.path.isfile(options.dst+"/manifest/"+id+"_manifest.json"):
            print(f'  Found manifest-file locally.')
            with open(options.dst+"/manifest/"+id+"_manifest.json", 'r') as openfile:
                res["manifest"] = json.load(openfile)
            
        else:
            print(f'  Fetching manifest-file from NRK. This should not happen now.')
            
            # Added manually to prevent NRK 
            raise Exception("This should not happen now! It tries to contact NRK here.")
            
            # Fetch the manifest blob
            murl = "https://psapi.nrk.no" + res["info"]["playable"]["resolve"]
            r = requests.get(murl)
            if r.status_code != 200:
                raise Exception("Failed to load manifest from '%s'" % murl)

            res["manifest"] = json.loads(r.text)
                    
            # Save
            with open(options.dst+"/manifest/"+id+"_manifest.json", "w") as outfile:
                json.dump(res["manifest"], outfile)

        # Now we find the core playlist URL
        if os.path.isfile(options.dst+"/playlist/"+id+"_playlist.json"):
            print(f'  Found playlist-file locally.')
            with open(options.dst+"/playlist/"+id+"_playlist.json", 'r') as openfile:
                res["m3u8"] = json.load(openfile)["m3u8"]
            
        else:
            print(f'  Fetching playlist-file from NRK.')
            
            # Fetch the playlist blob
            purl = res["manifest"]["playable"]["assets"][0]["url"]
            r = requests.get(purl)
            
            if r.status_code != 200:
                raise Exception("  Failed to get playlist from '%s'" % purl)

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
            
            # Save
            with open(options.dst+"/playlist/"+id+"_playlist.json", "w") as outfile:
                json.dump({"m3u8": res["m3u8"]}, outfile)


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

        if "m3u8" not in info:
            print("No playlist in info", info)
            raise Exception("Missing HLS playlist for '%s'" % target)

        url = info["m3u8"].replace("no_audio_only", "audio_only")
        cmd = ["ffmpeg", "-y", "-i", url, "-vn", "-c:a", "copy", target]

        print("  Extracting audio to %s" % target)
        p = subprocess.run(cmd,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)

        if p.returncode != 0:
            raise Exception("Extraction failed '%s', code %s" % (cmd, p))

    def extract_vtt(self, info, target):

        if os.path.exists(target) and os.stat(target).st_size > 0:
            print("  Subtitle already present at '%s'" % target)
            return
        #print(info)
        #exit(-1)
        if "vtt" not in info or info["vtt"] == None:
            print("VTT url" + str(target))
            print ("Missing VTT url for '%s'" % target)
            #raise Exception("Missing VTT url for '%s'" % target)
            return

        print("  Extracting subtitles to %s" % target)
        r = requests.get(info["vtt"])
        if r.status_code != 200:
            raise Exception("Failed to download vtt from '%s'" % info["vtt"])
        with open(target, "w") as f:
            f.write(r.text)

    def dump_at(self, id, target_dir):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        if not os.path.exists(target_dir+"/audio"):
            os.makedirs(target_dir+"/audio")
        
        if not os.path.exists(target_dir+"/"+self.vtt_folder):
            os.makedirs(target_dir+"/"+self.vtt_folder)
        
        if not os.path.exists(target_dir+"/subtitles_"+self.vtt_folder):
            os.makedirs(target_dir+"/subtitles_"+self.vtt_folder)
            
        if not os.path.exists(target_dir+"/segments"):
            os.makedirs(target_dir+"/segments")

        if not os.path.exists(target_dir+"/manifest"):
            os.makedirs(target_dir+"/manifest")

        if not os.path.exists(target_dir+"/info"):
            os.makedirs(target_dir+"/info")

        if not os.path.exists(target_dir+"/playlist"):
            os.makedirs(target_dir+"/playlist")    
        
        if not os.path.exists(target_dir+"/unavailable"):
            os.makedirs(target_dir+"/unavailable") 
        
        start_time = time.time()
        print("Processing", id)
        info = self.resolve_urls(id)
        
        #Debug
        if not info:
            return None


        target_audio = os.path.join(target_dir+"/audio", "%s.mp4" % id)
        target_vtt = os.path.join(target_dir+"/"+self.vtt_folder, "%s.vtt" % id)

        self.extract_audio(info, target_audio)
        self.extract_vtt(info, target_vtt)

        elapsed = time.time() - start_time
        print("  OK in ", NRKExtractor.time_to_string(elapsed))

        info["audio"] = target_audio
        info["subtitles"] = target_vtt
        info["elapsed"] = elapsed
        info["vtt_folder"] = self.vtt_folder

        #print("Audio file", info["audio"])
        #print("Subtitle file", info["subtitles"])

        subtitles_destination = os.path.splitext(info["audio"])[0].replace("/audio","/segments") + ".json"
        
        #if os.path.exists(subtitles_destination):
        #    return info
        subtitles = self.load_subtitles(info["subtitles"])
        subtitles = self.resync(info["audio"], subtitles,  options)
        #print(subtitles)
        # self.find_good_areas(subtitles)

        print("*** SAVING")
        self.save_jsonlines(subtitles, subtitles_destination, info)

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

    # def find_good_areas(self, subtitles):

    #     bad = 0
    #     playtime = 0
    #     for sub in subtitles.items:
    #         # If sub is two lines and BOTH start with "-" it's two different
    #         # people and hence the timing is shit
    #         lines = sub["text"].split("<br>")
    #         if len(lines) > 1:
    #             if lines[0].strip().startswith("—") and \
    #                lines[1].strip().startswith("—"):
    #                 # print("BAD SUB", sub)
    #                 bad += 1
    #                 continue

    #         playtime += sub["end"] - sub["start"]

    #     print("  -- %d subtitles, %d are bad" % (len(subtitles.items), bad))
    #     print("  Playtime: %s" % NRKExtractor.time_to_string(playtime))

    def load_subtitles(self, subtitlefile, max_gap_s=0.4):
        subs = SubParser()
        subs.load_srt(subtitlefile)

        # We merge those with continuation mark '—'
        updated = []
        for item in subs.items:
            lines = item["text"].split("<br>")
            # If item has *3* lines, the first one is (likely) a person's name - remove it for now
            #if len(lines) == 3:
            #    item["text"] = "<br>".join(lines[1:])

            # If both lines start with "—" it's two people - ignore
            # if len(lines) == 2 and lines[0].startswith("—") and lines[1].startswith("—"):
            #    print("Two people, skipping")
            #    continue

            if len(updated) == 0:
                updated.append(item)
                continue
            try:
                #if updated[-1]["text"][-1] == "—" and item["text"][0] == "—":
                #    # Merge
                #    updated[-1]["text"] = updated[-1]["text"][:-1] + item["text"][2:]
                #    updated[-1]["end"] = item["end"]
                if 0 and item["start"] - updated[-1]["end"] < max_gap_s:  # Need to sync them first
                    updated[-1]["text"] = updated[-1]["text"] + "<br>" + item["text"]
                    updated[-1]["end"] = item["end"]
                else:
                    updated.append(item)
            except:
                print("Index out of range")
            
        subs.items = updated
        return subs

    def resync(self, audiofile, subtitles, options, max_gap_s=0.5):
        detector = VoiceDetector('silero')
        
        detector.select_sourcefile(audiofile)
        
        segments = detector.analyze(aggressive=options.aggressive,
                                    max_pause=float(options.max_pause),
                                    max_segment_length=float(options.max_segment_length))

        subtitles.items = detector.realign_subs(segments, subtitles.items, options)
        

        # Merge too close ones?
        if 0:
            updated = []
            for item in subtitles.items:
                if len(updated) == 0:
                    updated.append(item)
                    continue
                if item["start"] - updated[-1]["end"] < max_gap_s:
                    updated[-1]["text"] = updated[-1]["text"] + "<br>" + item["text"]
                    updated[-1]["end"] = item["end"]
                else:
                    updated.append(item)

            subtitles.items = updated

        # Adding the segments to the subtitles object
        subtitles.segments = segments


        return subtitles
    
    def save_jsonlines(self, subtitles, destination, info):
        def build_entry(item, info):
            entry = {
                    "id": info["id"]+"_"+str(int(item["start"]*1000))+"_"+str(int(item["end"]*1000)),
                    "start_time": int(item["start"]*1000),
                    "end_time": int(item["end"]*1000),
                    "duration": int((item["end"] - item["start"])*1000),
                    "program_id": info["id"],
                    "medium": info["info"]["medium"],
                    "serieimageurl" : info["info"]["serieimageurl"],
                    "programimageurl" : info["info"]["preplay"]["poster"]["images"][0]["url"],
                    "source": "NRK TV",
                    "category": info["manifest"]["statistics"]["luna"]["data"]["category"],
                    "title": info["info"]["preplay"]["titles"]["title"],
                    "availability_information": info["info"]["availability"]["information"],
                    "is_geoblocked":info["info"]["availability"]['isGeoBlocked'],
                    "on_demand_from":info["info"]["availability"]["onDemand"]["from"],
                    "on_demand_to":info["info"]["availability"]["onDemand"]["to"],
                    "external_embedding_allowed":info["info"]["availability"]['externalEmbeddingAllowed'],
                    "subtitle": info["info"]["preplay"]["titles"]["subtitle"],
                    "audio": os.path.basename(info["audio"]),
                    "vtt_folder": info["vtt_folder"]
                }
            return entry

        # Save segments
        with open(destination, "w") as f:
            # Write a single line pr entry that's good
            for idx, item in enumerate(subtitles.segments):
                entry = build_entry(item,info)
                f.write(json.dumps(entry) + "\n")
        
        # Save subtitles
        with open(destination.replace("segments","/subtitles_"+self.vtt_folder).replace(".json","_subtitles.json"), "w") as f:
            # Write a single line pr entry that's good
            for idx, item in enumerate(subtitles.items):
                entry = build_entry(item,info)
                #item["text"] = item["text"].replace("<br>"," ").replace("\t"," ").replace("\n"," ").replace("\r"," ")
                item["text"] = " ".join(item["text"].split())
                sub = {"text": item["text"]}
                entry = {**entry,**sub}
                if item["text"] != "":
                    f.write(json.dumps(entry) + "\n")
        
    vtt_folder = "vtt"


if __name__ == "__main__":

    parser = ArgumentParser()
    
    parser.add_argument("-f", "--flist", dest="flist", help="Processes all files in vtt folder. Set to True to process", default=False)
    
    parser.add_argument("-s", "--source", dest="src", help="Source url or NRK program ID")

    parser.add_argument("-d", "--destination", dest="dst",
                        help="Destination folder (new subfolder will be made)",

                        default="/data/nrk/")
    parser.add_argument("-v", "--vtt_folder", dest="vtt_folder",
                        help="Folder to look for vtt files. Written to metadata",
                        default="vtt")

    parser.add_argument("-i", "--info", dest="infoonly", help="Only show info",
                        action="store_true", default=False)

    parser.add_argument("-r", "--multi", dest="num_episodes",
                        help="If given, how many next episodes to download (if available)",
                        default=1)

    parser.add_argument("-t", "--total", dest="all_episodes",
                        help="Downloads all episodes in all seasons in the entire serie",
                        action="store_true", default=False)
                    
    parser.add_argument("-a", "--aggressive", dest="aggressive", help="How aggressive (0-3, 3 is most aggressive), default 1", default=1)
    parser.add_argument("-e", "--error_halt", dest="error_halt", help="Halt on errors. Stops executing and gives real error messages.", default=False)
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

    extractor = NRKExtractor()
    extractor.vtt_folder = options.vtt_folder
    
    if options.infoonly:
        info = extractor.resolve_urls(options.src)

        #print(json.dumps(info, indent=" "))
        raise SystemExit(0)

    def process_subtitle(options):
        result = True
        
        if os.path.isfile(options.dst+"/subtitles_"+options.vtt_folder+"/"+options.src+"_subtitles.json"):
                print(f'{(options.dst+"/subtitles_"+options.vtt_folder+"/"+options.src+"_subtitles.json")} has already been processed.')
        elif os.path.isfile(options.dst+"/unavailable/"+options.src+".json"):
                print(f'{(options.dst+"/unavailable/"+options.src+".json")} is not playable.')       
        else:
            print("\n\n* Preparing to process "+options.src)

            if options.all_episodes and int(options.num_episodes)>1:
                print("Please do not use the -r and the -t option together.")
                raise SystemExit(0)

            

            
            if options.all_episodes:
                ef=episodefetcher()
                ef.episodebuilder(options.src)
                
                for i in ef.episodegenerator():
                    info = extractor.dump_at(i, options.dst)
                
                next_id = None

            else:
                if options.error_halt == False or options.error_halt == 0: 
                #If not all
                    try:
                        info = extractor.dump_at(options.src, options.dst)
                    except:
                        print("**************************************")
                        print(f"**** Failed to process {options.src} **")
                        print("**************************************")
                        result = False
                else:
                    info = extractor.dump_at(options.src, options.dst)
                
                try:
                    next_id = info["info"]["_embedded"]["next"]["id"]
                    for i in range(1, int(options.num_episodes)):
                        info = extractor.dump_at(next_id, options.dst)
                        next_id = info["info"]["_embedded"]["next"]["id"]
                except Exception:
                    next_id = None    
        return result    



    if options.flist=="True" or options.flist=="1":
        dir_path = f'{options.dst}/{options.vtt_folder}/*.vtt'
        res = glob.glob(dir_path)
        random.shuffle(res)
        
        success = 0
        failure = 0
        for f in tqdm(res):
            options.src = os.path.basename(f).replace(".vtt","")
            result = process_subtitle(options)
            if result:
                success += 1
            else:
                failure +=1
        print(f"\n\nFinished. \nSuccess: {success}\nFailure: {failure}\nTotal: {failure + success}")
        
        unavailable_dir = f'{options.dst}/unavailable'
        num_unavailable = (len([entry for entry in os.listdir(unavailable_dir) if os.path.isfile(os.path.join(unavailable_dir, entry))]))
        print(f"\nIn addition {num_unavailable} files are marked as \"unavailable\".")
        
    else:
        process_subtitle(options)
    
        
 