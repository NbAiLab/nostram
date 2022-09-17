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

    def __init__(self, method=None) -> None:
        self.detector = VoiceDetector(method=method)

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
        audio_segment_filename = f"{entry['episode_id']}_{entry['start_time_ms']:010d}_{entry['end_time_ms']:010d}_{entry['duration_ms']}.wav"
        audio_segment_dir = audio_segments_dir + f"/{entry['episode_id']}"

        if not os.path.exists(audio_segment_dir):
            os.makedirs(audio_segment_dir)

        audio_segment_destinaton = os.path.join(
            audio_segment_dir, audio_segment_filename
        )
        self.extract_audio(
            audio,
            audio_segment_destinaton,
            start=entry['start_time_ms'],
            end=entry['end_time_ms']
        )

    def dump_at(self, info, target_dir, extract_audio_segments=False):
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
        if extract_audio_segments and not os.path.exists(audio_segments_dir):
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
        
        segments, target_audio_wav = self.resync(target_audio,  options)
        self.save_jsonlines(segments, target_segment, info, target_audio_wav, audio_segments_dir, extract_audio_segments)
        
        #Remove tmp-file
        os.remove(target_audio_wav)

        return info

    def resync(self, audiofile, options, max_gap_s=0.5):
        self.detector.select_sourcefile(audiofile)
        segments = self.detector.analyze(
            aggressive=options.aggressive,
            max_pause=float(options.max_pause),
            max_segment_length=float(options.max_segment_length),
            threshold=float(options.silero_threshold)
        )
        print(f"Detected {len(segments)} audio segments.")
        return segments, self.detector.sourcefile

    def save_jsonlines(self, segments, destination, info, audio, audio_segments_dir, extract_audio_segments=False):
        # Save segments
        with open(destination, "w") as destination_file:
            # Write a single line pr entry that's good
            if extract_audio_segments:
                segments_iter = tqdm(
                    segments, desc="Saving audio segments", total=len(segments)
                )
            else:
                segments_iter = segments
            for item in segments_iter:
                entry = {
                    "id": info["episode_id"] + "_" + str(int(item["start"] * 1000)) + "_" + str(int(item["end"] * 1000)),
                    "start_time_ms": int(item["start"] * 1000),
                    "end_time_ms": int(item["end"] * 1000),
                    "duration_ms": int((item["end"] - item["start"]) * 1000),
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
                if extract_audio_segments:
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
    parser.add_argument("--vad_method", dest="vad_method", help="VAD method. Either 'silero' for SileroVAD or 'webrtc' for WebRTCVad (s)", default="webrtc")
    parser.add_argument("--silero_threshold", dest="silero_threshold", help="SileroVAD threshold to consider a segment speeech", default=0.75)
    parser.add_argument("-e", "--extract_audio_segments", dest="extract_audio_segments", help="Extract audio segments",
                        action="store_true", default=False)

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
    options.silero_threshold = float(options.silero_threshold)

    extractor = EpisodeExtractor(method=options.vad_method)

    print("\n\n* Starting processing " + options.src)

    with jsonlines.open(options.src) as reader:
        for obj in reader:
            info = extractor.dump_at(obj, options.dst, options.extract_audio_segments)


