#!/usr/bin/env python3

"""

NORCE Research Institute 2022, Njaal Borch <njbo@norceresearch.no>
Licensed under GPL v3

"""


import contextlib
import sys
import wave
import json
import webrtcvad
import operator
from argparse import ArgumentParser
import tempfile
import os


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class VoiceDetector:

    def __init__(self, sourcefile, output_dir=None):

        self.is_tmp = False
        self.output_dir = output_dir

        if not sourcefile.endswith(".wav"):
            self.is_tmp = True
            self.sourcefile = self.convert(sourcefile)
        else:
            self.sourcefile = sourcefile

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def __del__(self):
        if self.is_tmp:
            os.remove(self.sourcefile)

    def analyze(self, aggressive=2, max_segment_length=8, max_pause=0, frame_length=10):
        audio, sample_rate = self.read_wave(self.sourcefile)
        vad = webrtcvad.Vad(int(aggressive))
        frames = self.frame_generator(frame_length, audio, sample_rate)
        frames = list(frames)
        segments = self.vad_collector(sample_rate, frame_length, 3, vad, frames,
                                      output_dir=self.output_dir,
                                      max_segment_length=max_segment_length,
                                      max_pause=max_pause)

        return segments

    def read_wave(self, path):
        """Reads a .wav file.

        Takes the path, and returns (PCM audio data, sample rate).
        """
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000, 48000)
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate

    def frame_generator(self, frame_duration_ms, audio, sample_rate):
        """Generates audio frames from PCM audio data.

        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.

        Yields Frames of the requested duration.
        """
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(self, sample_rate, frame_duration_ms,
                      padding_frames, vad, frames,
                      output_dir=None, max_segment_length=None, max_pause=0):
        """
        If output dir is given, speech audio segments are saved there
        """
        triggered = False
        segments = []
        voiced_frames = []
        segment_data = []
        frames_speech = 0
        frames_audio = 0
        padding = start = end = 0
        for idx, frame in enumerate(frames):
            is_speech = vad.is_speech(frame.bytes, sample_rate)
            if is_speech:
                frames_speech += 1
            else:
                frames_audio += 1

            if is_speech:
                segment_data.append(frame)
            if not triggered and is_speech:
                triggered = True
                start = idx * frame_duration_ms
                # segment_data = [frame]
                # if start == 0:
                #    raise SystemExit("What, starts with voice (%d)?" % idx)

            elif triggered and (not is_speech or \
                  (idx * frame_duration_ms) - start > max_segment_length * 1000):
                if padding < padding_frames:
                    padding += 1
                    continue
                triggered = False
                end = idx * frame_duration_ms

                s = {"type": "voice", "start": start / 1000., "end": end / 1000., "idx": idx}

                if output_dir:
                    target = os.path.join(output_dir, "segment_%08d.wav" % idx)

                merged = False
                if max_pause and len(segments) > 0:
                    if s["start"] - segments[-1]["end"] < max_pause and \
                     s["end"] - segments[-1]["start"] < max_segment_length:

                        # Only merge if the last segment is too short
                        if segments[-1]["end"] - segments[-1]["start"] < 4.0:
                            merged = True
                            # MERGE
                            # print("MERGING", segments[-1]["end"], s["start"], segments[-1]["idx"])
                            segments[-1]["end"] = s["end"]
                            # We should overwrite the last file if output_dir is given!
                            if output_dir:
                                target = segments[-1]["file"]
                                segment_data = segments[-1]["data"] + segment_data
                            s = segments[-1]

                # Save the audio segment if requested
                if output_dir:
                    # target = os.path.join(output_dir, "segment_%08d.wav" % idx)
                    with wave.open(target, "w") as target_f:
                        target_f.setnchannels(1)
                        target_f.setsampwidth(2)
                        target_f.setframerate(sample_rate)
                        for d in segment_data:
                            target_f.writeframes(d.bytes)
                    s["file"] = target
                    s["data"] = segment_data

                if not merged:
                    segments.append(s)
                start = end = padding = 0
                segment_data = []
            elif triggered and is_speech:
                padding = 0
        
        for s in segments:
            if "data" in s:
                del s["data"]
        return segments

    def convert(self, mp3file):

        import subprocess
        fd, tmpfile = tempfile.mkstemp(suffix=".wav")
        print("Extracting audio to", tmpfile)
        cmd = ["ffmpeg", "-i", mp3file, "-vn", "-ac", "1", "-y", tmpfile]
        print(" ".join(cmd))
        s = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        s.wait()
        print(s.poll())

        print("Analyzing")
        return tmpfile

    def realign_subs(self, segments, subs, options):

        updated = kept = 0    
        subs = sorted(subs, key=operator.itemgetter("start"))

        for sub in subs:
            found = False
            # Find a start in the segments that is very close, and if found, re-align
            for segment in segments:
                if abs(segment["start"] - sub["start"]) < options.max_adjust:

                    if not found:
                        # Calculate cps
                        orig = (sub["start"], sub["end"])

                        sub["start"] = segment["start"]

                        found = True
                        updated += 1
                    else:
                        # Found start, find end too
                        if abs(segment["end"] - sub["end"]) < options.max_adjust:
                            sub["end"] = max(sub["start"] + options.min_time, segment["end"])
                            break

                    # print("ADJUST", orig, "->", (sub["start"], sub["end"]), cps, newcps, sub["text"])
                    # break
            if not found:
                # print("Keeping", (sub["start"], sub["end"]), sub["text"])
                kept += 1

            cps = len(sub["text"]) / (sub["end"] - sub["start"])
            if options.max_cps and cps > options.max_cps:
                # print("** Too fast")
                sub["end"] = sub["start"] + len(sub["text"]) / float(options.max_cps)
            if options.min_cps and cps < options.min_cps:
                # print("** Too slow", (sub["start"], sub["end"]), (len(sub["text"]) / float(options.min_cps)))
                sub["end"] = sub["start"] + max(options.min_time,  (len(sub["text"]) / float(options.min_cps)))
            newcps = len(sub["text"]) / (sub["end"] - sub["start"])


        # Do some additional checking - if two subs close very close to each other, bundle them
        threshold = 0.4
        # If some overlap with a tiny bit, shorten down the first
        for idx, sub in enumerate(subs):
            if idx > 0:
                if abs(sub["end"] - subs[idx-1]["end"]) < threshold:
                    # print("Aligning ends", subs[idx-1], sub)
                    subs[idx-1]["end"] = sub["end"]
                if subs[idx-1]["end"] - sub["start"]  < threshold * 2 and subs[idx-1]["end"] - sub["start"] > 0:
                    # print("Overlapping\n", subs[idx-1],"\n", sub)
                    subs[idx-1]["end"] = sub["start"] - 0.001

        # Sanity
        for sub in subs:
            if sub["end"] < sub["start"]:
                raise SystemExit("Super-wrong, end is before start", sub)

        print("Updated", updated, "kept", kept)

        return subs


if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="src", help="Source audio/video file", required=True)
    parser.add_argument("-a", "--aggressive", dest="aggressive", help="How aggressive (0-3, 3 is most aggressive)", default=2)
    parser.add_argument("-s", "--sub", dest="sub", help="Subtitle file (json), if given it will be realigned and written to output file", required=False)
    parser.add_argument("-o", "--output", dest="dst", help="Output file", required=False)
    parser.add_argument("--output_dir", dest="output_dir", help="Output directory for segments (if not given, not saved)", default=None)

    parser.add_argument("--min_cps", dest="min_cps", help="Minimum CPS", default=12)
    parser.add_argument("--max_cps", dest="max_cps", help="Maximum CPS", default=18)
    parser.add_argument("--max_adjust", dest="max_adjust", help="Maximum adjustment (s)", default=0.7)
    parser.add_argument("--min_time", dest="min_time", help="Minimium time for a sub (s)", default=1.2)
    parser.add_argument("--max_pause", dest="max_pause", help="Merge if closer than this (if >0s)", default=0)
    parser.add_argument("--max_segment_length", dest="max_segment_length", help="Max segment length", default=30)

    options = parser.parse_args()

    options.min_cps = float(options.min_cps)
    options.max_cps = float(options.max_cps)
    options.max_adjust = float(options.max_adjust)
    options.min_time = float(options.min_time)


    if 0:

        is_tmp = False
        if not options.src.endswith(".wav"):
            is_tmp = True
            options.src = convert(options.src)

        if options.output_dir and not os.path.exists(options.output_dir):
            os.makedirs(options.output_dir)


        audio, sample_rate = read_wave(options.src)
        vad = webrtcvad.Vad(int(options.aggressive))
        frames = frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(sample_rate, 30, 3, vad, frames, output_dir=options.output_dir, max_pause=float(options.max_pause))

    detector = VoiceDetector(options.src, output_dir=options.output_dir)
    segments = detector.analyze(aggressive=options.aggressive,
                                max_pause=float(options.max_pause),
                                max_segment_length=float(options.max_segment_length))

    if not options.sub:
        if options.dst:
            with open(options.dst, "w") as f:
                json.dump(segments, f, indent=" ")
        else:
            print(json.dumps(segments, indent=" "))

    # Read the sub file and try to align
    if options.sub:
        print("Aligning with subs from", options.sub)
        with open(options.sub, "r") as f:
            subs = json.load(f)

            subs = detector.realign_subs(segments, subs, options)

        if options.dst:
            print("Saving to", options.dst)
            with open(options.dst, "w") as f:
                json.dump(subs, f, indent=" ")
        else:
            print("Not saving updated file")

    if 0:
        if is_tmp:
            os.remove(options.src)