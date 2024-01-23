import webrtcvad
import collections
import os
import math
import subprocess
import tempfile
import argparse
import wave
import json


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class VoiceDetector:
    def __init__(self, aggressive=2):
        self.vad = webrtcvad.Vad(aggressive)
        self._continued = None

    def frame_generator(self, frame_duration_ms, audio, sample_rate):
        """Generates audio frames from PCM audio data.

        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.

        Yields Frames of the requested duration.
        """
        # print(f"GEN, frame_duration_ms: {frame_duration_ms}, sample_rate: {sample_rate}, audio: {len(audio)})")
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            if not os.path.exists("/tmp/firstframe.wav"):
                print(f'offset: {offset}, n: {n}')
                import wave
                f = wave.open("/tmp/firstframe.wav", "wb")
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(sample_rate)
                f.writeframes(audio[offset:offset + (n * 33)])
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(self, sample_rate, frame_duration_ms,
                      padding_frames, vad, frames, min_pause=0, time_offset=0):
        """
        padding means that we will ignore that many frames at the start of a
        voice segment
        """
        triggered = False
        segments = []
        frames_speech = 0
        frames_audio = 0
        padding = start = end = 0
        for idx, frame in enumerate(frames):
            is_speech = vad.is_speech(frame.bytes, sample_rate)
            # is_speech = vad.is_speech(frame, sample_rate)
            if is_speech:
                frames_speech += 1
            else:
                frames_audio += 1

            if not triggered and is_speech:
                triggered = True
                start = idx * frame_duration_ms

            elif triggered and not is_speech:
                if padding < padding_frames:
                    padding += 1
                    continue
                triggered = False
                end = idx * frame_duration_ms

                s = {"type": "voice",
                     "start": time_offset + (start / 1000.),
                     "end": time_offset + (end / 1000.)}

                merged = False
                if min_pause and len(segments) > 0:
                    if s["start"] - segments[-1]["end"] < min_pause:

                        # Only merge if the last segment is too short
                        if segments[-1]["end"] - segments[-1]["start"] < 4.0:
                            merged = True
                            # MERGE
                            segments[-1]["end"] = s["end"]
                            # We should overwrite the last file if output_dir is given!
                            s = segments[-1]

                if not merged:
                    segments.append(s)
                start = end = padding = 0
            elif triggered and is_speech:
                padding = 0
        return segments

    def get_continous_speech(self, audio, min_pause=0.5,
                             frame_len_ms=30, sample_rate=16000,
                             min_length=2, time_offset=0):
        """
        none_if_at_end returns None if there is no break that is long enough.
        min_length means that no VAD will be reported for anything shorter
        """

        frames = self.frame_generator(frame_len_ms, audio, sample_rate)
        frames = list(frames)
        segments = list(self.vad_collector(sample_rate, frame_len_ms, 3, self.vad, frames,
                                           time_offset=time_offset))
        # print("Segments", segments)
        # We now have segments with start and end, find the first continous
        # block with no more than min_pause between the items
        blocks = []
        start_ts = None
        for i, segment in enumerate(segments):
            if start_ts is None:
                start_ts = segment["start"]
                continue  # First segment

            # If the total length is too short, continue
            if segments[i-1]["end"] - start_ts < min_length:
                continue

            if segment["start"] - segments[i-1]["end"] > min_pause:
                blocks.append({"start": start_ts, "end": segments[i-1]["end"]})
                start_ts = segment["start"]

        # If there we didn't use the final one, add it
        if len(segments) > 0 and segments[-1]["end"] - start_ts > min_length and \
              (len(blocks) == 0 or segments[-1]["end"] - start_ts > min_pause):
            blocks.append({"start": start_ts, "end": segments[-1]["end"]})
        return blocks

    def get_regions(self, audio_data, start_s, current_s,
                    min_pause=1.0, min_length=5,
                    final=False, prefix_s=0):
        """
        If the data is finalized, set final=True
        Returns updated start_s, current_s, and list of VAD regions
        """
        vads = self.get_continous_speech(audio_data, min_pause=min_pause,
                                         min_length=min_length,
                                         time_offset=start_s)
        to_process = []
        for vad in vads:
            if vad == vads[-1] and not final:
                # Last one
                if self._continued:
                    # Is this a continuation?
                    if vad["start"] - self._continued["end"] < min_pause:
                        self._continued = {"start": self._continued["start"],
                                           "end": vad["end"]}
                    return vad["end"], vad["end"] + 30, to_process
                self._continued = vad
                return vad["end"], vad["end"] + 30, to_process

            if self._continued:
                if vad["start"] - self._continued["end"] < min_pause:
                    # A continuation
                    to_process.append({"start": self._continued["start"] - prefix_s,
                                       "end": vad["end"]})
                    self._continued = None
                    continue
                else:
                    # This was not a continuation, just add the possible
                    # continuation candidate
                    to_process.append({"start": self._continued["start"],
                                       "end": self._continued["end"]})
                    self._continued = None

            if current_s - vad["end"] < min_pause and not final:
                self._continued = vad
                continue

            to_process.append({"start": vad["start"] - prefix_s, "end": vad["end"]})

            # We add a bit to avoid detecting the last bit of detected speech
            start_s = vad["end"] + 0.1

        current_s += 30
        if self._continued and final:
            to_process.append({"start": self._continued["start"] - prefix_s,
                               "end": self._continued["end"]})

        return start_s, current_s, to_process

    def convert_file(self, source, tmpdir=None, duration=None):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tmpdir) as f:
            output_file = f.name
        cmd = ["ffmpeg", "-i", source, "-loglevel", "error", "-y", "-ac", "1", "-ar", "16000", output_file]
        if duration:
            cmd.extend(["-t", duration])
        
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"Error processing file {source}: {e.output.decode()}")
            return None  # Return None or appropriate value to indicate error

        return output_file
    
    def process_file(self, src, dst=None, aggressive=2, min_pause=0.5, chunked=False):
        converted = False
        if not os.path.splitext(src)[1] == ".wav":
            converted_src = self.convert_file(src)
            if converted_src is None:  # Check if conversion failed
                return False  # Return False as the file could not be processed
            src = converted_src
            converted = True

        try:
            with open(src, "rb") as f:
                f.read(44)  # skip header
                audio_data = f.read()
                sample_rate = 16000
        except IOError as e:
            print(f"Error opening file {src}: {e}")
            return False  # Return False as there was an error in reading the file

        if converted:
            os.remove(src)

        content_len = len(audio_data) / (sample_rate * 2)
        start_ts = 0
        current_ts = 30
        if not chunked:
            current_ts = content_len
        vd = VoiceDetector(aggressive=aggressive)

        all_segments = []
        while True:
            final = current_ts == content_len
            chunk = audio_data[math.floor(start_ts * sample_rate * 2):
                            math.floor(current_ts * sample_rate * 2)]
            start_ts, current_ts, segments = vd.get_regions(chunk, start_ts, current_ts,
                                                            min_pause=min_pause,
                                                            final=final)
            current_ts = min(content_len, current_ts)
            if len(all_segments) > 0:
                if segments and all_segments[-1]["start"] == segments[0]["start"]:
                    all_segments.pop()
            all_segments.extend(segments)
            if final:
                break
            continue

        if dst:
            with open(dst, "w") as f:
                import json
                json.dump(all_segments, f, indent=2)

        # Check if any speech segment is detected
        has_voice = len(all_segments) > 0
        return has_voice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Voice Activity Detection')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to the audio file, directory, or JSON Lines file')
    parser.add_argument('--aggressive', type=int, default=3,
                        help='Aggressiveness of VAD (0-3), lower is more speech')
    parser.add_argument('--pause', type=float, default=0.5,
                        help='Minimum pause length in seconds')
    parser.add_argument("--chunked", action="store_true", default=False, help="Chunked mode")
    args = parser.parse_args()

    vd = VoiceDetector(aggressive=args.aggressive)

    if args.path.endswith('.jsonl'):
        # Process each line in the JSON Lines file
        with open(args.path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if data.get('source') == 'nrk_tv_silence':
                    id = data.get('id', '')
                    # Assuming the id field is used to construct the audio filename
                    # Extracting the subfolder from the id
                    mainfolder = "/mnt/lv_ai_1_ficino/ml/ncc_speech_v5/inference_4/mp3/nrk_tv/"
                    subfolder = id.split('_')[0][-2:] + "/"
                    mp3_file_path = mainfolder+subfolder + id + '.mp3'

                    if os.path.isfile(mp3_file_path):
                        has_voice = vd.process_file(mp3_file_path, None, args.aggressive, args.pause, args.chunked)
                        if has_voice:
                            # print(f'{mp3_file_path}')
                            print(f'{id}')
                        else:
                            ...
                            #print(f'No voice detected.')
                    else:
                        print(f"Error: File not found - {mp3_file_path}")
                        
    # Check if the path is a directory or a file
    if os.path.isdir(args.path):
        # Process each MP3 file in the directory
        for filename in os.listdir(args.path):
            if filename.endswith('.mp3'):
                file_path = os.path.join(args.path, filename)
                has_voice = vd.process_file(file_path, None, args.aggressive, args.pause, args.chunked)
                print(f'{filename}: {has_voice}')
    elif os.path.isfile(args.path) and args.path.endswith('.mp3'):
        # Process a single MP3 file
        has_voice = vd.process_file(args.path, None, args.aggressive, args.pause, args.chunked)
        print(f'{os.path.basename(args.path)}: {has_voice}')
    else:
        print("Invalid path or unsupported file format. Please provide a valid MP3 file or directory.")
