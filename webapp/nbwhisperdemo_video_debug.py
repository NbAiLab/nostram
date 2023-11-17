import logging
import math
import os
import shutil
import tempfile
import time
from multiprocessing import Pool

import gradio as gr
import jax.numpy as jnp
import numpy as np
import yt_dlp as youtube_dl
from jax.experimental.compilation_cache import compilation_cache as cc
from pydub import AudioSegment
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.pipelines.audio_utils import ffmpeg_read
import tempfile
import base64

from whisper_jax import FlaxWhisperPipline

cc.initialize_cache("./jax_cache")

import argparse
import re

# Define valid checkpoints and corresponding batch sizes
valid_checkpoints = {
    "tiny": 128,
    "base": 128,
    "small": 256,
    "medium": 32,
    "large": 8,
}

# Create the parser
parser = argparse.ArgumentParser(description='Run the transcription script with a specific checkpoint.')
parser.add_argument('--checkpoint', type=str, help='The checkpoint to use for the model.', required=True)

# Parse the arguments
args = parser.parse_args()

# Check if the checkpoint is valid
found_batch_size = None
for keyword, batch_size in valid_checkpoints.items():
    if keyword in args.checkpoint.lower():
        found_batch_size = batch_size
        break

if found_batch_size is None:
    print(f"Error: The specified checkpoint is not supported.")
    exit(1)

# If the checkpoint is valid, set it and the corresponding batch size
checkpoint = args.checkpoint
BATCH_SIZE = found_batch_size

# Generate title from the checkpoint name
title_parts = checkpoint.split("/")
title = title_parts[-1]  # Take the part after the slash
title = title.replace("-", " ").lower()  # Replace hyphens with spaces

title = title.title()
title = title.replace("Nb Whisper", "NB-Whisper")
title = title.replace("Beta", "(beta)")
title = title.replace("Rc", "RC")

CHUNK_LENGTH_S = 30
NUM_PROC = 32
FILE_LIMIT_MB = 1000
YT_LENGTH_LIMIT_S = 10800  # limit to 3 hour YouTube files

description = ""

article = f"Submit feedback [here](https://forms.gle/cCQzdox9N2ENDczV7). Backend running JAX on a TPU v3 through support from the " \
          f"[TRC](https://sites.research.google/trc/about/) programme. " \
          f"Whisper JAX [code](https://github.com/sanchit-gandhi/whisper-jax) and Gradio demo by 🤗 Hugging Face."

language_names = sorted(TO_LANGUAGE_CODE.keys())

logger = logging.getLogger("whisper-jax-app")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


def convert_to_proper_time_format(time):
    time_parts = time.split(":")
    if len(time_parts) == 2:
        return f"00:{time}"
    elif len(time_parts) == 3 and len(time_parts[0]) == 1:
        return f"0{time}"
    else:
        return time


def format_to_vtt(text, timestamps, transcription_type="verbatim", style=""):
    if not timestamps:
        return None

    # Define styles for different types of transcriptions
    verbatim_style = "line:10% align:center position:50% size:100%"
    semantic_style = "line:90% align:center position:50% size:100%"

    # Set style based on transcription type
    style = verbatim_style if transcription_type == "verbatim" else semantic_style
    
    vtt_lines = [
        f"WEBVTT",
        "",
        "NOTE",
        f"Denne transkripsjonen er autogenerert av Nasjonalbibliotekets {title} basert på OpenAIs Whisper-modell.",
        f"Se detaljer og last ned modellen her: https://huggingface.co/{checkpoint}.",
        "",
        "0",
        f"00:00:00.000 --> 00:00:06.000 {style}".strip(),
        f"(Automatisk teksting av {title})",
        ""
    ]
    counter = 1
    for chunk in text.split("\n"):
        try:
            start_time, rest = chunk.split(" -> ")
            end_time, subtitle_text = rest.split("] ")
        except ValueError:
            print(f"Skipping malformed chunk: {chunk}")
            continue

        start_time = start_time.replace("[", "").replace(",", ".")
        end_time = end_time.replace(",", ".")

        start_time = convert_to_proper_time_format(start_time)
        end_time = convert_to_proper_time_format(end_time)

        # Don't let the disclaimer overlap with the first subtitle
        if start_time.startswith("00:00:0") and int(start_time[7]) < 6:
            vtt_lines[7] = vtt_lines[7].replace("00:00:06.000", start_time)

        subtitle_text = subtitle_text.strip()
        subtitle_text = subtitle_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        subtitle_text = split_long_lines(subtitle_text)

        vtt_lines.append(str(counter))
        vtt_lines.append(f"{start_time} --> {end_time} {style}".strip())
        vtt_lines.append(subtitle_text)
        vtt_lines.append("")

        counter += 1

    return "\n".join(vtt_lines)


def merge_and_sort_subtitles(vtt_file1, vtt_file2):
    def extract_subtitles(vtt_file):
        with open(vtt_file, 'r') as file:
            lines = file.readlines()

        # Find the index where actual subtitles start
        start_index = 0
        for i, line in enumerate(lines):
            if "-->" in line:
                start_index = i - 1
                break

        # Split lines into groups of 3 (index, timestamp, text)
        return [lines[i:i+3] for i in range(start_index, len(lines), 3)], start_index

    # Extract subtitles from both files
    subtitles1, start_index1 = extract_subtitles(vtt_file1)
    subtitles2, _ = extract_subtitles(vtt_file2)

    def sort_subtitles(subtitles):
        # Sort by the start time in the timestamp
        return sorted(subtitles, key=lambda x: x[1])

    # Merge and sort subtitles
    merged_subtitles = sort_subtitles(subtitles1 + subtitles2)

    # Read header from the first file
    with open(vtt_file1, 'r') as file:
        header = ''.join(file.readlines()[:start_index1])

    # Combine header and sorted subtitle groups
    combined_vtt = header + ''.join([''.join(group) for group in merged_subtitles])

    return combined_vtt


def split_long_lines(subtitle_text):
    lines = 1 + len(subtitle_text) // 60
    if lines > 1:
        original_text = subtitle_text
        # Split into multiple lines
        words = subtitle_text.split(" ")
        total_len = len(subtitle_text)
        target_len = total_len / lines

        word_len = [len(word) + 1 for word in words if word]
        totals = np.cumsum(word_len)

        indices = []
        for l in range(lines - 1):
            idx = np.argmin(np.abs((l + 1) * target_len - totals))
            indices.append(idx + 1)

        subtitle_text = "\n".join(" ".join(words[i:j]) for i, j in zip([None] + indices, indices + [None]))
        if original_text != subtitle_text.replace("\n", " "):
            print("Hei")
    return subtitle_text


def format_to_srt(text, timestamps):
    if not timestamps:
        return None

    srt_lines = []
    counter = 1

    for chunk in text.split("\n"):
        start_time, rest = chunk.split(" -> ")
        end_time, subtitle_text = rest.split("] ")

        start_time = start_time.replace("[", "").replace(".", ",")
        end_time = end_time.replace(".", ",")

        start_time = convert_to_proper_time_format(start_time)
        end_time = convert_to_proper_time_format(end_time)

        srt_lines.append(str(counter))
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(subtitle_text.strip())
        srt_lines.append("")

        counter += 1

    return "\n".join(srt_lines)


def identity(batch):
    return batch


# Copied from https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/utils.py#L50
def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = "."):
    if seconds is not None:
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    else:
        # we have a malformed timestamp so just return it as is
        return seconds


if __name__ == "__main__":
    pipeline = FlaxWhisperPipline(checkpoint, dtype=jnp.bfloat16, batch_size=BATCH_SIZE)
    stride_length_s = CHUNK_LENGTH_S / 6
    chunk_len = round(CHUNK_LENGTH_S * pipeline.feature_extractor.sampling_rate)
    stride_left = stride_right = round(stride_length_s * pipeline.feature_extractor.sampling_rate)
    step = chunk_len - stride_left - stride_right
    pool = Pool(NUM_PROC)

    # do a pre-compile step so that the first user to use the demo isn't hit with a long transcription time
    logger.info("compiling forward call...")
    start = time.time()
    random_inputs = {"input_features": np.ones((BATCH_SIZE, 80, 3000))}
    random_timestamps = pipeline.forward(random_inputs, batch_size=BATCH_SIZE, return_timestamps=True)
    compile_time = time.time() - start
    logger.info(f"compiled in {compile_time}s")


    def tqdm_generate(inputs: dict, language: str, task: str, return_timestamps: bool, progress: gr.Progress):
        inputs_len = inputs["array"].shape[0]
        all_chunk_start_idx = np.arange(0, inputs_len, step)
        num_samples = len(all_chunk_start_idx)
        num_batches = math.ceil(num_samples / BATCH_SIZE)
        dummy_batches = list(
            range(num_batches)
        )  # Gradio progress bar not compatible with generator, see https://github.com/gradio-app/gradio/issues/3841

        dataloader = pipeline.preprocess_batch(inputs, chunk_length_s=CHUNK_LENGTH_S, batch_size=BATCH_SIZE)
        progress(0, desc="Pre-processing audio file...")
        logger.info("pre-processing audio file...")
        dataloader = pool.map(identity, dataloader)
        logger.info("done post-processing")

        if language == "Bokmål":
            language = "no"
        elif language == "Nynorsk":
            language = "nn"
        else:
            language = "en"
            
        if task == "Verbatim":
            task = "transcribe"
            
        if task == "Semantic":
            task = "translate"
        
        start_time = time.time()
        logger.info(f"transcribing... language:{language}, task:{task}")
        model_outputs = []
        # iterate over our chunked audio samples - always predict timestamps to reduce hallucinations
        verbatim_outputs = []
        semantic_outputs = []
        for batch, _ in zip(dataloader, progress.tqdm(dummy_batches, desc="Transcribing...")):
            if task == "Both":
                verbatim_outputs.append(
                    pipeline.forward(batch, batch_size=BATCH_SIZE, task="transcribe", language=language, return_timestamps=True)
                )
                semantic_outputs.append(
                    pipeline.forward(batch, batch_size=BATCH_SIZE, task="translate", language=language, return_timestamps=True)
                )
            else:
                model_outputs.append(
                    pipeline.forward(batch, batch_size=BATCH_SIZE, task=task, language=language, return_timestamps=True)
                )      
            
        runtime = time.time() - start_time
        logger.info("done transcription")
        logger.info("post-processing...")
        
        # Post-process and combine results if 'Both' is selected
        if task == "Both":
            verbatim_post_processed = pipeline.postprocess(verbatim_outputs, return_timestamps=True)
            semantic_post_processed = pipeline.postprocess(semantic_outputs, return_timestamps=True)

            verbatim_text = verbatim_post_processed["text"]
            semantic_text = semantic_post_processed["text"]

            combined_text = f"Verbatim Transcription:\n{verbatim_text}\n\nSemantic Transcription:\n{semantic_text}"
            text = combined_text
        else:
            post_processed = pipeline.postprocess(model_outputs, return_timestamps=True)
            text = post_processed["text"]
            if return_timestamps:
                timestamps = post_processed.get("chunks")
                timestamps = [
                    f"[{format_timestamp(chunk['timestamp'][0])} -> {format_timestamp(chunk['timestamp'][1])}] {chunk['text']}"
                    for chunk in timestamps
                ]
                text = "\n".join(str(feature) for feature in timestamps)
            logger.info(f"done post-processing")
            logger.info(f"transcribed {len(text.split())} words and {len(text)} characters in {runtime:.2f}s")
            
        return text, runtime

    def prepare_audio_for_transcription(file):
        tmpdirname = tempfile.mkdtemp()
        file_path = os.path.join(tmpdirname, file.name)
        shutil.move(file.name, file_path)
        file_size_mb = os.stat(file_path).st_size / (1024 * 1024)
        if file_size_mb > FILE_LIMIT_MB:
            raise Exception(f"File size exceeds limit: {file_size_mb:.2f}MB / {FILE_LIMIT_MB}MB")

        if file_path.endswith(".mp4"):
            video = AudioSegment.from_file(file_path, "mp4")
            audio_path_pydub = file_path.replace(".mp4", ".wav")
            video.export(audio_path_pydub, format="wav")
            with open(audio_path_pydub, "rb") as f:
                file_contents = f.read()
        else:
            with open(file_path, "rb") as f:
                file_contents = f.read()
            video_file_path = re.sub(r"\.[^.]+$", ".mp4", file_path)
            ffmpeg_cmd = f'ffmpeg -y -f lavfi -i color=c=black:s=1280x720 -i "{file_path}" ' \
                        f'-shortest -fflags +shortest -loglevel error "{video_file_path}"'
            os.system(ffmpeg_cmd)
            file_path = video_file_path

        return file_contents, file_path
    
    def create_transcript_file(text, file_path, return_timestamps):
        if return_timestamps:
            transcript_content = format_to_vtt(text, return_timestamps, style="line:50% align:center position:50% size:100%")
            subtitle_display = re.sub(r"\.[^.]+$", "_middle.vtt", file_path)
            with open(subtitle_display, "w") as f:
                f.write(transcript_content)
            transcript_content = format_to_vtt(text, return_timestamps)
            transcript_file_path = re.sub(r"\.[^.]+$", ".vtt", file_path)
        else:
            transcript_content = text
            transcript_file_path = re.sub(r"\.[^.]+$", ".txt", file_path)
            subtitle_display = None

        with open(transcript_file_path, "w") as f:
            f.write(transcript_content)

        return transcript_file_path, subtitle_display
    
    def perform_transcription(file_contents, language, task, return_timestamps, progress):
        inputs = ffmpeg_read(file_contents, pipeline.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": pipeline.feature_extractor.sampling_rate}
        logger.info("done loading")
        
        text, runtime = tqdm_generate(inputs, language=language, task=task, return_timestamps=return_timestamps, progress=progress)
        return text, runtime


    def transcribe_chunked_audio(file, language, task, return_timestamps, progress=gr.Progress()):
        if file is None:
            logger.warning("No audio file provided")
            raise gr.Error("No audio file submitted! Please upload an audio file before submitting your request.")
        
        file_contents, file_path = prepare_audio_for_transcription(file)
        
        if task == "Both":
            # Transcribe for Verbatim
            verbatim_text, _ = perform_transcription(file_contents, language, "Verbatim", return_timestamps, progress)
            verbatim_vtt_path, verbatim_subtitle_display = create_transcript_file(verbatim_text, file_path, return_timestamps)

            # Transcribe for Semantic
            semantic_text, runtime = perform_transcription(file_contents, language, "Semantic", return_timestamps, progress)
            semantic_vtt_path, semantic_subtitle_display = create_transcript_file(semantic_text, file_path, return_timestamps)

            # Merge and sort subtitles
            merged_subtitles = merge_and_sort_subtitles(verbatim_vtt_path, semantic_vtt_path)

            # Combine the texts for display in UI
            text = "Verbatim translation:\n" + verbatim_text + "\n\n" + "Semantic translation:\n" + semantic_text

            # Use the merged subtitles for display and download options
            subtitle_display = merged_subtitles  # This needs to be formatted for display in UI
            transcript_file_path = None  # Since individual files are available for download, not the merged one
        else:
            # Handle as before for Verbatim or Semantic only
            text, runtime = perform_transcription(file_contents, language, task, return_timestamps, progress)
            transcript_file_path, subtitle_display = create_transcript_file(text, file_path, return_timestamps)

        
        text, runtime = perform_transcription(file_contents, language, task, return_timestamps, progress)
        transcript_file_path, subtitle_display = create_transcript_file(text, file_path, return_timestamps)

        if file_path.endswith(".mp4"):
            value = [file_path, subtitle_display] if subtitle_display is not None else file_path
            o0 = youtube.output_components[0].update(visible=True, value=value)
            o1 = youtube.output_components[1].update(visible=False)
        else:
            o0 = youtube.output_components[1].update(visible=False)
            o1 = youtube.output_components[1].update(visible=True, value=file_path)

        return o0, o1, text, runtime, transcript_file_path


    def _return_yt_html_embed(yt_url):
        video_id = yt_url.split("?v=")[-1]
        HTML_str = (
            f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
            " </center>"
        )
        return HTML_str


    def download_yt_audio(yt_url, folder, video=False):
        info_loader = youtube_dl.YoutubeDL()
        try:
            info = info_loader.extract_info(yt_url, download=False)
        except youtube_dl.utils.DownloadError as err:
            raise gr.Error(str(err))

        file_length = info["duration_string"]
        file_h_m_s = file_length.split(":")
        file_h_m_s = [int(sub_length) for sub_length in file_h_m_s]
        if len(file_h_m_s) == 1:
            file_h_m_s.insert(0, 0)
        if len(file_h_m_s) == 2:
            file_h_m_s.insert(0, 0)

        file_length_s = file_h_m_s[0] * 3600 + file_h_m_s[1] * 60 + file_h_m_s[2]
        if file_length_s > YT_LENGTH_LIMIT_S:
            yt_length_limit_hms = time.strftime("%HH:%MM:%SS", time.gmtime(YT_LENGTH_LIMIT_S))
            file_length_hms = time.strftime("%HH:%MM:%SS", time.gmtime(file_length_s))
            raise gr.Error(f"Maximum YouTube length is {yt_length_limit_hms}, got {file_length_hms} YouTube video.")

        fpath = os.path.join(folder, f"{info['id'].replace('.', '_')}.mp4")

        video = "bestvideo[height <=? 720]" if video else "worstvideo"
        ydl_opts = {"outtmpl": fpath, "format": f"{video}[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"}
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([yt_url])
            except youtube_dl.utils.ExtractorError as err:
                raise gr.Error(str(err))

        # Return ID
        return fpath


    def transcribe_youtube(yt_url, language, task, return_timestamps, progress=gr.Progress()):
        use_youtube_player = False
        progress(0, desc="Loading audio file...")
        logger.info("loading youtube file...")
        html_embed_str = _return_yt_html_embed(yt_url)

        tmpdirname = tempfile.mkdtemp()
        video_filepath = download_yt_audio(yt_url, tmpdirname, video=return_timestamps)

        with open(video_filepath, "rb") as f:
            inputs = f.read()

        inputs = ffmpeg_read(inputs, pipeline.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": pipeline.feature_extractor.sampling_rate}
        logger.info("done loading...")
        text, runtime = tqdm_generate(inputs, language=language, task=task,
                                      return_timestamps=return_timestamps, progress=progress)

        if return_timestamps:
            transcript_content = format_to_vtt(text, return_timestamps)
            transcript_file_path = re.sub(r"\.[^.]+$", ".vtt", video_filepath)
        else:
            transcript_content = text
            transcript_file_path = re.sub(r"\.[^.]+$", ".txt", video_filepath)

        with open(transcript_file_path, "w") as f:
            f.write(transcript_content)

        if use_youtube_player:
            o0 = youtube.output_components[0].update(visible=True, value=html_embed_str)
            o1 = youtube.output_components[1].update(visible=False)
        else:
            o0 = youtube.output_components[0].update(visible=False)
            value = [video_filepath, transcript_file_path] if return_timestamps else video_filepath
            o1 = youtube.output_components[1].update(visible=True, value=value)

        return o0, o1, text, runtime, transcript_file_path


    # microphone_chunked = gr.Interface(
    #     fn=transcribe_chunked_audio,
    #     inputs=[
    #         gr.inputs.Audio(source="microphone", optional=True, type="filepath"),
    #         gr.inputs.Radio(["Bokmål", "Nynorsk", "English"], label="Output language", default="Bokmål"),
    #         gr.inputs.Radio(["Verbatim", "Semantic"], label="Transcription style", default="Verbatim"),
    #         gr.inputs.Checkbox(default=True, label="Return timestamps"),
    #     ],
    #     outputs=[
    #         gr.outputs.Textbox(label="Transcription").style(show_copy_button=True),
    #         gr.outputs.Textbox(label="Transcription Time (s)"),
    #         gr.outputs.File(label="Download")
    #     ],
    #     allow_flagging="never",
    #     title=title,
    #     description=description,
    #     article=article,
    # )

    audio_chunked = gr.Interface(
        fn=transcribe_chunked_audio,
        inputs=[
            gr.inputs.File(optional=True, label="File (audio/video)", type="file"),
            gr.inputs.Radio(["Bokmål", "Nynorsk", "English"], label="Output language", default="Bokmål"),
            gr.inputs.Radio(["Verbatim", "Semantic", "Both"], label="Transcription style", default="Verbatim"),
            gr.inputs.Checkbox(default=True, label="Return timestamps"),
        ],
        outputs=[
            gr.Video(label="Video", visible=False),
            gr.Audio(label="Audio", visible=False),
            gr.outputs.Textbox(label="Transcription").style(show_copy_button=True),
            gr.outputs.Textbox(label="Transcription Time (s)"),
            gr.outputs.File(label="Download"),
        ],
        allow_flagging="never",
        title=title,
        description=description,
        article=article,
    )

    youtube = gr.Interface(
        fn=transcribe_youtube,
        inputs=[
            gr.inputs.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL"),
            gr.inputs.Radio(["Bokmål", "Nynorsk", "English"], label="Output language", default="Bokmål"),
            gr.inputs.Radio(["Verbatim", "Semantic", "Both"], label="Transcription style", default="Semantic"),
            gr.inputs.Checkbox(default=True, label="Return timestamps"),
            # gr.inputs.Checkbox(default=False, label="Use YouTube player"),
        ],
        outputs=[
            gr.outputs.HTML(label="Video"),
            gr.outputs.Video(label="Video"),
            gr.outputs.Textbox(label="Transcription").style(show_copy_button=True),
            gr.outputs.Textbox(label="Transcription Time (s)"),
            gr.outputs.File(label="Download"),
        ],
        allow_flagging="never",
        title=title,
        examples=[
            ["https://www.youtube.com/watch?v=_uv74o8hG30", "Bokmål", "Verbatim",True, False],
            ["https://www.youtube.com/watch?v=JtbZWIcj0kbk", "Bokmål", "Verbatim",True, False],
            ["https://www.youtube.com/watch?v=vauTloX4HkU", "Bokmål", "Semantic",True, False]
        ],
        cache_examples=False,
        description=description,
        article=article,
    )

    demo = gr.Blocks()

    with demo:
        gr.Image("nb-logo-full-cropped.png", show_label=False, interactive=False, height=100, container=False)
        gr.TabbedInterface([audio_chunked, youtube], ["File", "YouTube"])

    demo.queue(concurrency_count=1, max_size=5)
    demo.launch(server_name="0.0.0.0", share=True, show_api=True)
