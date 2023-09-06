import logging
import math
import os
import tempfile
import time
from multiprocessing import Pool

import gradio as gr
import jax.numpy as jnp
import numpy as np
import yt_dlp as youtube_dl
from jax.experimental.compilation_cache import compilation_cache as cc
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.pipelines.audio_utils import ffmpeg_read
import tempfile

from whisper_jax import FlaxWhisperPipline

cc.initialize_cache("./jax_cache")

import argparse

# Define valid checkpoints, corresponding batch sizes, and titles
valid_checkpoints = {
    "NbAiLab/nb-whisper-tiny-beta": 32,
    "NbAiLab/nb-whisper-base-beta": 32,
    "NbAiLab/nb-whisper-small-beta": 32,
    "NbAiLab/nb-whisper-medium-beta": 32,
    "NbAiLab/nb-whisper-large-beta": 32,
}

titles = {
    "NbAiLab/nb-whisper-tiny-beta": "NB Whisper Tiny BETA ⚡️",
    "NbAiLab/nb-whisper-base-beta": "NB Whisper Base BETA ⚡️",
    "NbAiLab/nb-whisper-small-beta": "NB Whisper Small BETA ⚡️",
    "NbAiLab/nb-whisper-medium-beta": "NB Whisper Medium BETA ⚡️",
    "NbAiLab/nb-whisper-large-beta": "NB Whisper Large BETA ⚡️",
}

# Create the parser
parser = argparse.ArgumentParser(description='Run the transcription script with a specific checkpoint.')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='The checkpoint to use for the model. Must be one of: ' + ', '.join(valid_checkpoints.keys()))

# Parse the arguments
args = parser.parse_args()

# Check if the checkpoint is valid
if args.checkpoint not in valid_checkpoints:
    print(
        f"Error: The specified checkpoint is not supported. Please choose from: {', '.join(valid_checkpoints.keys())}")
    exit(1)

# If the checkpoint is valid, set it, the corresponding batch size, and title
checkpoint = args.checkpoint
BATCH_SIZE = valid_checkpoints[checkpoint]
title = titles[checkpoint]

CHUNK_LENGTH_S = 30
NUM_PROC = 32
FILE_LIMIT_MB = 1000
YT_LENGTH_LIMIT_S = 10800  # limit to 3 hour YouTube files

description = f"This is a demo of the {title}. " \
              "The model is trained by the [AI-Lab at the National Library of Norway](https://ai.nb.no/). "

article = f"Backend running JAX on a TPU v4-8 through the generous support of the " \
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

        srt_lines.append(str(counter))
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(subtitle_text.strip())
        srt_lines.append("")

        counter += 1

    return "\n".join(srt_lines)


def save_to_temp_file(srt_content, suffix):
    """
    Saves the SRT content to a temporary file and returns the path to that file.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w") as f:
        f.write(srt_content)
    return f.name


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


    def tqdm_generate(inputs: dict, language: str, return_timestamps: bool, progress: gr.Progress):
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
            task = "transcribe"
        elif language == "Nynorsk":
            language = "nn"
            task = "transcribe"
        else:
            language = "no"
            task = "translate"

        model_outputs = []
        start_time = time.time()
        logger.info("transcribing...")
        # iterate over our chunked audio samples - always predict timestamps to reduce hallucinations
        for batch, _ in zip(dataloader, progress.tqdm(dummy_batches, desc="Transcribing...")):
            model_outputs.append(
                pipeline.forward(batch, batch_size=BATCH_SIZE, task=task, language=language, return_timestamps=True))
        runtime = time.time() - start_time
        logger.info("done transcription")
        logger.info("post-processing...")
        post_processed = pipeline.postprocess(model_outputs, return_timestamps=True)
        text = post_processed["text"]
        if return_timestamps:
            timestamps = post_processed.get("chunks")
            timestamps = [
                f"[{format_timestamp(chunk['timestamp'][0])} -> {format_timestamp(chunk['timestamp'][1])}] {chunk['text']}"
                for chunk in timestamps
            ]
            text = "\n".join(str(feature) for feature in timestamps)
        logger.info("done post-processing")
        return text, runtime


    def transcribe_chunked_audio(inputs, language, return_timestamps, progress=gr.Progress()):
        progress(0, desc="Loading audio file...")
        logger.info("loading audio file...")
        if inputs is None:
            logger.warning("No audio file")
            raise gr.Error("No audio file submitted! Please upload an audio file before submitting your request.")
        file_size_mb = os.stat(inputs).st_size / (1024 * 1024)
        if file_size_mb > FILE_LIMIT_MB:
            logger.warning("Max file size exceeded")
            raise gr.Error(
                f"File size exceeds file size limit. Got file of size {file_size_mb:.2f}MB for a limit of {FILE_LIMIT_MB}MB."
            )

        with open(inputs, "rb") as f:
            inputs = f.read()

        inputs = ffmpeg_read(inputs, pipeline.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": pipeline.feature_extractor.sampling_rate}
        logger.info("done loading")
        text, runtime = tqdm_generate(inputs, language=language,
                                      return_timestamps=return_timestamps, progress=progress)

        if return_timestamps:
            srt_content = format_to_srt(text, return_timestamps)
            file_path = save_to_temp_file(srt_content, ".srt")
        else:
            file_path = save_to_temp_file(text, ".txt")

        return text, runtime, file_path


    def _return_yt_html_embed(yt_url):
        video_id = yt_url.split("?v=")[-1]
        HTML_str = (
            f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
            " </center>"
        )
        return HTML_str


    def download_yt_audio(yt_url, filename):
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

        ydl_opts = {"outtmpl": filename, "format": "worstvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"}
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([yt_url])
            except youtube_dl.utils.ExtractorError as err:
                raise gr.Error(str(err))


    def transcribe_youtube(yt_url, language, return_timestamps, progress=gr.Progress()):
        progress(0, desc="Loading audio file...")
        logger.info("loading youtube file...")
        html_embed_str = _return_yt_html_embed(yt_url)
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "video.mp4")
            download_yt_audio(yt_url, filepath)

            with open(filepath, "rb") as f:
                inputs = f.read()

        inputs = ffmpeg_read(inputs, pipeline.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": pipeline.feature_extractor.sampling_rate}
        logger.info("done loading...")
        text, runtime = tqdm_generate(inputs, language=language,
                                      return_timestamps=return_timestamps, progress=progress)

        if return_timestamps:
            srt_content = format_to_srt(text, return_timestamps)
            file_path = save_to_temp_file(srt_content, ".srt")
        else:
            file_path = save_to_temp_file(text, ".txt")

        return html_embed_str, text, runtime, file_path


    microphone_chunked = gr.Interface(
        fn=transcribe_chunked_audio,
        inputs=[
            gr.inputs.Audio(source="microphone", optional=True, type="filepath"),
            gr.inputs.Radio(["Bokmål", "Nynorsk", "English"], label="Output language", default="Bokmål"),
            gr.inputs.Checkbox(default=False, label="Return timestamps"),
        ],
        outputs=[
            gr.outputs.Textbox(label="Transcription").style(show_copy_button=True),
            gr.outputs.Textbox(label="Transcription Time (s)"),
            gr.outputs.File(label="Download")
        ],
        allow_flagging="never",
        title=title,
        description=description,
        article=article,
    )

    audio_chunked = gr.Interface(
        fn=transcribe_chunked_audio,
        inputs=[
            gr.inputs.Audio(source="upload", optional=True, label="Audio file", type="filepath"),
            gr.inputs.Radio(["Bokmål", "Nynorsk", "English"], label="Output language", default="Bokmål"),
            gr.inputs.Checkbox(default=False, label="Return timestamps"),
        ],
        outputs=[
            gr.outputs.Textbox(label="Transcription").style(show_copy_button=True),
            gr.outputs.Textbox(label="Transcription Time (s)"),
            gr.outputs.File(label="Download")
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
            gr.inputs.Checkbox(default=False, label="Return timestamps"),
        ],
        outputs=[
            gr.outputs.HTML(label="Video"),
            gr.outputs.Textbox(label="Transcription").style(show_copy_button=True),
            gr.outputs.Textbox(label="Transcription Time (s)"),
            gr.outputs.File(label="Download .srt")
        ],
        allow_flagging="never",
        title=title,
        examples=[
            ["https://www.youtube.com/watch?v=_uv74o8hG30", "Bokmål", False],
            ["https://www.youtube.com/watch?v=JtbZWIcj0bk", "Bokmål", False],
            ["https://www.youtube.com/watch?v=vauTloX4HkU", "Bokmål", False]
        ],
        cache_examples=False,
        description=description,
        article=article,
    )

    demo = gr.Blocks()

    with demo:
        gr.Image("nb-logo-full-cropped.png", show_label=False, interactive=False, height=100, container=False)
        gr.TabbedInterface([microphone_chunked, audio_chunked, youtube], ["Microphone", "Audio File", "YouTube"])

    demo.queue(concurrency_count=1, max_size=5)
    demo.launch(server_name="0.0.0.0", port="80", show_api=False)
