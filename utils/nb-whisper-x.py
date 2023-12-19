import argparse
import json
import whisperx
import gc
import torch

def transcribe_audio(audio_file, language, model_size, output_format):
    device = "cuda"
    compute_type = "float16"  # Change to "int8" if low on GPU mem (may reduce accuracy)
    batch_size = 32  # Reduce if low on GPU mem. Higher means faster

    # Selecting the correct model based on language and model size
    whisper_model = "NbAiLabBeta/nb-whisper-tiny" if model_size == "tiny" else "NbAiLabBeta/nb-whisper-" + model_size
    wav2vec_model = "NbAiLab/nb-wav2vec2-1b-bokmaal" if language == "no" else (
        "NbAiLab/nb-wav2vec2-300m-nynorsk" if language == "nn" else None)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(whisper_model, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(json.dumps(result["segments"], indent=4))  # Before alignment

    # Cleaning up resources
    # gc.collect()
    # torch.cuda.empty_cache()
    # del model

    # 2. Align whisper output
    if wav2vec_model:
        model_a, metadata = whisperx.load_align_model(model_name=wav2vec_model, language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        print(json.dumps(result["segments"], indent=4))  # After alignment

        # Cleaning up resources
        # gc.collect()
        # torch.cuda.empty_cache()
        # del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(json.dumps(result["segments"], indent=4))  # Segments are now assigned speaker IDs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe an audio file using WhisperX.')
    parser.add_argument('--audio_file', type=str, required=True, help='Path to the audio file.')
    parser.add_argument('--language', type=str, default='no', choices=['no', 'nn', 'en'], help='Language of the audio (no, nn, en).')
    parser.add_argument('--model_size', type=str, default='tiny', choices=['tiny', 'base', 'small', 'medium', 'large'], help='Size of the Whisper model.')
    parser.add_argument('--output_format', type=str, default='json', choices=['json'], help='Output format of the transcription.')

    args = parser.parse_args()
    transcribe_audio(args.audio_file, args.language, args.model_size, args.output_format)
