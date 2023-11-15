import argparse
import tempfile
from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig

def main(model_path, audio_path, commit_hash=None,task="transcribe",language="no",num_beams=1,chunk_length=30):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Load the model and processor
        if commit_hash:
            config = WhisperConfig.from_pretrained(model_path, revision=commit_hash, from_flax=True)
            model = WhisperForConditionalGeneration.from_pretrained(model_path, revision=commit_hash, from_flax=True)
            processor = WhisperProcessor.from_pretrained(model_path, revision=commit_hash, from_flax=True)
        else:
            config = WhisperConfig.from_pretrained(model_path, from_flax=True)
            model = WhisperForConditionalGeneration.from_pretrained(model_path, from_flax=True)
            processor = WhisperProcessor.from_pretrained(model_path, from_flax=True)
            commit_hash="None"

        # Save the model, processor, and config in the tmp directory
        config.save_pretrained(tmp_dir)
        model.save_pretrained(tmp_dir)
        processor.save_pretrained(tmp_dir)

        # Load the model and processor from the tmp directory for the pipeline
        asr = pipeline("automatic-speech-recognition", model=tmp_dir, tokenizer=tmp_dir)

        # Process the audio and print results
        result = asr(audio_path, return_timestamps=True, chunk_length_s=chunk_length, generate_kwargs={'task': task, 'language': language, 'num_beams': num_beams})
        
        breakpoint()
        text = result['text']
        word_count = len(text.split())

        print("Transcribed Text:\n", text)
        print(f"\nWord Count: {word_count}. Commit hash: {commit_hash}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech to Text Transcription")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Whisper model")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams (default: 1)")
    parser.add_argument("--chunk_length", type=int, default=30, help="Chunk length (default: 30)")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="Task to perform: 'transcribe' or 'translate' (default: transcribe)")
    parser.add_argument("--language", type=str, default="no", choices=["no", "nn", "en"], help="Target language: 'no', 'nn' or 'translate' (default: no)")
    parser.add_argument("--commit_hash", type=str, default=None, help="Specific commit hash for the model (optional)")
    args = parser.parse_args()

    main(args.model_path, args.audio_path, args.commit_hash,args.task,args.language,args.num_beams,args.chunk_length)
