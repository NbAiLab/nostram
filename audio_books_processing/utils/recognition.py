from pathlib import Path
from typing import List

import torch
from huggingsound import SpeechRecognitionModel, KenshoLMDecoder

from utils import vad


def process_files(files: List[Path], cache_dir: Path = Path('cache'),
                  asr_model: str = 'classla/wav2vec2-xls-r-parlaspeech-hr', lm_path: Path = Path('lm.binary')):
    vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = vad_utils

    model = SpeechRecognitionModel(asr_model)
    model.token_set.tokens.pop(48)
    decoder = KenshoLMDecoder(model.token_set, lm_path=str(lm_path), alpha=0.5, beta=1.5)

    cache_dir.mkdir(exist_ok=True)

    results = []
    for file in files:
        file = Path(file)
        wav = read_audio(file, sampling_rate=16000)
        vad_ts = get_speech_timestamps(wav, vad_model, sampling_rate=16000, speech_pad_ms=1000, return_seconds=True)
        vad_ts = vad.resample(vad_ts)
        filename = file.stem
        segments = []
        for segnum, seg in enumerate(vad_ts):
            nb = int(seg['start'] * 16000)
            ne = int(seg['end'] * 16000)
            seg = (wav[nb:ne] * torch.iinfo(torch.int16).max).to(torch.int16)
            seg_path = cache_dir / f'{filename}_{segnum:03d}.wav'
            save_audio(seg_path, seg, sampling_rate=16000)
            segments.append(str(seg_path))

        transcriptions = model.transcribe(segments, decoder=decoder, batch_size=4)

        for trans, seg in zip(transcriptions, vad_ts):
            results.append({'file': file.name,
                            'start': seg['start'],
                            'end': seg['end'],
                            'text': trans['transcription']})

    return results
