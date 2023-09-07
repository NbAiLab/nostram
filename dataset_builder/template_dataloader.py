# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""NCC-S Whisper: Wrapper around NCC-S for Whisper training"""

import io
import json
import tarfile
import datasets
#from datasets.tasks import AutomaticSpeechRecognition


_CITATION = """\
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={2022},
  url={https://arxiv.org/abs/}
}
"""

_DESCRIPTION = """\
This database was created from NB deposit recordings
"""

_HOMEPAGE = "https://ai.nb.no"

# Example: https://huggingface.co/datasets/NbAiLab/<NCC_S>/resolve/main/data/train/<NCC_S>-no-0001-of-0001.tar.gz
_DATA_URL = "https://huggingface.co/datasets/NbAiLab/<NCC_S>/resolve/main/data/{split}/<NCC_S>-{lang_code}-{shard_idx:04d}-{shard_total:04d}.tar.gz"
# Example: https://huggingface.co/datasets/NbAiLab/<NCC_S>/resolve/main/data/test/<NCC_S>-no-0001-of-0001.json
_METADATA_URL = "https://huggingface.co/datasets/NbAiLab/<NCC_S>/resolve/main/data/{split}/<NCC_S>-{lang_code}-{shard_idx:04d}-{shard_total:04d}.json"

_SHARDS = {
    "no": {
        datasets.Split.TRAIN: 256,
        datasets.Split.VALIDATION: 1,
        datasets.Split.TEST: 1,
    },
}

_SOURCES = ["audio_books", "fleurs", "nrk", "nst", "stortinget"]
_SHARDS["no"].update({f"validation_{source}": 1 for source in _SOURCES if source != "nst"})
_SHARDS["no"].update({f"test_{source}": 1 for source in _SOURCES})


class <NCC_S>Config(datasets.BuilderConfig):
    """BuilderConfig for NCC_S."""

    def __init__(self, *args, **kwargs):
        """BuilderConfig for NCC_S.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(<NCC_S>Config, self).__init__(*args, **kwargs)


class <NCC_S>(datasets.GeneratorBasedBuilder):
    """NCC_S dataset."""

    DEFAULT_WRITER_BATCH_SIZE = 1000
    BUILDER_CONFIGS = [
        <NCC_S>Config(
            name="no",
            version=datasets.Version("1.0.1"),
            description="<NCC_S> Norwegian",
        ),
    ]

    def __init__(self, *args, post_processors=None, **kwargs):
        if not isinstance(post_processors, (tuple, list)):
            post_processors = [post_processors]
        self.post_processors = post_processors
        super().__init__(*args, **kwargs)

    def _info(self):
        sampling_rate = 16000
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "id": datasets.Value("string"),
                "group_id": datasets.Value("string"),
                "source": datasets.Value("string"),
                "audio_language": datasets.Value("string"),
                "audio": datasets.features.Audio(sampling_rate=sampling_rate),
                "audio_duration": datasets.Value("int32"),
                "previous_text": datasets.Value("string"),
                "text_language": datasets.Value("string"),
                "text": datasets.Value("string"),
                "translated_text_no": datasets.Value("string"),
                "translated_text_nn": datasets.Value("string"),
                "translated_text_en": datasets.Value("string"),
                "translated_text_es": datasets.Value("string"),
                "timestamped_text": datasets.Value("string"),
                "wav2vec_wer": datasets.Value("float32"),
                "whisper_wer": datasets.Value("float32"),
                "verbosity_level": datasets.Value("int32"),
                "file": datasets.Value("string"),
                "channels": datasets.Value("int32"),
                "frequency": datasets.Value("int32"),
                "language": datasets.Value("string"),
                "task": datasets.Value("string"),
                "_post_processor": datasets.Value("string"),
            }),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            # task_templates=[
            #     AutomaticSpeechRecognition(
            #         audio_column="audio",
            #         transcription_column="text"
            #     )
            # ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_urls = {}
        splits = _SHARDS[self.config.name].keys()
        for split in splits:
            data_urls[split] = []
            shard_total = _SHARDS["no"][split]
            for shard_idx in range(1, shard_total + 1):
                # .../data/{split}/NCC_S_{lang_code}_{split}_{mic}-{shard_idx}-of-{shard_total}
                string_formatting = dict(
                    split=split,
                    lang_code="no",
                    shard_idx=shard_idx,
                    shard_total=shard_total
                )
                data_urls[split] += [(
                    _METADATA_URL.format(**string_formatting),
                    _DATA_URL.format(**string_formatting)
                )]
        return [
            datasets.SplitGenerator(
                name=split, gen_kwargs={
                    "filepaths": dl_manager.download(data_urls[split]),
                }
            ) for split in splits
        ]

    def _generate_examples(self, filepaths):
        """Yields examples."""
        data_fields = list(self._info().features.keys())
        # Remove so they don't get checked from the metadata mapping
        data_fields.remove("audio")
        data_fields.remove("language")
        data_fields.remove("task")
        data_fields.remove("_post_processor")
        for metadata_path, archive_path in filepaths:
            metadata = {}
            with open(metadata_path) as metadata_file:
                for line in metadata_file.read().split("\n"):
                    if line:
                        metadata_object = json.loads(line)
                        metadata_key = metadata_object["id"]
                        if metadata_object["source"] == "NST":
                            metadata_key = metadata_key[4:]
                        metadata[metadata_key] = metadata_object
            with open(archive_path, "rb") as archive_fs:
                archive_bytes = io.BytesIO(archive_fs.read())
                with tarfile.open(fileobj=archive_bytes, mode="r") as tar:
                    for audio_file in tar.getmembers():
                        if audio_file.isfile() and audio_file.name.endswith(".mp3"):
                            metadata_key = f'{audio_file.name.replace(".mp3", "")}'
                            fields = {key: metadata[metadata_key].get(key, "") for key in data_fields}
                            fields["file"] = fields["id"] + ".mp3"
                            fields["channels"] = 1
                            fields["frequency"] = 16000
                            fields["task"] = "transcribe"
                            fields["language"] = fields["text_language"]
                            fields["_post_processor"] = None
                            audio_bytes = tar.extractfile(audio_file).read()
                            audio_dict = {"bytes": audio_bytes, "path": audio_file.name}
                            metadata_dict = {
                                "id": metadata_key,
                                "audio": audio_dict,
                                **fields
                            }
                            for func in self.post_processors:
                                if func is None:
                                    yield metadata_key, metadata_dict
                                else:
                                    func_name = func.__name__ if func.__name__ else hex(id(func)).replace("0x", "lambda-")
                                    result = func(metadata_dict)
                                    if result:
                                        result["_post_processor"] = func_name
                                        yield f"{metadata_key}_{func_name}", result
