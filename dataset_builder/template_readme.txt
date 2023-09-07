
YAML tags: null
annotations_creators:
  - no-annotation
language_creators:
  - found
language:
  - 'no'
  - nn
 
license:
  - Other
multilinguality:
  - multilingual
pretty_name:  <dataset_name>
size_categories:
  - 2G<n<1B
source_datasets:
  - original
task_categories:
  - sequence-modeling
task_ids:
  - language-modeling

# Dataset Card: NbAiLab/<dataset_name>
# - Internal version <versionnumber>

## General Information
The <dataset_name> (Norwegian Colossal Corpus - Speech) is a speech corpus published by the National Library of Norway created as part of the [Norwegian Speech Transformer Models](https://huggingface.co/datasets/NbAiLab) (NoSTraM) project. The corpus was intended for training automatic speech recognition (ASR) models, and consists of a total of <sumduration> hours (<noclips> individual sound clips) and their transcriptions in Norwegian Bokmål.

## Dataset Sources
The <dataset_name> corpus has the following features:

*  Subtitles from NRK. (<nrkduration> hours - <nrkclips> clips). The clips are obtained using the NRK subtitle API. The method for aligning is based on software originally developed as part of the Media Future Project. More details about the procedure and the scripts used can be found [here](https://github.com/NbAiLab/nostram/tree/main/extractor).

* The version of the [NST-corpus](https://huggingface.co/datasets/NbAiLab/NST) where all non-complete sentences are removed. (<nstduration> hours - <nstclips> clips)

* The Bokmål part of the [NPSC-corpus](https://huggingface.co/datasets/NbAiLab/NPSC), that is post-processed to add punctuation and capitalisation. (<npscduration> hours - <npscclips> clips)


## Potential Use Cases
The <dataset_name> corpus can be used for various purposes, including but not limited to:

- Training Automatic Speech Recognition models.
- Building text-to-speech systems.
- Research in speech recognition and natural language processing.
- Developing language models.

## License
The <dataset_name> corpus is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. 

## Citation
The corpus was created and cleaned by Freddy Wetjen, Rolv-Arild Braaten and Per Egil Kummervold. No publication is so far published based on this copus. 

