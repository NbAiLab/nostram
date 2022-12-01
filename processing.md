# Process Steps Building Corpus
This contains direct paths to internal systems. Do not expect to run this without modifications. The principles are however still useful.

```bash

python subtitle_extractor.py -f True -d /nfsmounts/datastore/ncc_speech_corpus/source_1/nrk_annotated/ -v vtt_transcribe_translate
python subtitle_extractor.py -f True -d /nfsmounts/datastore/ncc_speech_corpus/source_1/nrk_annotated/ -v vtt_translate

for f in /nfsmounts/datastore/ncc_speech_corpus/source_1/nrk_annotated/subtitles_vtt_transcribe_translate/*.json; do (cat "${f}"; echo) >> ../../../json_2/nrk/transcribe_translate.json; done
for f in /nfsmounts/datastore/ncc_speech_corpus/source_1/nrk_annotated/subtitles_vtt_translate/*.json; do (cat "${f}"; echo) >> ../../../json_2/nrk/translate.json; done


# Generate the NST dataset
python create_nst.py --input_file /nfsmounts/datastore/ncc_speech_corpus/source_1/nst/nst_test.json --output_folder /nfsmounts/datastore/ncc_speech_corpus/json_2/nst/ --mp3_folder /nfsmounts/datastore/ncc_speech_corpus/source_1/nst/NST/data/test/mp3/
python create_nst.py --input_file /nfsmounts/datastore/ncc_speech_corpus/source_1/nst/nst_train.json --output_folder /nfsmounts/datastore/ncc_speech_corpus/json_2/nst/ --mp3_folder /nfsmounts/datastore/ncc_speech_corpus/source_1/nst/NST/data/train/mp3/

```
