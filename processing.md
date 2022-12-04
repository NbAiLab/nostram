# Process Steps Building Corpus
This contains direct paths to internal systems. Do not expect to run this without modifications. The principles are however still useful.

```bash
## SOURCE 1
# Generate NRK Subtitles
# python /mnt/lv_ai_1_dante/ml/pere/nostram/extractor/subtitle_extractor.py -f True -d /nfsmounts/datastore/ncc_speech_corpus/source_1/nrk_annotated -v vtt_transcribe_translate
# python /mnt/lv_ai_1_dante/ml/pere/nostram/extractor/subtitle_extractor.py -f True -d /nfsmounts/datastore/ncc_speech_corpus/source_1/nrk_annotated -v vtt_translate

# Faster way to the commands above
tmux new-session -d -s transcribe_translate; for i in {1..50}; do tmux new-window -t transcribe_translate:$i "python /mnt/lv_ai_1_dante/ml/pere/nostram/extractor/subtitle_extractor.py -f True -d /nfsmounts/datastore/ncc_speech_corpus/source_1/nrk_annotated -v vtt_transcribe_translate"; done
tmux new-session -d -s translate; for i in {1..50}; do tmux new-window -t translate:$i "python /mnt/lv_ai_1_dante/ml/pere/nostram/extractor/subtitle_extractor.py -f True -d /nfsmounts/datastore/ncc_speech_corpus/source_1/nrk_annotated -v vtt_translate"; done


## JSON 2
#Collate the subtitles into a singe directory
rm /nfsmounts/datastore/ncc_speech_corpus/json_2/nrk.json
for f in /nfsmounts/datastore/ncc_speech_corpus/source_1/nrk_annotated/subtitles_vtt_transcribe_translate/*.json; do (cat "${f}"; echo) >> /nfsmounts/datastore/ncc_speech_corpus/json_2/nrk.json; done
for f in /nfsmounts/datastore/ncc_speech_corpus/source_1/nrk_annotated/subtitles_vtt_translate/*.json; do (cat "${f}"; echo) >> /nfsmounts/datastore/ncc_speech_corpus/json_2/nrk.json; done

# Generate the NST dataset
python /mnt/lv_ai_1_dante/ml/pere/nostram/subtitle_processing/create_nst.py --input_file /nfsmounts/datastore/ncc_speech_corpus/source_1/nst/nst_test.json --output_folder /nfsmounts/datastore/ncc_speech_corpus/json_2/ --mp3_folder /nfsmounts/datastore/ncc_speech_corpus/source_1/nst/NST/data/test/mp3/
python /mnt/lv_ai_1_dante/ml/pere/nostram/subtitle_processing/create_nst.py --input_file /nfsmounts/datastore/ncc_speech_corpus/source_1/nst/nst_train.json --output_folder /nfsmounts/datastore/ncc_speech_corpus/json_2/ --mp3_folder /nfsmounts/datastore/ncc_speech_corpus/source_1/nst/NST/data/train/mp3/

# Generate the NPSC Bokmål dataset
python /mnt/lv_ai_1_dante/ml/pere/nostram/subtitle_processing/create_npsc_nob.py --input_file /nfsmounts/datastore/ncc_speech_corpus/source_1/npsc/npsc_eval.json --output_folder /nfsmounts/datastore/ncc_speech_corpus/json_2//
python /mnt/lv_ai_1_dante/ml/pere/nostram/subtitle_processing/create_npsc_nob.py --input_file /nfsmounts/datastore/ncc_speech_corpus/source_1/npsc/npsc_test.json --output_folder /nfsmounts/datastore/ncc_speech_corpus/json_2/
python /mnt/lv_ai_1_dante/ml/pere/nostram/subtitle_processing/create_npsc_nob.py --input_file /nfsmounts/datastore/ncc_speech_corpus/source_1/npsc/npsc_train.json --output_folder /nfsmounts/datastore/ncc_speech_corpus/json_2/

# Generate the NPSC Nynorsk dataset
python /mnt/lv_ai_1_dante/ml/pere/nostram/subtitle_processing/create_npsc_nno.py --input_file /nfsmounts/datastore/ncc_speech_corpus/source_1/npsc/npsc_eval.json --output_folder /nfsmounts/datastore/ncc_speech_corpus/json_2/
python /mnt/lv_ai_1_dante/ml/pere/nostram/subtitle_processing/create_npsc_nno.py --input_file /nfsmounts/datastore/ncc_speech_corpus/source_1/npsc/npsc_test.json --output_folder /nfsmounts/datastore/ncc_speech_corpus/json_2/
python /mnt/lv_ai_1_dante/ml/pere/nostram/subtitle_processing/create_npsc_nno.py --input_file /nfsmounts/datastore/ncc_speech_corpus/source_1/npsc/npsc_train.json --output_folder /nfsmounts/datastore/ncc_speech_corpus/json_2/

## CLEAN 3
## Here the corpus collations are directories, while the individual sub-corpora are single files
python /mnt/lv_ai_1_dante/ml/pere/nostram/subtitle_processing/clean.py --input_file /nfsmounts/datastore/ncc_speech_corpus/json_2/nrk.json --output_folder /nfsmounts/datastore/ncc_speech_corpus/clean_json_3/test/


```
