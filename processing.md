# Processing Scripts
You need to modify the paths below before running this script

```bash
# Set the path to the subtitle_extractor.py script as an environment variable
export EXTRACTOR_PATH="/mnt/lv_ai_1_dante/ml/pere/nostram/extractor"

# Set the path to the other processing scripts as an environment variable
export PROCESSING_PATH="/mnt/lv_ai_1_dante/ml/pere/nostram/subtitle_processing"

# Set the path to the ncc_speech_corpus directory as an environment variable
export CORPUS_PATH="/nfsmounts/datastore/ncc_speech_corpus"

# Set corpus name
export CORPUS_NAME="NCC_S2"


## SOURCE 1
# Generate NRK Subtitles
# python $EXTRACTOR_PATH/subtitle_extractor.py -f True -d $CORPUS_PATH/source_1/nrk_annotated -v vtt_transcribe_translate
# python $EXTRACTOR_PATH/subtitle_extractor.py -f True -d $CORPUS_PATH/source_1/nrk_annotated -v vtt_translate

# Faster way to the commands above
tmux new-session -d -s transcribe_translate; for i in {1..50}; do tmux new-window -t transcribe_translate:$i "python $EXTRACTOR_PATH/subtitle_extractor.py -f True -d $CORPUS_PATH/source_1/nrk_annotated -v vtt_transcribe_translate"; done
tmux new-session -d -s translate; for i in {1..50}; do tmux new-window -t translate:$i "python $EXTRACTOR_PATH/subtitle_extractor.py -f True -d $CORPUS_PATH/source_1/nrk_annotated -v vtt_translate"; done

# Download the Fleurs dataset
python $PROCESSING_PATH/download_fleurs.py --output_folder $CORPUS_PATH/source_1/fleurs
python $PROCESSING_PATH/download_fleurs.py --output_folder $CORPUS_PATH/source_1/fleurs
python $PROCESSING_PATH/download_fleurs.py --output_folder $CORPUS_PATH/source_1/fleurs


## JSON 2
#Collate the subtitles into a singe directory
rm $CORPUS_PATH/json_2/nrk.json
for f in $CORPUS_PATH/source_1/nrk_annotated/subtitles_vtt_transcribe_translate/*.json; do (cat "${f}"; echo) >> $CORPUS_PATH/json_2/nrk.json; done
for f in $CORPUS_PATH/source_1/nrk_annotated/subtitles_vtt_translate/*.json; do (cat "${f}"; echo) >> $CORPUS_PATH/json_2/nrk.json; done

# Generate the NST dataset
python $PROCESSING_PATH/create_nst.py --input_file $CORPUS_PATH/source_1/nst/nst_test.json --output_folder $CORPUS_PATH/json_2/ --mp3_folder $CORPUS_PATH/source_1/nst/NST/data/test/mp3/
python $PROCESSING_PATH/create_nst.py --input_file $CORPUS_PATH/source_1/nst/nst_train.json --output_folder $CORPUS_PATH/json_2/ --mp3_folder $CORPUS_PATH/source_1/nst/NST/data/train/mp3/

# Generate the NPSC Bokmål dataset
python $PROCESSING_PATH/create_npsc_nob.py --input_file $CORPUS_PATH/source_1/npsc/npsc_eval.json --output_folder $CORPUS_PATH/json_2/
python $PROCESSING_PATH/create_npsc_nob.py --input_file $CORPUS_PATH/source_1/npsc/npsc_test.json --output_folder $CORPUS_PATH/json_2/
python $PROCESSING_PATH/create_npsc_nob.py --input_file $CORPUS_PATH/source_1/npsc/npsc_train.json --output_folder $CORPUS_PATH/json_2/

# Generate the NPSC Nynorsk dataset
python $PROCESSING_PATH/create_npsc_nno.py --input_file $CORPUS_PATH/source_1/npsc/npsc_eval.json --output_folder $CORPUS_PATH/json_2/
python $PROCESSING_PATH/create_npsc_nno.py --input_file $CORPUS_PATH/source_1/npsc/npsc_test.json --output_folder $CORPUS_PATH/json_2/
python $PROCESSING_PATH/create_npsc_nno.py --input_file $CORPUS_PATH/source_1/npsc/npsc_train.json --output_folder $CORPUS_PATH/json_2/

#Generate the Fleurs dataset
python $PROCESSING_PATH/create_fleurs.py --input_file $CORPUS_PATH/source_1/fleurs/norwegian_fleurs-test.json --output_folder $CORPUS_PATH/json_2/ --mp3_folder $CORPUS_PATH/source_1/fleurs/audio/
python $PROCESSING_PATH/create_fleurs.py --input_file $CORPUS_PATH/source_1/fleurs/norwegian_fleurs-validation.json --output_folder $CORPUS_PATH/json_2/ --mp3_folder $CORPUS_PATH/source_1/fleurs/audio/
python $PROCESSING_PATH/create_fleurs.py --input_file $CORPUS_PATH/source_1/fleurs/norwegian_fleurs-train.json --output_folder $CORPUS_PATH/json_2/ --mp3_folder $CORPUS_PATH/source_1/fleurs/audio/

## CLEAN 3
## Here the corpus collations are directories, while the individual sub-corpora are single files
## The directory needs a config.json file to specify parameters for `clean.py`
# The code below copies just the files needed in one specific corpus. You might need other files here.
# python $PROCESSING_PATH/clean.py --input_file $CORPUS_PATH/json_2/nrk.json --output_folder $CORPUS_PATH/clean_json_3/test/

#Copies the entire NRK corpus to train
python $PROCESSING_PATH/clean.py --input_file $CORPUS_PATH/json_2/nrk.json --output_folder $CORPUS_PATH/clean_json_3/$CORPUS_NAME/train/ --audio_input_folder $CORPUS_PATH/source_1/nrk_annotated/audio  --audio_output_folder $CORPUS_PATH/clean_json_3/$CORPUS_NAME/audio/

#Copies NST train corpus to train
python $PROCESSING_PATH/clean.py --input_file $CORPUS_PATH/json_2/nst_train.json --output_folder $CORPUS_PATH/clean_json_3/$CORPUS_NAME/train/ --audio_input_folder $CORPUS_PATH/source_1/nst/NST/data/train/mp3/  --audio_output_folder $CORPUS_PATH/clean_json_3/$CORPUS_NAME/audio/

#Copies NPSC Bokmål train corpus to train
python $PROCESSING_PATH/clean.py --input_file $CORPUS_PATH/json_2/npsc_train_nob.json --output_folder $CORPUS_PATH/clean_json_3/$CORPUS_NAME/train/ --audio_input_folder $CORPUS_PATH/source_1/npsc/NPSC_orto/data/train/extract/audio/  --audio_output_folder $CORPUS_PATH/clean_json_3/$CORPUS_NAME/audio/

#Copies Fleurs train corpus to train
#Excluded from training
#python $PROCESSING_PATH/clean.py --input_file $CORPUS_PATH/json_2/norwegian_fleurs-train.json --output_folder $CORPUS_PATH/clean_json_3/$CORPUS_NAME/train/ --audio_input_folder $CORPUS_PATH/source_1/fleurs/audio/  --audio_output_folder $CORPUS_PATH/clean_json_3/$CORPUS_NAME/audio/

#Copies Fleurs test corpus to test
python $PROCESSING_PATH/clean.py --input_file $CORPUS_PATH/json_2/norwegian_fleurs-test.json --output_folder $CORPUS_PATH/clean_json_3/$CORPUS_NAME/test/ --audio_input_folder $CORPUS_PATH/source_1/fleurs/audio/  --audio_output_folder $CORPUS_PATH/clean_json_3/$CORPUS_NAME/audio/

#Copies Fleurs validation corpus to validation
python $PROCESSING_PATH/clean.py --input_file $CORPUS_PATH/json_2/norwegian_fleurs-validation.json --output_folder $CORPUS_PATH/clean_json_3/$CORPUS_NAME/validation/ --audio_input_folder $CORPUS_PATH/source_1/fleurs/audio/  --audio_output_folder $CORPUS_PATH/clean_json_3/$CORPUS_NAME/audio/

# Extra processing on the Stortinget dataset
# Adding meta-data
python $PROCESSING_PATH/stortinget_processing/add_meta.py --input_file_name $CORPUS_PATH/clean_json_3/validation/stortinget_validation.json --output_file_name $CORPUS_PATH/clean_json_3/validation/stortinget_validation_meta.json
python $PROCESSING_PATH/stortinget_processing/add_meta.py --input_file_name $CORPUS_PATH/clean_json_3/test/stortinget_test.json --output_file_name $CORPUS_PATH/clean_json_3/test/stortinget_test_meta.json
python $PROCESSING_PATH/stortinget_processing/add_meta.py --input_file_name $CORPUS_PATH/clean_json_3/train/stortinget_train.json --output_file_name $CORPUS_PATH/clean_json_3/train/stortinget_train_meta.json

## This is probably the best way of creating the needed mp3 files
cat $CORPUS_PATH/clean_json_3/$CORPUS_NAME/audio/nrk_process_list.sh | xargs -P 30 -I '{}' sh -c '{}'
cat $CORPUS_PATH/clean_json_3/$CORPUS_NAME/audio/norwegian_fleurs-test_process_list.sh | xargs -P 30 -I '{}' sh -c '{}'
cat $CORPUS_PATH/clean_json_3/$CORPUS_NAME/audio/norwegian_fleurs-validation_process_list.sh | xargs -P 30 -I '{}' sh -c '{}'
#cat $CORPUS_PATH/clean_json_3/$CORPUS_NAME/audio/norwegian_fleurs-train_process_list.sh | xargs -P 30 -I '{}' sh -c '{}'
cat $CORPUS_PATH/clean_json_3/$CORPUS_NAME/audio/nst_train_process_list.sh | xargs -P 30 -I '{}' sh -c '{}'
cat $CORPUS_PATH/clean_json_3/$CORPUS_NAME/audio/npsc_train_nob_process_list.sh | xargs -P 30 -I '{}' sh -c '{}'


## Create the needed mp3 files
## Currently not recommended
#python $PROCESSING_PATH/create_mp3_files.py --input_shell_script /nfsmounts/datastore/ncc_speech_corpus/clean_json_3/$CORPUS_NAME/audio/norwegian_fleurs-test_process_list.sh
##python $PROCESSING_PATH/create_mp3_files.py --input_shell_script /nfsmounts/datastore/ncc_speech_corpus/clean_json_3/$CORPUS_NAME/audio/norwegian_fleurs-train_process_list.sh
#python $PROCESSING_PATH/create_mp3_files.py --input_shell_script /nfsmounts/datastore/ncc_speech_corpus/clean_json_3/$CORPUS_NAME/audio/norwegian_fleurs-validation_process_list.sh
#python $PROCESSING_PATH/create_mp3_files.py --input_shell_script /nfsmounts/datastore/ncc_speech_corpus/clean_json_3/$CORPUS_NAME/audio/npsc_train_nob_process_list.sh
#python $PROCESSING_PATH/create_mp3_files.py --input_shell_script /nfsmounts/datastore/ncc_speech_corpus/clean_json_3/$CORPUS_NAME/audio/nst_train_process_list.sh
#python $PROCESSING_PATH/create_mp3_files.py --input_shell_script /nfsmounts/datastore/ncc_speech_corpus/clean_json_3/$CORPUS_NAME/audio/nrk_process_list.sh
```

# Create the merged transcript files - Currently just on a small test file
python $PROCESSING_PATH/add_transcriptions.py --input_file $CORPUS_PATH/clean_json_3/$CORPUS_NAME/train/nrk_small.json --transcript_file $CORPUS_PATH/clean_json_3/$CORPUS_NAME/train/nrk_wav2vec_transcript_small.json --output_folder $CORPUS_PATH/transcribed_json_4/$CORPUS_NAME/train/

# Manual changes to the files
jq -c 'select(.id | IN("stortinget-20100114-094334_8384400_8411300", "stortinget-20100302-100000_15268100_15293600", "stortinget-20100614-095552_2m_del_1_13521400_13547400", "stortinget-20110321-115501_2m_7288900_7310700", "stortinget-20111209-085501_6667700_6689000", "stortinget-20121025-095235_5761300_5790200", "stortinget-20121121-095439_5556900_5586500", "stortinget-20130307-095509_3652050_3680000", "stortinget-20141211-155720_12709000_12734100", "stortinget-20150217-095731_22209600_22210400", "stortinget-20150616-155450_7698500_7725000", "stortinget-20160217-095626_10071800_10100100", "stortinget-20170328-095500_15454300_15481400", "stortinget-20170613-085456_3097100_3124100", "stortinget-20181210-152505_14013700_14037800", "stortinget-20190604-155520_8092400_8119700", "stortinget-20200421-115527_15503200_15530100", "stortinget-20210427-095501_21558200_21584300", "stortinget-20210615-095521_1324000_1350500", "stortinget-20220511-095359_14382600_14410900") | not)' input.jsonl > output.jsonl

jq -c 'select(.id | IN("no19x173-07071999-1452_u0173083", "no12x767-02071999-0853_u0767043", "no20x404-05081999-1242_u0404193") | not)' input.jsonl > output.jsonl

