# Set the path to the subtitle_extractor.py script as an environment variable
EXTRACTOR_PATH="/mnt/lv_ai_1_dante/ml/pere/nostram/extractor"

# Set the path to the other processing scripts as an environment variable
PROCESSING_PATH="/mnt/lv_ai_1_dante/ml/pere/nostram/subtitle_processing"


# Set the path to the ncc_speech_corpus directory as an environment variable
CORPUS_PATH="/nfsmounts/datastore/ncc_speech_corpus"


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

# Generate the NPSC Bokm??l dataset
python $PROCESSING_PATH/create_npsc_nob.py --input_file $CORPUS_PATH/source_1/npsc/npsc_eval.json --output_folder $CORPUS_PATH/json_2//
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
# The code below copies just the files needed in one specific corpus. You might need other files here.
# python $PROCESSING_PATH/clean.py --input_file $CORPUS_PATH/json_2/nrk.json --output_folder $CORPUS_PATH/clean_json_3/test/

#Copies the entire NRK corpus to train
python $PROCESSING_PATH/clean.py --input_file $CORPUS_PATH/json_2/nrk.json --output_folder $CORPUS_PATH/clean_json_3/NCC_S/train/ --audio_input_folder $CORPUS_PATH/source_1/nrk_annotated/audio  --audio_output_folder $CORPUS_PATH/clean_json_3/NCC_S/audio/

#Copies NST train corpus to train
python $PROCESSING_PATH/clean.py --input_file $CORPUS_PATH/json_2/nst_train.json --output_folder $CORPUS_PATH/clean_json_3/NCC_S/train/ --audio_input_folder $CORPUS_PATH/source_1/nst/NST/data/train/mp3/  --audio_output_folder $CORPUS_PATH/clean_json_3/NCC_S/audio/

#Copies NPSC Bokm??l train corpus to train
python $PROCESSING_PATH/clean.py --input_file $CORPUS_PATH/json_2/npsc_train_nob.json --output_folder $CORPUS_PATH/clean_json_3/NCC_S/train/ --audio_input_folder $CORPUS_PATH/source_1/npsc/NPSC_orto/data/train/extract/audio/  --audio_output_folder $CORPUS_PATH/clean_json_3/NCC_S/audio/

#Copies Fleurs train corpus to train
#Excluded from training
#python $PROCESSING_PATH/clean.py --input_file $CORPUS_PATH/json_2/norwegian_fleurs-train.json --output_folder $CORPUS_PATH/clean_json_3/NCC_S/train/ --audio_input_folder $CORPUS_PATH/source_1/fleurs/audio/  --audio_output_folder $CORPUS_PATH/clean_json_3/NCC_S/audio/

#Copies Fleurs test corpus to test
python $PROCESSING_PATH/clean.py --input_file $CORPUS_PATH/json_2/norwegian_fleurs-test.json --output_folder $CORPUS_PATH/clean_json_3/NCC_S/test/ --audio_input_folder $CORPUS_PATH/source_1/fleurs/audio/  --audio_output_folder $CORPUS_PATH/clean_json_3/NCC_S/audio/

#Copies Fleurs validation corpus to validation
python $PROCESSING_PATH/clean.py --input_file $CORPUS_PATH/json_2/norwegian_fleurs-validation.json --output_folder $CORPUS_PATH/clean_json_3/NCC_S/validation/ --audio_input_folder $CORPUS_PATH/source_1/fleurs/audio/  --audio_output_folder $CORPUS_PATH/clean_json_3/NCC_S/audio/


## This is probably the best way of creating the needed mp3 files
cat $CORPUS_PATH/clean_json_3/NCC_S/audio/nrk_process_list.sh | xargs -P 30 -I '{}' sh -c '{}'
cat $CORPUS_PATH/clean_json_3/NCC_S/audio/norwegian_fleurs-test_process_list.sh | xargs -P 30 -I '{}' sh -c '{}'
cat $CORPUS_PATH/clean_json_3/NCC_S/audio/norwegian_fleurs-validation_process_list.sh | xargs -P 30 -I '{}' sh -c '{}'
#cat $CORPUS_PATH/clean_json_3/NCC_S/audio/norwegian_fleurs-train_process_list.sh | xargs -P 30 -I '{}' sh -c '{}'
cat $CORPUS_PATH/clean_json_3/NCC_S/audio/nst_train_process_list.sh | xargs -P 30 -I '{}' sh -c '{}'
cat $CORPUS_PATH/clean_json_3/NCC_S/audio/npsc_train_nob_process_list.sh | xargs -P 30 -I '{}' sh -c '{}'


## Create the needed mp3 files
## Currently not recommended
#python $PROCESSING_PATH/create_mp3_files.py --input_shell_script /nfsmounts/datastore/ncc_speech_corpus/clean_json_3/NCC_S/audio/norwegian_fleurs-test_process_list.sh
##python $PROCESSING_PATH/create_mp3_files.py --input_shell_script /nfsmounts/datastore/ncc_speech_corpus/clean_json_3/NCC_S/audio/norwegian_fleurs-train_process_list.sh
#python $PROCESSING_PATH/create_mp3_files.py --input_shell_script /nfsmounts/datastore/ncc_speech_corpus/clean_json_3/NCC_S/audio/norwegian_fleurs-validation_process_list.sh
#python $PROCESSING_PATH/create_mp3_files.py --input_shell_script /nfsmounts/datastore/ncc_speech_corpus/clean_json_3/NCC_S/audio/npsc_train_nob_process_list.sh
#python $PROCESSING_PATH/create_mp3_files.py --input_shell_script /nfsmounts/datastore/ncc_speech_corpus/clean_json_3/NCC_S/audio/nst_train_process_list.sh
#python $PROCESSING_PATH/create_mp3_files.py --input_shell_script /nfsmounts/datastore/ncc_speech_corpus/clean_json_3/NCC_S/audio/nrk_process_list.sh
