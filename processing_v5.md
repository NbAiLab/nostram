# ncc_speech_5
Main dataset as of September 2023. Please see the complete description of [Corpus Structure](corpus_structure.md)

# Current Status
The status can be updated by running ```python nostram/utils/json_stats.py /mnt/lv_ai_1_ficino/ml/ncc_speech_v5```.

## Target Directory: /mnt/lv_ai_1_ficino/ml/ncc_speech_v5
### Directory: clean_3
| Directory | File | Lines     |
| --------- | ---- | ---------:|
| clean_3 | <empty> | <empty>    |
| clean_3/audio_books | <empty> | <empty>    |
| clean_3/audio_books/test | audio_books_no_test.json |      1,500 |
| clean_3/audio_books/test | audio_books_nn_test.json |      1,500 |
| clean_3/audio_books/train | audio_books_no_train.json |  1,128,637 |
| clean_3/audio_books/train | audio_books_nn_train.json |     12,208 |
| clean_3/audio_books/validation | audio_books_nn_validation.json |      1,500 |
| clean_3/audio_books/validation | audio_books_no_validation.json |      1,500 |
| clean_3/fleurs | norwegian_fleurs-test.json |        357 |
| clean_3/fleurs | norwegian_fleurs-validation.json |        163 |
| clean_3/nrk_tv | <empty> | <empty>    |
| clean_3/nrk_tv/both | <empty> | <empty>    |
| clean_3/nrk_tv/short | config.json |         26 |
| clean_3/nrk_tv/short/log | <empty> | <empty>    |
| clean_3/nrk_tv/standard | config.json |         26 |
| clean_3/nrk_tv/standard/log | <empty> | <empty>    |
| clean_3/nrk_tv_old | <empty> | <empty>    |
| clean_3/nrk_tv_old/both | nrk.json |          0 |
| clean_3/nrk_tv_old/short | nrk.json |  7,737,160 |
| clean_3/nrk_tv_old/short | config.json |         26 |
| clean_3/nrk_tv_old/short/log | <empty> | <empty>    |
| clean_3/nrk_tv_old/standard | nrk.json |  4,074,362 |
| clean_3/nrk_tv_old/standard | config.json |         26 |
| clean_3/nrk_tv_old/standard/log | <empty> | <empty>    |
| clean_3/nst | nst_train.json |    144,546 |
| clean_3/nst | nst_largetest.json |     31,332 |
| clean_3/nst | nst_validation.json |      1,500 |
| clean_3/nst | nst_test.json |      1,500 |
| clean_3/silence | silence.json |    109,019 |
| clean_3/stortinget | stortinget_train.json |    580,827 |
| clean_3/stortinget | stortinget_test.json |      1,520 |
| clean_3/stortinget | stortinget_validation.json |      1,714 |
| **Total** |      | **13,830,949** |

### Directory: inference_4
| Directory | File | Lines     |
| --------- | ---- | ---------:|
| inference_4 | <empty> | <empty>    |
| inference_4/inference_dataset | <empty> | <empty>    |
| inference_4/inference_dataset/audio_books | <empty> | <empty>    |
| inference_4/inference_dataset/audio_books/test | audio_books_test.json |      1,500 |
| inference_4/inference_dataset/audio_books/train | audio_books_train.json |  1,145,530 |
| inference_4/inference_dataset/audio_books/validation | audio_books_validation.json |      1,500 |
| inference_4/inference_dataset/fleurs | <empty> | <empty>    |
| inference_4/inference_dataset/fleurs/test | norwegian_fleurs-test.json |        357 |
| inference_4/inference_dataset/fleurs/validation | norwegian_fleurs-validation.json |        163 |
| inference_4/inference_dataset/nrk_tv | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv/{train} | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_nn | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_nn/test | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_nn/validation | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_no | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_no/test | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_no/validation | <empty> | <empty>    |
| inference_4/inference_dataset/nst | <empty> | <empty>    |
| inference_4/inference_dataset/nst/test | nst_test.json |      1,500 |
| inference_4/inference_dataset/nst/train | nst_train.json |    144,546 |
| inference_4/inference_dataset/nst/validation | nst_validation.json |      1,500 |
| inference_4/inference_dataset/silence | <empty> | <empty>    |
| inference_4/inference_dataset/silence/test | silence_test.json |      1,000 |
| inference_4/inference_dataset/silence/train | silence_train.json |    107,019 |
| inference_4/inference_dataset/silence/validation | silence_validation.json |      1,000 |
| inference_4/inference_dataset/stortinget | <empty> | <empty>    |
| inference_4/inference_dataset/stortinget/train | stortinget_train.json |    580,827 |
| inference_4/inference_dataset/stortinget_no | <empty> | <empty>    |
| inference_4/inference_dataset/stortinget_no/test | stortinget_no_test.json |      1,402 |
| inference_4/inference_dataset/stortinget_no/validation | stortinget_validation_test.json |      1,545 |
| inference_4/inference_result | <empty> | <empty>    |
| inference_4/processed | <empty> | <empty>    |
| **Total** |      | **1,989,389** |

### Directory: translation_5
| Directory | File | Lines     |
| --------- | ---- | ---------:|
| translation_5 | <empty> | <empty>    |
| translation_5/processed | <empty> | <empty>    |
| translation_5/translation_files | <empty> | <empty>    |
| **Total** |      | ** 0** |



# Copy Structure
The following command creates all the necssary folders if they do not exist.

```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5"
mkdir -p "$base_dir"/{clean_3/{nrk_tv/{standard,short,both,mp3},stortinget,fleurs,nst,audio_books},inference_4/{mp3/{nrk_tv,stortinget,fleurs,nst,audio_books},inference_dataset/{nrk_tv/train,nrk_tv_no/{test,validation},nrk_tv_nn/{test,validation},stortinget/train,stortinget_no/{test,validation},stortinget_nn/{test,validation},fleurs/{test,validation},nst/{train,test,validation},audio_books/train,audio_books_no/{test,validation},audio_books_nn/{test,validation}},inference_result,processed},translation_5/{translation_files,processed}}


```
This should create the following structure:
```plaintext
$base_dir/
|-- clean_3/
|   |-- nrk_tv/
|   |   |-- standard
|   |   |-- short
|   |   |-- both
|   |   |-- mp3
|   |-- stortinget
|   |-- fleurs
|   |-- nst
|   |-- audio_books
|-- inference_4/
|   |-- mp3
|   |   |-- nrk_tv
|   |   |-- stortinget
|   |   |-- fleurs
|   |   |-- nst
|   |   |-- audio_books
|   |-- inference_dataset
|   |   |-- nrk_tv
|   |   |   |-- train
|   |   |-- nrk_tv_no
|   |   |   |-- test
|   |   |   |-- validation
|   |   |-- nrk_tv_nn
|   |   |   |-- test
|   |   |   |-- validation
|   |   |-- stortinget
|   |   |   |-- train
|   |   |-- stortinget_no
|   |   |   |-- test
|   |   |   |-- validation
|   |   |-- stortinget_nn
|   |   |   |-- test
|   |   |   |-- validation
|   |   |-- fleurs
|   |   |   |-- test
|   |   |   |-- validation
|   |   |-- nst
|   |   |   |-- train
|   |   |   |-- test
|   |   |   |-- validation
|   |   |-- audio_books
|   |   |   |-- train
|   |   |-- audio_books_no
|   |   |   |-- test
|   |   |   |-- validation
|   |   |-- audio_books_nn
|   |   |   |-- test
|   |   |   |-- validation
|   |-- inference_result
|   |-- processed
|   |-- inference_corpus
|   |   |-- train
|   |   |-- test
|   |   |-- validation
|-- translation_5/
    |-- translation_files
    |-- processed

```


# raw_1 and partly json_2
Not needed in v5. If needed, content needs to be copied from ```ncc_speech_corpus``` and ```ncc_speech_corpus2```.

# json_2
Audiobooks are stored in json_2 on a form

# clean_3
### Fleurs
Fleurs data is copied unmodified from ```ncc_speech_corpus/json_2```. The clean script would have changed several of the transcripts. However, it is kept unchanged here to be able to follow the development over time.
```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
archive_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_corpus2/transcribed_json_4";

cp $archive_dir/fleurs/validation/norwegian_fleurs-validation.json $base_dir/clean_3/fleurs/;
cp $archive_dir/fleurs/test/norwegian_fleurs-test.json $base_dir/clean_3/fleurs/;
```

### Stortinget
The ```clean_text-script``` is used to copy data from ```ncc_speech_corpus2/clean_json_3```. Some additional processing is done on this set. For instance have NER been run. 

```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
clean_text_dir="/mnt/lv_ai_1_ficino/ml/perk/nostram/utils";
archive_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_corpus2/clean_json_3";

python $clean_text_dir/clean_text.py --input_file $archive_dir/stortinget/test/stortinget_test_cleaned.json --output_file $base_dir/clean_3/stortinget/stortinget_test.json;
python $clean_text_dir/clean_text.py --input_file $archive_dir/stortinget/validation/stortinget_validation_cleaned.json --output_file $base_dir/clean_3/stortinget/stortinget_validation.json;
python $clean_text_dir/clean_text.py --input_file $archive_dir/stortinget/train/stortinget_train_cleaned.json --output_file $base_dir/clean_3/stortinget/stortinget_train.json;

#python $clean_text_dir/clean_text.py --input_file $archive_dir/stortinget_eval.json --output_file #$base_dir/clean_3/stortinget/stortinget_validation.json;
#python $clean_text_dir/clean_text.py --input_file $archive_dir/stortinget_train.json --output_file $base_dir/clean_3/stortinget/stortinget_train.json;

```
### NST
The ```clean_text-script``` is used to copy data from ```ncc_speech_corpus2/transcribed_json_3```. 

```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
clean_text_dir="/mnt/lv_ai_1_ficino/ml/perk/nostram/utils";
archive_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_corpus2/transcribed_json_4";

python $clean_text_dir/clean_text.py --input_file $archive_dir/nst/test/nst_test.json --output_file $base_dir/clean_3/nst/nst_largetest.json;
python $clean_text_dir/clean_text.py --input_file $archive_dir/nst/train/nst_train.json --output_file $base_dir/clean_3/nst/nst_train.json;

# Reduce the size of the NST validation and test set
sed -n '1,1500p' $base_dir/clean_3/nst/nst_largetest.json > $base_dir/clean_3/nst/nst_test.json;
sed -n '1501,3000p' $base_dir/clean_3/nst/nst_largetest.json > $base_dir/clean_3/nst/nst_validation.json;
```
### SILENCE
This can just be copied from ```ncc_speech_corpus2/transcribed_json_3```. No reason to clean.
```bash
archive_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_corpus2/clean_json_3";
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";

cp $archive_dir/nrk/nrk_tv_silence.json $base_dir/clean_3/silence/silence.json

```


### NRK TV
```bash
# Set working dirs
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
program_dir="/mnt/lv_ai_1_ficino/ml/perk/nostram/subtitle_processing";
audio_dir="/nfsmounts/datastore/ncc_speech_corpus/source_1/nrk_annotated/audio";
archive_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_corpus/json_2";
final_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5/inference_4/inference_processed/ncc_speech_v7"

# Create the config.json with these settings:
echo -e "{\n\t\"max_duplicates_text_program\": 10,\n\t\"min_alphawords_subtitle\": 0,\n\t\"min_length_subtitle\": 1,\n\t\"min_words_subtitle\": 0,\n\t\"normalise_unicode\": true,\n\t\"drop_subtitles_with_encoding_errors\": true,\n\t\"drop_subtitles_with_curly_brackets\": true,\n\t\"simultaneous_subtitles\": \"delete\",\n\t\"task\": [\"transcribe\", \"translate\"],\n\t\"drop_italics\": true,\n\t\"drop_inaudible\": true,\n\t\"drop_invalid_durations\": true,\n\t\"merge_subtitles\": true,\n\t\"drop_multiple_speakers\": false,\n\t\"combine_continued_sentences\": false,\n\t\"make_bigger_segments\": true,\n\t\"target_duration_seconds\": 28,\n\t\"max_duration_seconds\": 29,\n\t\"pad_with_silence\": true,\n\t\"add_empty_captions\": true,\n\t\"detect_lang_text\": true,\n\t\"allow_lang_text\": [\"nob\", \"nno\"],\n\t\"remove_cpossible\": true,\n\t\"max_separation_seconds\": 5\n}" > $base_dir/clean_3/nrk_tv/standard/config.json;
echo -e "{\n\t\"max_duplicates_text_program\": 10,\n\t\"min_alphawords_subtitle\": 0,\n\t\"min_length_subtitle\": 1,\n\t\"min_words_subtitle\": 0,\n\t\"normalise_unicode\": true,\n\t\"drop_subtitles_with_encoding_errors\": true,\n\t\"drop_subtitles_with_curly_brackets\": true,\n\t\"simultaneous_subtitles\": \"delete\",\n\t\"task\": [\"transcribe\", \"translate\"],\n\t\"drop_italics\": true,\n\t\"drop_inaudible\": true,\n\t\"drop_invalid_durations\": true,\n\t\"merge_subtitles\": true,\n\t\"drop_multiple_speakers\": false,\n\t\"combine_continued_sentences\": false,\n\t\"make_bigger_segments\": false,\n\t\"target_duration_seconds\": 28,\n\t\"max_duration_seconds\": 29,\n\t\"pad_with_silence\": true,\n\t\"add_empty_captions\": true,\n\t\"detect_lang_text\": true,\n\t\"allow_lang_text\": [\"nob\", \"nno\"],\n\t\"remove_cpossible\": true,\n\t\"max_separation_seconds\": 5\n}" > $base_dir/clean_3/nrk_tv/short/config.json;

# Clean the files - This takes roughly 4 hours
python $program_dir/clean.py --input_file $archive_dir/nrk.json --output_folder $base_dir/clean_3/nrk_tv/standard --audio_input_folder $audio_dir  --audio_output_folder $base_dir/clean_3/nrk_tv/mp3_standard/;
python $program_dir/clean.py --input_file $archive_dir/nrk.json --output_folder $base_dir/clean_3/nrk_tv/short --audio_input_folder $audio_dir  --audio_output_folder $base_dir/clean_3/nrk_tv/mp3_short/;

# Concatenate files and remove duplicates (can be extended with extra files)
# Not that in the current example these are not the same files as above. This should be changed if it is run again.
cat $base_dir/clean_3/nrk_tv/standard/nrk.json $base_dir/clean_3/nrk_tv/short/nrk.json | jq -c . | sort -k1,1 -s | awk '!seen[$1]++' > $base_dir/clean_3/nrk_tv/both/nrk.json;

# We also need to create clean Bokmål/Nynorsk test and validation files.
jq -c 'select(.text_language=="nn")' $base_dir/clean_3/nrk_tv/both/nrk.json > $base_dir/clean_3/nrk_tv/both/nrk_nn.json
jq -c 'select(.text_language=="no")' $base_dir/clean_3/nrk_tv/both/nrk.json > $base_dir/clean_3/nrk_tv/both/nrk_no.json
shuf "$base_dir/clean_3/nrk_tv/both/nrk_nn.json" | awk -v base_dir="$base_dir" 'NR <= 1500 {print > base_dir "/clean_3/nrk_tv/both/nrk_nn_test.json"} NR > 1500 && NR <= 3000 {print > base_dir "/clean_3/nrk_tv/both/nrk_nn_validation.json"} NR > 3000 {print > base_dir "/clean_3/nrk_tv/both/nrk_nn_train.json"}'
shuf "$base_dir/clean_3/nrk_tv/both/nrk_no.json" | awk -v base_dir="$base_dir" 'NR <= 1500 {print > base_dir "/clean_3/nrk_tv/both/nrk_no_test.json"} NR > 1500 && NR <= 3000 {print > base_dir "/clean_3/nrk_tv/both/nrk_no_validation.json"} NR > 3000 {print > base_dir "/clean_3/nrk_tv/both/nrk_no_train.json"}'


# Create the audio files
# It might be safer to create two folders here, and generate both of them separately
cat $base_dir/clean_3/nrk_tv/mp3/nrk_process_list.sh | xargs -P 30 -I '{}' sh -c '{}';
```

### Validate all JSON files
```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
program_dir="/mnt/lv_ai_1_ficino/ml/perk/nostram/utils";
eval_samples_nr=1000

# Typically it is not necessary to validate the entire file, since all the lines in a file has the same structure

#Fleurs
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/fleurs/norwegian_fleurs-test.json
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/fleurs/norwegian_fleurs-validation.json
#NST
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/nst/nst_train.json
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/nst/nst_test.json
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/nst/nst_validation.json
#Stortinget
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/stortinget/stortinget_train.json
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/stortinget/stortinget_test.json
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/stortinget/stortinget_validation.json
#NRK TV
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/nrk_tv/both/nrk.json
# Audio Books
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/audio_books/train/audio_books_nn_train.json;
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/audio_books/train/audio_books_no_train.json;
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/audio_books/test/audio_books_nn_test.json;
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/audio_books/test/audio_books_no_test.json;
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/audio_books/validation/audio_books_nn_validation.json;
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/audio_books/validation/audio_books_no_validation.json;
#Silence
python $program_dir/validate_dataset.py -n $eval_samples_nr $base_dir/clean_3/silence/silence.json;
```

# inference_4

### Stortinget
Here we need to make a test and validation dataset that contains only Norwegian Bokmål
```bash
#Stortinget
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
cp $base_dir/clean_3/stortinget/stortinget_train.json $base_dir/inference_4/inference_dataset/stortinget/train/;
jq -c 'select(.text_language=="no")' $base_dir/clean_3/stortinget/stortinget_test.json > $base_dir/inference_4/inference_dataset/stortinget_no/test/stortinget_no_test.json;
jq -c 'select(.text_language=="no")' $base_dir/clean_3/stortinget/stortinget_validation.json > $base_dir/inference_4/inference_dataset/stortinget_no/validation/stortinget_no_validation.json;
jq -c 'select(.text_language=="nn")' $base_dir/clean_3/stortinget/stortinget_test.json > $base_dir/inference_4/inference_dataset/stortinget_nn/test/stortinget_nn_test.json;
jq -c 'select(.text_language=="nn")' $base_dir/clean_3/stortinget/stortinget_validation.json > $base_dir/inference_4/inference_dataset/stortinget_nn/validation/stortinget_nn_validation.json;
```

### Fleurs, NST and NRK
No more processing is needed here. Just copy the correct files to the correct folder

```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";

#Fleurs
cp $base_dir/clean_3/fleurs/norwegian_fleurs-test.json $base_dir/inference_4/inference_dataset/fleurs/test/;
cp $base_dir/clean_3/fleurs/norwegian_fleurs-validation.json $base_dir/inference_4/inference_dataset/fleurs/validation/;
#NST
cp $base_dir/clean_3/nst/nst_train.json $base_dir/inference_4/inference_dataset/nst/train/
cp $base_dir/clean_3/nst/nst_test.json $base_dir/inference_4/inference_dataset/nst/test/
cp $base_dir/clean_3/nst/nst_validation.json $base_dir/inference_4/inference_dataset/nst/validation/
#NRK
cp $base_dir/clean_3/nrk_tv/both/nrk_nn_test.json $base_dir/inference_4/inference_dataset/nrk_tv_nn/test/
cp $base_dir/clean_3/nrk_tv/both/nrk_nn_validation.json $base_dir/inference_4/inference_dataset/nrk_tv_nn/validation/
cp $base_dir/clean_3/nrk_tv/both/nrk_no_test.json $base_dir/inference_4/inference_dataset/nrk_tv_no/test/
cp $base_dir/clean_3/nrk_tv/both/nrk_no_validation.json $base_dir/inference_4/inference_dataset/nrk_tv_no/validation/
cp $base_dir/clean_3/nrk_tv/both/nrk_nn_train.json $base_dir/inference_4/inference_dataset/nrk_tv/train/
cp $base_dir/clean_3/nrk_tv/both/nrk_no_train.json $base_dir/inference_4/inference_dataset/nrk_tv/train/

```

### Copy files to inference_corpus
```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
corpus_name="ncc_speech_inference_v5"
rm $base_dir/inference_4/inference_corpus/$corpus_name/train/*.json
rm $base_dir/inference_4/inference_corpus/$corpus_name/test/*.json
rm $base_dir/inference_4/inference_corpus/$corpus_name/validation/*.json
cp $base_dir/inference_4/inference_dataset/*/train/*.json $base_dir/inference_4/inference_corpus/$corpus_name/train/
cp $base_dir/inference_4/inference_dataset/*/test/*.json $base_dir/inference_4/inference_corpus/$corpus_name/test/
cp $base_dir/inference_4/inference_dataset/*/validation/*.json $base_dir/inference_4/inference_corpus/$corpus_name/validation/
```

### MP3
We will copy the mp3-files from earlier versions
```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
archive_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_corpus2";

cp -r $archive_dir/transcribed_json_4/audio/nst/audio/* $base_dir/inference_4/mp3/nst/.;
cp -r $archive_dir/transcribed_json_4/audio/fleurs/audio/*.* $base_dir/inference_4/mp3/fleurs/.;
cp -r $archive_dir/transcribed_json_4/audio/stortinget/audio/* $base_dir/inference_4/mp3/stortinget/.;
cp -rf $base_dir/clean_3/nrk_tv/mp3/* $base_dir/inference_4/mp3/nrk_tv/.;

```

### Validate MP3
Since JSON already is validated, we concentrate on evaluating mp3-files
```bash
program_dir="/mnt/lv_ai_1_ficino/ml/perk/nostram/utils";
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
corpus_name="ncc_speech_inference_v5"

for f in $base_dir/inference_4/inference_corpus/$corpus_name/*/*.json; do python $program_dir/validate_mp3.py "$f"; done
```

# Generate Dataset for Inference
Now we are ready to do the generation of the actual dataset. Freddy fills in this.

# Run Inference
To be describe

# Process Inference Results
### Download
```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
bucket="gs://nb-whisper-transcript/";

cd $base_dir/inference_4/inference_result/downloads
gsutil -m cp -r $bucket/M* .
```
### Merge
```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
program_dir="/home/perk/nostram/utils";
result_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5/inference_4/inference_result/merged";
tsv_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5/inference_4/inference_result/downloads/"
for f in $base_dir/inference_4/inference_corpus/ncc_speech_inference_v5/train/*train.json; do python $program_dir/merge_pseudo_labels.py --input_json $f --input_tsv_dir $tsv_dir --output_json "$result_dir/${f##*/}"; done


```

### Process
Before processing, we need to copy the right config.json
```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
program_dir="/home/perk/nostram/utils";
result_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5/inference_4/inference_processed/ncc_speech_v7";

# Clean corpus using unmodified train/validation files (these do not need config.json)
for f in $base_dir/inference_4/inference_result/merged/*train*.json; do python /home/perk/nostram/utils/post_clean.py --input_filename $f --output_folder $result_dir/train --prune; done

# Lets also take some test and validation files
for f in $base_dir/inference_4/inference_result/merged/*train*.json; do python /home/perk/nostram/utils/post_clean.py --input_filename $f --output_folder $result_dir/train --prune; done

#And the Fleurs files
python /home/perk/nostram/utils/post_clean.py --input_filename /mnt/lv_ai_1_ficino/ml/ncc_speech_v5/inference_4/inference_processed/test/fleurs/norwegian_fleurs-test.json --output_folder $result_dir/test --prune
python /home/perk/nostram/utils/post_clean.py --input_filename /mnt/lv_ai_1_ficino/ml/ncc_speech_v5/inference_4/inference_processed/validation/fleurs/norwegian_fleurs-validation.json --output_folder $result_dir/validation --prune

# Very clean corpus
# result_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5/inference_4/inference_processed/ncc_speech_v6_veryclean";
# for f in $base_dir/inference_4/inference_result/merged/*train*.json; do python /home/perk/nostram/utils/post_clean.py --input_filename $f --output_folder $result_dir/train --prune; done

```

### Translate
We will need some files to translate. We will use the nrk_tv since these are all in Norwegian. We will also filter out the Norwegian Bokmål-part, since this will work best.
```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
program_dir="/home/perk/nostram";
result_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5/translation_5/translation_files";
translated_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5/translation_5/translated"
processed_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5/translation_5/processed/train"
final_dir="final_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5/translation_5/final/ncc_speech_v7"

# Extract the parts we want to translate from NRK
jq -c 'select(.source == "nrk_tv" and .text_language=="no")' $base_dir/inference_4/inference_processed/ncc_speech_v7/train/nrk_no_train.json > $result_dir/nrk_no_train_nrk_tv.json

# Do the same for Stortinget
jq -c 'select(.source == "stortinget" and .text_language=="no")' $base_dir/inference_4/inference_processed/ncc_speech_v7/train/stortinget_train.json > $result_dir/stortinget_train_no.json

# Audio books and nst can just be copies

# nst can just be copied since it is Bokmål only

# This will give a file of roughly 3M lines, or 350M characters. The price is $20 per million characters, so this should be roughly $7k.
# Lets convert it to the needed tsv-format. Note that this is considerably shorter (in lines), since we are grouping by id
python $program_dir/translate/create_translate_tsv.py --input_file_name $result_dir/nrk_no_train_nrk_tv.json --output_file_name $result_dir/nrk_no_train_nrk_tv.tsv

python $program_dir/translate/create_translate_tsv.py --input_file_name $result_dir/audio_books_no_train.json --output_file_name $result_dir/audio_books_no_train.tsv --target_field text

python $program_dir/translate/create_translate_tsv.py --input_file_name $result_dir/nst_train.json --output_file_name $result_dir/nst_train.tsv --target_field text

python $program_dir/translate/create_translate_tsv.py --input_file_name $result_dir/stortinget_train_no.json --output_file_name $result_dir/stortinget_train_no.tsv --target_field text

# Split the fields so they are below the Google Translate limit
head -n 5000 $result_dir/audio_books_no_train.tsv > $result_dir/audio_books_no_train1.tsv
tail -n +5001 $result_dir/audio_books_no_train.tsv | head -n 5000 > $result_dir/audio_books_no_train2.tsv
tail -n +10001 $result_dir/audio_books_no_train.tsv > $result_dir/audio_books_no_train3.tsv
head -n 5000 $result_dir/stortinget_train_no.tsv > $result_dir/stortinget_train_no1.tsv
tail -n +5001 $result_dir/stortinget_train_no.tsv > $result_dir/stortinget_train_no2.tsv

# Transfer all the files to the Google bucket
gsutil cp audio_books_no_train1.tsv gs://mtrans/audio_books_no_train1/
gsutil cp audio_books_no_train2.tsv gs://mtrans/audio_books_no_train2/
gsutil cp audio_books_no_train3.tsv gs://mtrans/audio_books_no_train3/
gsutil cp stortinget_train_no1.tsv gs://mtrans/stortinget_train_no1/
gsutil cp stortinget_train_no2.tsv gs://mtrans/stortinget_train_no2/
gsutil cp nst_train.tsv gs://mtrans/nst_train/

# Do the tranalation (there will probably be other codes for nrk in the new run
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/nrk_part0/nrk_part0.tsv --output_bucket_folder gs://mtrans/nrk_part0/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/nrk_part1a/nrk_part1a.tsv --output_bucket_folder gs://mtrans/nrk_part1b/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/nrk_part1b/nrk_part1b.tsv --output_bucket_folder gs://mtrans/nrk_part1b/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/nrk_part2a/nrk_part2a.tsv --output_bucket_folder gs://mtrans/nrk_part2a/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/nrk_part2b/nrk_part2b.tsv --output_bucket_folder gs://mtrans/nrk_part2b/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/nrk_part3a/nrk_part3a.tsv --output_bucket_folder gs://mtrans/nrk_part3a/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/nrk_part3b/nrk_part3b.tsv --output_bucket_folder gs://mtrans/nrk_part3b/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/nrk_part4a/nrk_part4a.tsv --output_bucket_folder gs://mtrans/nrk_part4a/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/nrk_part4b/nrk_part4b.tsv --output_bucket_folder gs://mtrans/nrk_part4b/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/nrk_part5a/nrk_part5a.tsv --output_bucket_folder gs://mtrans/nrk_part5a/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/nrk_part5b/nrk_part5b.tsv --output_bucket_folder gs://mtrans/nrk_part5b/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/audio_books_no_train1/audio_books_no_train1.tsv --output_bucket_folder gs://mtrans/audio_books_no_train1/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/audio_books_no_train2/audio_books_no_train2.tsv --output_bucket_folder gs://mtrans/audio_books_no_train2/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/audio_books_no_train3/audio_books_no_train3.tsv --output_bucket_folder gs://mtrans/audio_books_no_train3/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/stortinget_train_no1/stortinget_train_no1.tsv --output_bucket_folder gs://mtrans/stortinget_train_no1/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/stortinget_train_no2/stortinget_train_no2.tsv --output_bucket_folder gs://mtrans/stortinget_train_no2/output/ --timeout 72000
python $program_dir/translate/translate.py --input_bucket_file gs://mtrans/nst_train/nst_train.tsv --output_bucket_folder gs://mtrans/nst_train/output/ --timeout 72000

# Copy the finished translations locally
gsutil cp gs://mtrans/nrk_part0/output/mtrans_nrk_part0_nrk_no_train_nrk_tv_part0_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/nrk_part1b/output/mtrans_nrk_part1a_nrk_no_train_nrk_tv_part1a_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/nrk_part1b/output/mtrans_nrk_part1b_nrk_no_train_nrk_tv_part1b_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/nrk_part2a/output/mtrans_nrk_part2a_nrk_no_train_nrk_tv_part2a_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/nrk_part2b/output/mtrans_nrk_part2b_nrk_no_train_nrk_tv_part2b_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/nrk_part3a/output/mtrans_nrk_part3a_nrk_no_train_nrk_tv_part3a_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/nrk_part3b/output/mtrans_nrk_part3b_nrk_no_train_nrk_tv_part3b_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/nrk_part4a/output/mtrans_nrk_part4a_nrk_no_train_nrk_tv_part4a_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/nrk_part4b/output/mtrans_nrk_part4b_nrk_no_train_nrk_tv_part4b_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/nrk_part5a/output/mtrans_nrk_part5a_nrk_no_train_nrk_tv_part5a_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/nrk_part5b/output/mtrans_nrk_part5b_nrk_no_train_nrk_tv_part5b_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/audio_books_no_train1/output/mtrans_audio_books_no_train1_audio_books_no_train1_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/audio_books_no_train2/output/mtrans_audio_books_no_train2_audio_books_no_train2_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/audio_books_no_train3/output/mtrans_audio_books_no_train3_audio_books_no_train3_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/stortinget_train_no1/output/mtrans_stortinget_train_no1_stortinget_train_no1_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/stortinget_train_no2/output/mtrans_stortinget_train_no2_stortinget_train_no2_en_translations.tsv $translated_dir
gsutil cp gs://mtrans/nst_train/output/mtrans_nst_train_nst_train_en_translations.tsv $translated_dir

# Note that some of these might have a header. This needs to be deleted manually

# Concatenate into one file:
for file in $(ls $translated_dir/*.tsv); do cat $file >> $translated_dir/concatenated_file.tsv; tail -c1 $file | read -r _ || echo >> $translated_dir/concatenated_file.tsv; done

# Do the actual merging. This needs to be done for all the files.
python /home/perk/nostram/translate/merge_translated_text.py --input_json_file_name $base_dir/inference_4/inference_processed/ncc_speech_v7/train/audio_books_nn_train.json --input_tsv_file_name $translated_dir/concatenated_file.tsv --output_file_name $final_dir/train/audio_books_nn_train.json
python /home/perk/nostram/translate/merge_translated_text.py --input_json_file_name $base_dir/inference_4/inference_processed/ncc_speech_v7/train/audio_books_no_train.json --input_tsv_file_name $translated_dir/concatenated_file.tsv --output_file_name $final_dir/train/audio_books_no_train.json
python /home/perk/nostram/translate/merge_translated_text.py --input_json_file_name $base_dir/inference_4/inference_processed/ncc_speech_v7/train/nrk_nn_train.json --input_tsv_file_name $translated_dir/concatenated_file.tsv --output_file_name $final_dir/train/nrk_nn_train.json
python /home/perk/nostram/translate/merge_translated_text.py --input_json_file_name $base_dir/inference_4/inference_processed/ncc_speech_v7/train/nrk_no_train.json --input_tsv_file_name $translated_dir/concatenated_file.tsv --output_file_name $final_dir/train/nrk_no_train.json
python /home/perk/nostram/translate/merge_translated_text.py --input_json_file_name $base_dir/inference_4/inference_processed/ncc_speech_v7/train/nst_train.json --input_tsv_file_name $translated_dir/concatenated_file.tsv --output_file_name $final_dir/train/nst_train.json
python /home/perk/nostram/translate/merge_translated_text.py --input_json_file_name $base_dir/inference_4/inference_processed/ncc_speech_v7/train/stortinget_train.json --input_tsv_file_name $translated_dir/concatenated_file.tsv --output_file_name $final_dir/train/stortinget_train.json



```
# Styletune_6
These steps are only for creating the styletuning-dataset.

```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
program_dir="/home/perk/nostram/styletuning";
process_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5/styletune_6/process_style";
merged_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5/inference_4/inference_result/merged/";
transcribe_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5/styletune_6/transcribe_style";


python $program_dir/process_style.py --input_pattern "$merged_dir/*train*.json" --output_folder "$process_dir" --subcorpus nst
python $program_dir/process_style.py --input_pattern "$merged_dir/*train*.json" --output_folder "$process_dir" --subcorpus ellipses
python $program_dir/process_style.py --input_pattern "$merged_dir/*train*.json" --output_folder "$process_dir" --subcorpus hesitation
python $program_dir/process_style.py --input_pattern "$merged_dir/*train*.json" --output_folder "$process_dir" --subcorpus nst
python $program_dir/process_style.py --input_pattern "$merged_dir/*train*.json" --output_folder "$process_dir" --subcorpus clean_verbatim_no
python $program_dir/process_style.py --input_pattern "$merged_dir/*train*.json" --output_folder "$process_dir" --subcorpus clean_verbatim_nn

# Do simple deduplication into one single file
jq -s 'reduce .[] as $item ({}; .[$item.id + $item.task] //= $item) | map(.)' $process_dir/*.jsonl > $transcribe_dir/transcribe.jsonl



```

