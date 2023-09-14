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
| clean_3/silence | <empty> | <empty>    |
| clean_3/silence/clean_3b | <empty> | <empty>    |
| clean_3/silence/copy_3a | <empty> | <empty>    |
| clean_3/stortinget | stortinget_train.json |    580,827 |
| clean_3/stortinget | stortinget_test.json |      1,520 |
| clean_3/stortinget | stortinget_validation.json |      1,714 |
| **Total** |      | **13,721,930** |

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
| inference_4/inference_dataset/nrk_tv_transcribe | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_transcribe/test | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_transcribe/validation | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_translate | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_translate/test | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_translate/validation | <empty> | <empty>    |
| inference_4/inference_dataset/nst | <empty> | <empty>    |
| inference_4/inference_dataset/nst/test | nst_test.json |      1,500 |
| inference_4/inference_dataset/nst/train | nst_train.json |    144,546 |
| inference_4/inference_dataset/nst/validation | nst_validation.json |      1,500 |
| inference_4/inference_dataset/silence | <empty> | <empty>    |
| inference_4/inference_dataset/silence/test | silence_test.json |      1,000 |
| inference_4/inference_dataset/silence/train | silence_train.json |    107,019 |
| inference_4/inference_dataset/silence/validation | silence_validation.json |      1,000 |
| inference_4/inference_dataset/stortinget | <empty> | <empty>    |
| inference_4/inference_dataset/stortinget/test | stortinget_test.json |      1,520 |
| inference_4/inference_dataset/stortinget/train | stortinget_train.json |    580,827 |
| inference_4/inference_dataset/stortinget/validation | stortinget_validation.json |      1,714 |
| inference_4/inference_result | <empty> | <empty>    |
| inference_4/processed | <empty> | <empty>    |
| **Total** |      | ** 1,989,676** |

### Directory: translation_5
| Directory | File | Lines     |
| --------- | ---- | ---------:|
| translation_5 | <empty> | <empty>    |
| translation_5/processed | <empty> | <empty>    |
| translation_5/translation_files | <empty> | <empty>    |
| **Total** |      | **         0** |



# Copy Structure
The following command creates all the necssary folders if they do not exist.

```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5"
mkdir -p "$base_dir"/{clean_3/{nrk_tv/{standard,short,both,mp3},silence,stortinget,fleurs,nst,audio_books},inference_4/{inference_dataset/{mp3/{nrk_tv,silence,stortinget,fleurs,nst,audio_books},nrk_tv/{train},nrk_tv_translate/{test,validation},nrk_tv_transcribe/{test,validation},silence/{train,test,validation},stortinget/{train,test,validation},fleurs/{test,validation},nst/{train,test,validation},audio_books/{train,test,validation}},inference_result,processed},translation_5/{translation_files,processed}}


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
|   |-- silence
|   |-- stortinget
|   |-- fleurs
|   |-- nst
|   |-- stortinget
|   |-- audio_books
|-- inference_4/
|   |-- inference_dataset
|   |   |-- mp3
|   |   |   |-- nrk_tv
|   |   |   |-- silence
|   |   |   |-- stortinget
|   |   |   |-- fleurs
|   |   |   |-- nst
|   |   |   |-- audio_books
|   |   |-- nrk_tv
|   |   |   |-- train
|   |   |-- nrk_tv_no
|   |   |   |-- test
|   |   |   |-- validation
|   |   |-- nrk_tv_nn
|   |   |   |-- test
|   |   |   |-- validation
|   |   |-- silence
|   |   |   |-- train
|   |   |   |-- test
|   |   |   |-- validation
|   |   |-- stortinget
|   |   |   |-- train
|   |   |-- stortinget_no
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
|   |   |   |-- test
|   |   |   |-- validation
|   |-- inference_result
|   |-- processed
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

# Create the config.json with these settings:
echo -e "{\n\t\"max_duplicates_text_program\": 10,\n\t\"min_alphawords_subtitle\": 0,\n\t\"min_length_subtitle\": 1,\n\t\"min_words_subtitle\": 0,\n\t\"normalise_unicode\": true,\n\t\"drop_subtitles_with_encoding_errors\": true,\n\t\"drop_subtitles_with_curly_brackets\": true,\n\t\"simultaneous_subtitles\": \"delete\",\n\t\"task\": [\"transcribe\", \"translate\"],\n\t\"drop_italics\": true,\n\t\"drop_inaudible\": true,\n\t\"drop_invalid_durations\": true,\n\t\"merge_subtitles\": true,\n\t\"drop_multiple_speakers\": false,\n\t\"combine_continued_sentences\": false,\n\t\"make_bigger_segments\": true,\n\t\"target_duration_seconds\": 28,\n\t\"max_duration_seconds\": 29,\n\t\"pad_with_silence\": true,\n\t\"add_empty_captions\": true,\n\t\"detect_lang_text\": true,\n\t\"allow_lang_text\": [\"nob\", \"nno\"],\n\t\"remove_cpossible\": true,\n\t\"max_separation_seconds\": 5\n}" > $base_dir/clean_3/nrk_tv/standard/config.json;
echo -e "{\n\t\"max_duplicates_text_program\": 10,\n\t\"min_alphawords_subtitle\": 0,\n\t\"min_length_subtitle\": 1,\n\t\"min_words_subtitle\": 0,\n\t\"normalise_unicode\": true,\n\t\"drop_subtitles_with_encoding_errors\": true,\n\t\"drop_subtitles_with_curly_brackets\": true,\n\t\"simultaneous_subtitles\": \"delete\",\n\t\"task\": [\"transcribe\", \"translate\"],\n\t\"drop_italics\": true,\n\t\"drop_inaudible\": true,\n\t\"drop_invalid_durations\": true,\n\t\"merge_subtitles\": true,\n\t\"drop_multiple_speakers\": false,\n\t\"combine_continued_sentences\": false,\n\t\"make_bigger_segments\": false,\n\t\"target_duration_seconds\": 28,\n\t\"max_duration_seconds\": 29,\n\t\"pad_with_silence\": true,\n\t\"add_empty_captions\": true,\n\t\"detect_lang_text\": true,\n\t\"allow_lang_text\": [\"nob\", \"nno\"],\n\t\"remove_cpossible\": true,\n\t\"max_separation_seconds\": 5\n}" > $base_dir/clean_3/nrk_tv/short/config.json;

# Clean the files - This takes roughly 4 hours
python $program_dir/clean.py --input_file $archive_dir/nrk.json --output_folder $base_dir/clean_3/nrk_tv/standard --audio_input_folder $audio_dir  --audio_output_folder $base_dir/clean_3/nrk_tv/mp3/;
python $program_dir/clean.py --input_file $archive_dir/nrk.json --output_folder $base_dir/clean_3/nrk_tv/short --audio_input_folder $audio_dir  --audio_output_folder $base_dir/clean_3/nrk_tv/mp3/;

# Concatenate files and remove duplicates (can be extended with extra files)
cat $base_dir/clean_3/nrk_tv/standard/nrk.json $base_dir/clean_3/nrk_tv/short/nrk.json | jq -c . | sort -k1,1 -s | awk '!seen[$1]++' > $base_dir/clean_3/nrk_tv/both/nrk.json;

# Create the audio files
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
### Silence
Copy files, and make the split at the same time
```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
shuf "$base_dir/clean_3/silence/silence.json" | awk -v base="$base_dir" 'NR <= 1000 {print > base "/inference_4/inference_dataset/silence/test/silence_test.json"} NR > 1000 && NR <= 2000 {print > base "/inference_4/inference_dataset/silence/validation/silence_validation.json"} NR > 2000 {print > base "/inference_4/inference_dataset/silence/train/silence_train.json"}'
```
### Stortinget
Here we need to make a test and validation dataset that contains only Norwegian Bokmål
```bash
#Stortinget
cp $base_dir/clean_3/stortinget/stortinget_train.json $base_dir/inference_4/inference_dataset/stortinget/train/;
jq -c 'select(.text_language=="no")' $base_dir/clean_3/stortinget/stortinget_test.json > $base_dir/inference_4/inference_dataset/stortinget_no/test/stortinget_no_test.json;
jq -c 'select(.text_language=="no")' $base_dir/clean_3/stortinget/stortinget_validation.json > $base_dir/inference_4/inference_dataset/stortinget_no/validation/stortinget_validation_test.json;
```

### Fleurs and NST
No processing is needed here. Just copy the correct files into a single directory. 

```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";

#Fleurs
cp $base_dir/clean_3/fleurs/norwegian_fleurs-test.json $base_dir/inference_4/inference_dataset/fleurs/test/;
cp $base_dir/clean_3/fleurs/norwegian_fleurs-validation.json $base_dir/inference_4/inference_dataset/fleurs/validation/;
#NST
cp $base_dir/clean_3/nst/nst_train.json $base_dir/inference_4/inference_dataset/nst/train/
cp $base_dir/clean_3/nst/nst_test.json $base_dir/inference_4/inference_dataset/nst/test/
cp $base_dir/clean_3/nst/nst_validation.json $base_dir/inference_4/inference_dataset/nst/validation/
```
> JSON and mp3 should be validated

