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
| clean_3/fleurs | norwegian_fleurs-test.json |        357 |
| clean_3/fleurs | norwegian_fleurs-validation.json |        163 |
| clean_3/nrk_tv | <empty> | <empty>    |
| clean_3/nrk_tv/clean_3a | <empty> | <empty>    |
| clean_3/nrk_tv/clean_3a/short | nrk.json |        633 |
| clean_3/nrk_tv/clean_3a/short | config.json |         26 |
| clean_3/nrk_tv/clean_3a/short/log | <empty> | <empty>    |
| clean_3/nrk_tv/clean_3a/standard | nrk.json |        418 |
| clean_3/nrk_tv/clean_3a/standard | config.json |         26 |
| clean_3/nrk_tv/clean_3a/standard/log | <empty> | <empty>    |
| clean_3/nrk_tv/clean_3a/very_short | <empty> | <empty>    |
| clean_3/nrk_tv/split_3b | nrk.json |        929 |
| clean_3/nst | nst_train.json |    299,114 |
| clean_3/nst | nst_largetest.json |     63,088 |
| clean_3/nst | nst_validation.json |      1,500 |
| clean_3/nst | nst_test.json |      1,500 |
| clean_3/silence | <empty> | <empty>    |
| clean_3/silence/clean_3b | <empty> | <empty>    |
| clean_3/silence/copy_3a | <empty> | <empty>    |
| clean_3/stortinget | stortinget_train.json |    720,870 |
| clean_3/stortinget | stortinget_test.json |      1,872 |
| clean_3/stortinget | stortinget_validation.json |      2,041 |
| **Total** |      | ** 1,092,537** |

### Directory: inference_4
| Directory | File | Lines     |
| --------- | ---- | ---------:|
| inference_4 | <empty> | <empty>    |
| inference_4/inference_dataset | <empty> | <empty>    |
| inference_4/inference_dataset/audio_books | <empty> | <empty>    |
| inference_4/inference_dataset/audio_books/test | audio_books_test.json |      1,500 |
| inference_4/inference_dataset/audio_books/train | audio_books_train.json |  1,200,076 |
| inference_4/inference_dataset/audio_books/validation | audio_books_validation.json |      1,500 |
| inference_4/inference_dataset/fleurs | <empty> | <empty>    |
| inference_4/inference_dataset/fleurs/test | <empty> | <empty>    |
| inference_4/inference_dataset/fleurs/validation | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv/{train} | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_transcribe | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_transcribe/test | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_transcribe/validation | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_translate | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_translate/test | <empty> | <empty>    |
| inference_4/inference_dataset/nrk_tv_translate/validation | <empty> | <empty>    |
| inference_4/inference_dataset/nst | <empty> | <empty>    |
| inference_4/inference_dataset/nst/test | <empty> | <empty>    |
| inference_4/inference_dataset/nst/train | <empty> | <empty>    |
| inference_4/inference_dataset/nst/validation | <empty> | <empty>    |
| inference_4/inference_dataset/silence | <empty> | <empty>    |
| inference_4/inference_dataset/silence/test | <empty> | <empty>    |
| inference_4/inference_dataset/silence/train | <empty> | <empty>    |
| inference_4/inference_dataset/silence/validation | <empty> | <empty>    |
| inference_4/inference_dataset/stortinget | <empty> | <empty>    |
| inference_4/inference_dataset/stortinget/test | <empty> | <empty>    |
| inference_4/inference_dataset/stortinget/train | <empty> | <empty>    |
| inference_4/inference_dataset/stortinget/validation | <empty> | <empty>    |
| inference_4/inference_result | <empty> | <empty>    |
| inference_4/processed | <empty> | <empty>    |
| **Total** |      | ** 1,203,076** |

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
|   |   |-- nrk_tv_translate
|   |   |   |-- test
|   |   |   |-- validation
|   |   |-- nrk_tv_transcribe
|   |   |   |-- test
|   |   |   |-- validation
|   |   |-- silence
|   |   |   |-- train
|   |   |   |-- test
|   |   |   |-- validation
|   |   |-- stortinget
|   |   |   |-- train
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


# raw_1 and json_2
Not needed in v5. If needed, content needs to be copied from ```ncc_speech_corpus``` and ```ncc_speech_corpus2```.

# clean_3
### Fleurs
Fleurs data is copied unmodified from ```ncc_speech_corpus/json_2```. The clean script would have changed several of the transcripts. However, it is kept unchanged here to be able to follow the development over time.
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5"
clean_text_dir="/mnt/lv_ai_1_ficino/ml/perk/nostram/utils";
archive_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_corpus/json_2";

cp $archive_dir/norwegian_fleurs-validation.json $base_dir/clean_3/fleurs/
cp $archive_dir/norwegian_fleurs-test.json $base_dir/clean_3/fleurs/

### Stortinget and NST
The ```clean_text-script``` is used to copy data from ```ncc_speech_corpus/json_2```. Just some minor renaming and splitting needs to be done.
```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5"
clean_text_dir="/mnt/lv_ai_1_ficino/ml/perk/nostram/utils";
archive_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_corpus/json_2";
python $clean_text_dir/clean_text.py --input_file $archive_dir/stortinget_test.json --output_file $base_dir/clean_3/stortinget/stortinget_test.json
python $clean_text_dir/clean_text.py --input_file $archive_dir/stortinget_eval.json --output_file $base_dir/clean_3/stortinget/stortinget_validation.json
python $clean_text_dir/clean_text.py --input_file $archive_dir/stortinget_train.json --output_file $base_dir/clean_3/stortinget/stortinget_train.json



cd $base_dir
# Stortinget
# cp ../ncc_speech_corpus/json_2/stortinget_*.json clean_3/stortinget/
## Rename Stortinget validation file
mv clean_3/stortinget/stortinget_eval.json clean_3/stortinget/stortinget_validation.json
# Fleurs

# NST
cp ../ncc_speech_corpus/json_2/nst_test.json clean_3/nst/nst_largetest.json
cp ../ncc_speech_corpus/json_2/nst_train.json clean_3/nst/
# Reduce the size of the NST validation and test set
sed -n '1,1500p' clean_3/nst/nst_largetest.json > clean_3/nst/nst_test.json
sed -n '1501,3000p' clean_3/nst/nst_largetest.json > clean_3/nst/nst_validation.json
```

### NRK TV
```bash
# Set working dirs
program_dir="/mnt/lv_ai_1_ficino/ml/perk/nostram/subtitle_processing";
audio_dir="/nfsmounts/datastore/ncc_speech_corpus/source_1/nrk_annotated/audio";
archive_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_corpus/json_2/";

# First run a text_clean since we are using the input for creating timestamps
# Check this
# python /home/perk/nostram/utils/clean_text.py --input_file /mnt/lv_ai_1_ficino/ml/ncc_speech_corpus/json_2/nrk.json --output_file /mnt/lv_ai_1_ficino/ml/ncc_speech_v5/clean_3/nrk_tv/clean_text_3a/nrk.json


# Create the config.json with these settings:
echo -e "{\n\t\"max_duplicates_text_program\": 10,\n\t\"min_alphawords_subtitle\": 0,\n\t\"min_length_subtitle\": 1,\n\t\"min_words_subtitle\": 0,\n\t\"normalise_unicode\": true,\n\t\"drop_subtitles_with_encoding_errors\": true,\n\t\"drop_subtitles_with_curly_brackets\": true,\n\t\"simultaneous_subtitles\": \"delete\",\n\t\"task\": [\"transcribe\", \"translate\"],\n\t\"drop_italics\": true,\n\t\"drop_inaudible\": true,\n\t\"drop_invalid_durations\": true,\n\t\"merge_subtitles\": true,\n\t\"drop_multiple_speakers\": false,\n\t\"combine_continued_sentences\": false,\n\t\"make_bigger_segments\": true,\n\t\"target_duration_seconds\": 28,\n\t\"max_duration_seconds\": 30,\n\t\"pad_with_silence\": true,\n\t\"add_empty_captions\": true,\n\t\"detect_lang_text\": true,\n\t\"allow_lang_text\": [\"nob\", \"nno\"],\n\t\"remove_cpossible\": true,\n\t\"max_separation_seconds\": 5\n}" > $base_dir/clean_3/nrk_tv/standard/config.json
echo -e "{\n\t\"max_duplicates_text_program\": 10,\n\t\"min_alphawords_subtitle\": 0,\n\t\"min_length_subtitle\": 1,\n\t\"min_words_subtitle\": 0,\n\t\"normalise_unicode\": true,\n\t\"drop_subtitles_with_encoding_errors\": true,\n\t\"drop_subtitles_with_curly_brackets\": true,\n\t\"simultaneous_subtitles\": \"delete\",\n\t\"task\": [\"transcribe\", \"translate\"],\n\t\"drop_italics\": true,\n\t\"drop_inaudible\": true,\n\t\"drop_invalid_durations\": true,\n\t\"merge_subtitles\": true,\n\t\"drop_multiple_speakers\": false,\n\t\"combine_continued_sentences\": false,\n\t\"make_bigger_segments\": false,\n\t\"target_duration_seconds\": 28,\n\t\"max_duration_seconds\": 30,\n\t\"pad_with_silence\": true,\n\t\"add_empty_captions\": true,\n\t\"detect_lang_text\": true,\n\t\"allow_lang_text\": [\"nob\", \"nno\"],\n\t\"remove_cpossible\": true,\n\t\"max_separation_seconds\": 5\n}" > $base_dir/clean_3/nrk_tv/short/config.json

# Clean the files - Uncomment for fast test files
# python $program_dir/clean.py --input_file $base_dir/tull/nrk.json --output_folder $base_dir/clean_3/nrk_tv/clean_3a/standard --audio_input_folder $audio_dir  --audio_output_folder $base_dir/clean_3/nrk_tv/mp3/
# python $program_dir/clean.py --input_file $base_dir/tull/nrk.json --output_folder $base_dir/clean_3/nrk_tv/clean_3a/short --audio_input_folder $audio_dir  --audio_output_folder $base_dir/clean_3/nrk_tv/mp3/
python $program_dir/clean.py --input_file $archive_dir/nrk.json --output_folder $base_dir/clean_3/nrk_tv/standard --audio_input_folder $audio_dir  --audio_output_folder $base_dir/clean_3/nrk_tv/mp3/
python $program_dir/clean.py --input_file $archive_dir/nrk.json --output_folder $base_dir/clean_3/nrk_tv/short --audio_input_folder $audio_dir  --audio_output_folder $base_dir/clean_3/nrk_tv/mp3/


# Concatenate files and remove duplicates (can be extended with extra files)
cat $base_dir/clean_3/nrk_tv/standard/nrk.json $base_dir/clean_3/nrk_tv/short/nrk.json | jq -c . | sort -k1,1 -s | awk '!seen[$1]++' > $base_dir/clean_3/nrk_tv/both/nrk.json

# Create the audio files
cat $base_dir/clean_3/nrk_tv/mp3/nrk_process_list.sh | xargs -P 30 -I '{}' sh -c '{}'
```

> JSON should be validated

# inference_4
### Stortinget, Fleurs and NST
No processing is needed here. Just copy the correct files into a single directory. Skip duplications.

```bash
cd $base_dir
cp clean_3/stortinget/*.json inference_4/inference_dataset/
cp clean_3/fleurs/norwegian_fleurs-test.json inference_4/inference_dataset/norwegian_fleurs_test.json
cp clean_3/fleurs/norwegian_fleurs-validation.json inference_4/inference_dataset/norwegian_fleurs_validation.json
cp clean_3/fleurs/norwegian_fleurs-train.json inference_4/inference_dataset/norwegian_fleurs_train.json
cp clean_3/nst/nst_train.json inference_4/inference_dataset/
cp clean_3/nst/nst_test.json inference_4/inference_dataset/
cp clean_3/nst/nst_validation.json inference_4/inference_dataset/
```
> JSON and mp3 should be validated

