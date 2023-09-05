# ncc_speech_5
Main dataset as of September 2023. Please see the complete description of [Corpus Structure](corpus_structure.md)

# Current Status
The status can be updated by running ```python nostram/utils/json_stats.py /mnt/lv_ai_1_ficino/ml/ncc_speech_v5```.

## Target Directory: /mnt/lv_ai_1_ficino/ml/ncc_speech_v5
### Directory: clean_3
| Directory | File | Lines     |
| --------- | ---- | ---------:|
| clean_3 | <empty> | <empty>    |
| clean_3/fleurs | norwegian_fleurs-test.json |        357 |
| clean_3/fleurs | norwegian_fleurs-validation.json |        163 |
| clean_3/nrk_tv_silence | <empty> | <empty>    |
| clean_3/nrk_tv_transcribe | <empty> | <empty>    |
| clean_3/nrk_tv_translate | <empty> | <empty>    |
| clean_3/nrk_tv_veryshort | <empty> | <empty>    |
| clean_3/nst | nst_train.json |    299,114 |
| clean_3/nst | nst_largetest.json |     63,088 |
| clean_3/nst | nst_validation.json |      1,500 |
| clean_3/nst | nst_test.json |      1,500 |
| clean_3/stortinget | stortinget_train.json |    720,870 |
| clean_3/stortinget | stortinget_test.json |      1,872 |
| clean_3/stortinget | stortinget_validation.json |      2,041 |
| **Total** |      | **1,090,505** |

### Directory: inference_4
| Directory | File | Lines     |
| --------- | ---- | ---------:|
| inference_4 | <empty> | <empty>    |
| inference_4/inference_dataset | nst_train.json |    299,114 |
| inference_4/inference_dataset | stortinget_train.json |    720,870 |
| inference_4/inference_dataset | stortinget_test.json |      1,872 |
| inference_4/inference_dataset | nst_validation.json |      1,500 |
| inference_4/inference_dataset | stortinget_validation.json |      2,041 |
| inference_4/inference_dataset | norwegian_fleurs-test.json |        357 |
| inference_4/inference_dataset | nst_test.json |      1,500 |
| inference_4/inference_dataset | norwegian_fleurs-validation.json |        163 |
| inference_4/inference_result | <empty> | <empty>    |
| inference_4/processed | <empty> | <empty>    |
| **Total** |      | **1,027,417** |

### Directory: translation_5
| Directory | File | Lines     |
| --------- | ---- | ---------:|
| translation_5 | <empty> | <empty>    |
| translation_5/processed | <empty> | <empty>    |
| translation_5/translation_files | <empty> | <empty>    |
| **Total** |      | **0** |

# Copy Structure
The following command creates all the necssary folders if they do not exist.

```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
mkdir -p "$base_dir"/{clean_3/{nrk_tv_transcribe/{copy_3a,clean_3b},nrk_tv_translate/{copy_3a,clean_3b},nrk_tv_veryshort/{copy_3a,clean_3b},nrk_tv_silence/{copy_3a,clean_3b},stortinget,fleurs,nst,stortinget},inference_4/{inference_dataset,inference_result,processed},translation_5/{translation_files,processed}}
Husk at vi ikke trenger å splitte nrk i transcribe og translate. Kutt kataloger. Kutt silence

Make dir nrk/audio
```
This should create the following structure:
```plaintext
$base_dir/
|-- clean_3/
|   |-- nrk_tv_transcribe/
|   |   |-- copy_3a
|   |   |-- clean_3b
|   |-- nrk_tv_translate/
|   |   |-- copy_3a
|   |   |-- clean_3b
|   |-- nrk_tv_veryshort/
|   |   |-- copy_3a
|   |   |-- clean_3b
|   |-- nrk_tv_silence/
|   |   |-- copy_3a
|   |   |-- clean_3b
|   |-- stortinget
|   |-- fleurs
|   |-- nst
|   |-- stortinget
|-- inference_4/
|   |-- inference_dataset
|   |-- inference_result
|   |-- processed
|-- translation_5/
    |-- translation_files
    |-- processed

```


# raw_1 and json_2
Not needed in v5. If needed, content needs to be copied from ```ncc_speech_corpus``` and ```ncc_speech_corpus2```.

# clean_3

### Stortinget, Fleurs and NST
All data are directly copied from ```ncc_speech_corpus/json_2```. Just some minor renaming and splitting needs to be done.
```bash
cd $base_dir
# Stortinget
cp ../ncc_speech_corpus/json_2/stortinget_*.json clean_3/stortinget/
## Rename Stortinget validation file
mv clean_3/stortinget/stortinget_eval.json clean_3/stortinget/stortinget_validation.json
# Fleurs
cp ../ncc_speech_corpus/json_2/norwegian_fleurs-validation.json clean_3/fleurs/
cp ../ncc_speech_corpus/json_2/norwegian_fleurs-test.json clean_3/fleurs/
# NST
cp ../ncc_speech_corpus/json_2/nst_test.json clean_3/nst/nst_largetest.json
cp ../ncc_speech_corpus/json_2/nst_train.json clean_3/nst/
# Reduce the size of the NST validation and test set
sed -n '1,1500p' clean_3/nst/nst_largetest.json > clean_3/nst/nst_test.json
sed -n '1501,3000p' clean_3/nst/nst_largetest.json > clean_3/nst/nst_validation.json
```

### NRK TV
```bash
# jq -c 'select(.vtt_folder=="vtt_transcribe_translate")' ../ncc_speech_corpus/json_2/nrk.json > clean_3/nrk_tv_transcribe/copy_3a/nrk_tv_transcribe_all.json
# jq -c 'select(.vtt_folder=="vtt_translate")' ../ncc_speech_corpus/json_2/nrk.json > clean_3/nrk_tv_translate/copy_3a/nrk_tv_translate_all.json
# Create the config.json with these settings:
echo -e "{\n\t\"min_alphawords_subtitle\": 0,\n\t\"min_length_subtitle\": 1,\n\t\"min_words_subtitle\": 0,\n\t\"normalise_unicode\": true,\n\t\"drop_subtitles_with_encoding_errors\": true,\n\t\"drop_subtitles_with_curly_brackets\": true,\n\t\"simultaneous_subtitles\": \"delete\",\n\t\"task\": [\"transcribe\", \"translate\"],\n\t\"drop_italics\": true,\n\t\"drop_inaudible\": true,\n\t\"drop_invalid_durations\": true,\n\t\"merge_subtitles\": true,\n\t\"drop_multiple_speakers\": false,\n\t\"combine_continued_sentences\": false,\n\t\"make_bigger_segments\": true,\n\t\"target_duration_seconds\": 28,\n\t\"max_duration_seconds\": 30,\n\t\"pad_with_silence\": true,\n\t\"add_empty_captions\": true,\n\t\"detect_lang_text\": true,\n\t\"allow_lang_text\": [\"nob\", \"nno\"],\n\t\"remove_cpossible\": true,\n\t\"max_separation_seconds\": 5\n}" > $base_dir/clean_3/nrk_tv/config.json
echo -e "{\n\t\"min_alphawords_subtitle\": 0,\n\t\"min_length_subtitle\": 1,\n\t\"min_words_subtitle\": 0,\n\t\"normalise_unicode\": true,\n\t\"drop_subtitles_with_encoding_errors\": true,\n\t\"drop_subtitles_with_curly_brackets\": true,\n\t\"simultaneous_subtitles\": \"delete\",\n\t\"task\": [\"transcribe\", \"translate\"],\n\t\"drop_italics\": true,\n\t\"drop_inaudible\": true,\n\t\"drop_invalid_durations\": true,\n\t\"merge_subtitles\": true,\n\t\"drop_multiple_speakers\": false,\n\t\"combine_continued_sentences\": false,\n\t\"make_bigger_segments\": false,\n\t\"target_duration_seconds\": 28,\n\t\"max_duration_seconds\": 30,\n\t\"pad_with_silence\": true,\n\t\"add_empty_captions\": true,\n\t\"detect_lang_text\": true,\n\t\"allow_lang_text\": [\"nob\", \"nno\"],\n\t\"remove_cpossible\": true,\n\t\"max_separation_seconds\": 5\n}" > $base_dir/clean_3/nrk_tv_veryshort/config.json


program_dir="/mnt/lv_ai_1_ficino/ml/perk/nostram/subtitle_processing";
audio_dir="/nfsmounts/datastore/ncc_speech_corpus/source_1/nrk_annotated/audio";
python $program_dir/clean.py --input_file $base_dir/tull/nrk.json --output_folder $base_dir/clean_3/nrk_tv --audio_input_folder $audio_dir  --audio_output_folder $base_dir/clean_3/nrk_tv/audio/
python $program_dir/clean.py --input_file $base_dir/tull/nrk.json --output_folder $base_dir/clean_3/nrk_tv_veryshort --audio_input_folder $audio_dir  --audio_output_folder $base_dir/clean_3/nrk_tv_veryshort/audio/


# Create the audio files
cat $base_dir/clean_3/nrk_tv/audio/nrk_process_list.sh | xargs -P 30 -I '{}' sh -c '{}'
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

