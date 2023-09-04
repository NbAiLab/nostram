# ncc_speech_5
Main dataset as of September 2023. Plese see the complete description of [Corpus Structure](corpus_structure.md)

# Current Status
## Target Directory: /mnt/lv_ai_1_ficino/ml/ncc_speech_v5
| Directory | File | Lines     |
| --------- | ---- | ---------:|
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;clean_json_3/stortinget | stortinget_train.json |    720,870 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;clean_json_3/stortinget | stortinget_eval.json |      2,041 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;clean_json_3/stortinget | stortinget_test.json |      1,872 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;transcribed_json_4/stortinget/validation | stortinget_eval.json |      2,041 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;transcribed_json_4/stortinget/train | stortinget_train.json |    720,870 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;transcribed_json_4/stortinget/test | stortinget_test.json |      1,872 |

# Copy Structure
The following command creates all the necssary folders if they do not exist.

```bash
base_dir="/mnt/lv_ai_1_ficino/ml/ncc_speech_v5";
mkdir -p "$base_dir"/{clean_3/{nrk_tv_transcribe,nrk_tv_translate,nrk_tv_veryshort,nrk_tv_silence,stortinget,fleurs,nst,stortinget},inference_4/{inference_dataset,inference_result,processed},translation_5/{translation_files,processed}}
```

# json_2
This is not used in v5. Content is copied from ```ncc_speech_corput``` and ```ncc_speech_corpus2```.

# clean_3
### Stortinget
Stortinget data is copied from ```ncc_speech_corpus/json_2```
```bash
cd $base_dir
cp ../ncc_speech_corpus/json_2/stortinget_*.json clean_3/stortinget/
mv clean_3/stortinget/stortinget_eval.json clean_3/stortinget/stortinget_validation.json
```

# inference_4
### Stortinget
Currently there is no processing of Stortinget, since this is moved after the inference. Just moving to the inference dataset.
```bash
cd $base_dir
cp clean_3/stortinget/*.json inference_4/inference_dataset/
```


