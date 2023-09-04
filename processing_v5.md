
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

# json_2
This is not used in v5. Content is copied from ```ncc_speech_corput``` and ```ncc_speech_corpus2```.

# clean_json_3
## Stortinget
Stortinget data is copied from ```ncc_speech_corpus/json_2```
```bash
cd /mnt/lv_ai_1_ficino/ml/ncc_speech_v5/clean_json_3
mkdir stortinget
cd stortinget
cp ../../ncc_speech_corpus/json_2/stortinget*.json stortinget/
```

# transcribe_json_4
## Stortinget
We have dropped the wer/ner/clean-steps for this corpus. Now simply copying from clean_json_3 to transcribe_json_4
```bash
cd transcribed_json_4/
mkdir stortinget
mkdir stortinget/test
mkdir stortinget/train
mkdir stortinget/validation
cd ..
cp clean_json_3/stortinget/stortinget_train.json transcribed_json_4/stortinget/train/
cp clean_json_3/stortinget/stortinget_test.json transcribed_json_4/stortinget/test/
cp clean_json_3/stortinget/stortinget_validation.json transcribed_json_4/stortinget/validation/
cp clean_json_3/stortinget/stortinget_eval.json transcribed_json_4/stortinget/validation/
```


