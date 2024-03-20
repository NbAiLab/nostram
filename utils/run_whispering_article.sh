#!/bin/bash

model_paths=("NbAiLab/wav2vec2-1b-npsc-nst-bokmaal-repaired" "NbAiLab/nb-wav2vec2-1b-bokmaal-v2")
#"NbAiLab/nb-wav2vec2-300m-bokmaal" "NbAiLab/nb-wav2vec2-1b-bokmaal" "NbAiLab/nb-wav2vec2-300m-nynorsk" "NbAiLab/nb-wav2vec2-1b-nynorsk")

# Loop over each model path
for model_path in "${model_paths[@]}"; do
   # Loop over each split
        echo "Running model: $model_path on NbAiLab/ncc_speech_v7  and split: test_nst"
        python eval_model_wav2vec.py --dataset_path NbAiLab/ncc_speech_v7 --split "test_nst" --model_path "$model_path" --calculate_wer --save_file /home/perk/results.jsonl
done



# Loop over each model path
for model_path in "${model_paths[@]}"; do
    # Loop over each split
        echo "Running model: $model_path on NbAiLab/ncc_speech_v7 and split: norwegian_fleurs"
        python eval_model_wav2vec.py --dataset_path NbAiLab/ncc_speech_v7 --split "test_norwegian_fleurs" --model_path "$model_path" --calculate_wer --save_file /home/perk/results.jsonl
done



model_paths=("NbAiLab/nb-wav2vec2-300m-nynorsk" "NbAiLab/nb-wav2vec2-1b-nynorsk")
# Loop over each model path
for model_path in "${model_paths[@]}"; do
    # Loop over each split
        echo "Running model: $model_path on Sprakbanken/nb_samtale with name: verbatim and split: test"
        python eval_model_wav2vec.py --dataset_path mozilla-foundation/common_voice_16_1 --name "nn-NO" --text_field "sentence" --split "test" --model_path "$model_path" --calculate_wer --save_file /home/perk/results.jsonl
done

# Define the array of model paths
model_paths=("openai/whisper-tiny" "openai/whisper-base" "openai/whisper-small" "openai/whisper-medium" "openai/whisper-large" "openai/whisper-large-v2" "openai/whisper-large-v3" "NbAiLabBeta/nb-whisper-tiny" "NbAiLabBeta/nb-whisper-base" "NbAiLabBeta/nb-whisper-small" "NbAiLabBeta/nb-whisper-medium" "NbAiLabBeta/nb-whisper-large")


## Loop over each model path
for model_path in "${model_paths[@]}"; do
    # Loop over each split
        echo "Running model: $model_path on NbAiLab/ncc_speech_v7  and split: test_nst"
        python test_dataset.py --from_flax --dataset_path NbAiLab/ncc_speech_v7 --split "test_nst" --model_path "$model_path" --calculate_wer --save_file /home/perk/results.jsonl
done


## Loop over each model path
for model_path in "${model_paths[@]}"; do
    # Loop over each split
        echo "Running model: $model_path on NbAiLab/ncc_speech_v7  and split: test_norwegian_fleurs"
        python test_dataset.py --from_flax --dataset_path NbAiLab/ncc_speech_v7 --split "test_norwegian_fleurs" --model_path "$model_path" --calculate_wer --save_file /home/perk/results.jsonl
done




