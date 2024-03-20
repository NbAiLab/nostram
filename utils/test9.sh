#!/bin/bash


# Define the array of model paths
model_paths=("openai/whisper-tiny" "openai/whisper-base" "openai/whisper-small" "openai/whisper-medium" "openai/whisper-large" "openai/whisper-large-v2" "openai/whisper-large-v3" "NbAiLabBeta/nb-whisper-tiny" "NbAiLabBeta/nb-whisper-base" "NbAiLabBeta/nb-whisper-small" "NbAiLabBeta/nb-whisper-medium" "NbAiLabBeta/nb-whisper-large")



## Loop over each model path
for model_path in "${model_paths[@]}"; do
    # Loop over each split
        echo "Running model: $model_path on NbAiLab/ncc_speech_v7  and split: test_norwegian_fleurs"
        python test_dataset.py --from_flax --dataset_path NbAiLab/ncc_speech_v7 --split "test_norwegian_fleurs" --model_path "$model_path" --calculate_wer --save_file /home/perk/results.jsonl
done




