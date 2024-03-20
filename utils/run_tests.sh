#!/bin/bash

# Define the array of model paths
#model_paths=("openai/whisper-tiny" "openai/whisper-base" "openai/whisper-small" "openai/whisper-medium" "openai/whisper-large" "openai/whisper-large-v2" "openai/whisper-large-v3" "NbAiLabBeta/nb-whisper-tiny" "NbAiLabBeta/nb-whisper-base" "NbAiLabBeta/nb-whisper-small" "NbAiLabBeta/nb-whisper-medium" "NbAiLabBeta/nb-whisper-large" "NbAiLab/nb-whisper-tiny-RC1" "NbAiLab/nb-whisper-base-RC1" "NbAiLab/nb-whisper-small-RC1" "NbAiLab/nb-whisper-medium-RC1" "NbAiLab/nb-whisper-large-v2-RC3")
model_paths=("NbAiLabBeta/nb-whisper-tiny-verbatim" "NbAiLabBeta/nb-whisper-base-verbatim" "NbAiLabBeta/nb-whisper-small-verbatim" "NbAiLabBeta/nb-whisper-medium-verbatim" "NbAiLabBeta/nb-whisper-large-verbatim")


# Define the array of splits
splits=("test")

for model_path in "${model_paths[@]}"; do
    # Loop over each split
    for split in "${splits[@]}"; do
        echo "Running model: $model_path on mozilla-foundation/common_voice_16_1 for nn-NO with split: $split"
        python test_dataset.py --from_flax --dataset_path mozilla-foundation/common_voice_16_1 --name "nn-NO" --split "$split" --text_field "sentence" --language "nn" --model_path "$model_path" --calculate_wer --save_file /home/perk/results.jsonl
    done
done


# Loop over each model path
#for model_path in "${model_paths[@]}"; do
#    # Loop over each split
#        echo "Running model: $model_path on NbAiLab/NPSC with name: 16K_mp3_bokmaal and split: test"
#        python test_dataset.py --from_flax --dataset_path NbAiLab/NPSC --name "16K_mp3_bokmaal" --split "test" --model_path "$model_path" --calculate_wer --save_file /home/perk/results.jsonl
#done

# Loop over each model path
#for model_path in "${model_paths[@]}"; do
#    # Loop over each split
#        echo "Running model: $model_path on NbAiLab/NPSC with name: 16K_mp3_nynorsk and split: test"
#        python test_dataset.py --from_flax --dataset_path NbAiLab/NPSC --name "16K_mp3_nynorsk" --split "test" --language ="nn" --model_path "$model_path" --calculate_wer --save_file /home/perk/results.jsonl
#done







