#!/bin/bash

# Define the array of model paths
model_paths=("openai/whisper-tiny" "openai/whisper-base" "openai/whisper-small" "openai/whisper-medium" "openai/whisper-large" "openai/whisper-large-v2" "openai/whisper-large-v3" "NbAiLabBeta/nb-whisper-tiny" "NbAiLabBeta/nb-whisper-base" "NbAiLabBeta/nb-whisper-small" "NbAiLabBeta/nb-whisper-medium" "NbAiLabBeta/nb-whisper-large")

#model_paths=("openai/whisper-large-v2")

# Define the array of splits
#splits=("test_norwegian_fleurs" "test_nst")

#for model_path in "${model_paths[@]}"; do
#    # Loop over each split
#    for split in "${splits[@]}"; do
#        echo "Running model: $model_path on NbAiLab/ncc_speech_v7 with split: $split"
#        python test_dataset.py --dataset_path NbAiLab/ncc_speech_v7 --split "$split" --model_path "$model_path" --calculate_wer --save_file /home/perk/results.jsonl
#    done
#done


# Loop over each model path
for model_path in "${model_paths[@]}"; do
    # Loop over each split
        echo "Running model: $model_path on NbAiLab/NPSC with name: 16K_mp3_bokmaal and split: test"
        python test_dataset.py --dataset_path NbAiLab/NPSC --name "16K_mp3_bokmaal" --split "test" --model_path "$model_path" --calculate_wer --save_file /home/perk/results.jsonl
    done
done

# Loop over each model path
for model_path in "${model_paths[@]}"; do
    # Loop over each split
        echo "Running model: $model_path on NbAiLab/NPSC with name: 16K_mp3_nynorsk and split: test"
        python test_dataset.py --dataset_path NbAiLab/NPSC --name "16K_mp3_nynorsk" --split "test" --language ="nn" --model_path "$model_path" --calculate_wer --save_file /home/perk/results.jsonl
    done
done






