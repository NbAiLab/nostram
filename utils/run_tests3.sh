#!/bin/bash

# Define the array of model paths
#model_paths=("openai/whisper-tiny" "openai/whisper-base" "openai/whisper-small" "openai/whisper-medium" "openai/whisper-large" "openai/whisper-large-v2" "openai/whisper-large-v3" "NbAiLabBeta/nb-whisper-tiny" "NbAiLabBeta/nb-whisper-base" "NbAiLabBeta/nb-whisper-small" "NbAiLabBeta/nb-whisper-medium" "NbAiLabBeta/nb-whisper-large" "NbAiLab/nb-whisper-tiny-RC1" "NbAiLab/nb-whisper-base-RC1" "NbAiLab/nb-whisper-small-RC1" "NbAiLab/nb-whisper-medium-RC1" "NbAiLab/nb-whisper-large-v2-RC3")
#model_paths=("NbAiLabBeta/nb-whisper-tiny-verbatim" "NbAiLabBeta/nb-whisper-base-verbatim" "NbAiLabBeta/nb-whisper-small-verbatim" "NbAiLabBeta/nb-whisper-medium-verbatim" "NbAiLabBeta/nb-whisper-large-verbatim")

model_paths=("NbAiLab/nb-wav2vec2-300m-bokmaal" "NbAiLab/nb-wav2vec2-1b-bokmaal")

# Loop over each model path
for model_path in "${model_paths[@]}"; do
    # Loop over each split
        echo "Running model: $model_path on NbAiLab/NPSC with name: 16K_mp3_bokmaal and split: test"
        python eval_model_wav2vec.py --dataset_path NbAiLab/NPSC --name "16K_mp3_bokmaal" --split "test" --model_path "$model_path" --calculate_wer --save_file /home/perk/results.jsonl
done


model_paths=("NbAiLab/nb-wav2vec2-300m-nynorsk" "NbAiLab/nb-wav2vec2-1b-nynorsk")
# Loop over each model path
for model_path in "${model_paths[@]}"; do
    # Loop over each split
        echo "Running model: $model_path on NbAiLab/NPSC with name: 16K_mp3_nynorsk and split: test"
        python eval_model_wav2vec.py --dataset_path NbAiLab/NPSC --name "16K_mp3_nynorsk" --split "test" --model_path "$model_path" --calculate_wer --save_file /home/perk/results.jsonl
done


model_paths=("NbAiLab/nb-wav2vec2-300m-bokmaal" "NbAiLab/nb-wav2vec2-1b-bokmaal" "NbAiLab/nb-wav2vec2-300m-nynorsk" "NbAiLab/nb-wav2vec2-1b-nynorsk")

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



model_paths=("NbAiLab/nb-wav2vec2-300m-bokmaal" "NbAiLab/nb-wav2vec2-1b-bokmaal")
# Loop over each model path
for model_path in "${model_paths[@]}"; do
    # Loop over each split
        echo "Running model: $model_path on Sprakbanken/nb_samtale with name: verbatim and split: test"
        python eval_model_wav2vec.py --extra_clean --text_field "transcription" --dataset_path Sprakbanken/nb_samtale --name "verbatim" --split "test" --model_path "$model_path" --calculate_wer --save_file /home/perk/results.jsonl
done






