#!/bin/bash

dataset=Erik/data_recipes_instructor
tokenizer=openai-community/gpt2
model=openai-community/gpt2

# SFT Config
output_dir=tmp_trainer/`date '+%Y_%m_%d_%H_%M_%S'`
logging_strategy=epoch
save_strategy=epoch

# Run
python main.py \
    --dataset $dataset \
    --tokenizer $tokenizer \
    --model $model \
    --output-dir $output_dir \
    --logging-strategy $logging_strategy \
    --save-strategy $save_strategy