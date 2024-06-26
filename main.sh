#!/bin/bash

tokenizer=cyberagent/open-calm-7b
model=cyberagent/open-calm-7b

# SFT Config
output_dir=tmp_trainer/`date '+%Y_%m_%d_%H_%M_%S'`
eval_strategy=epoch
logging_strategy=epoch
save_strategy=epoch

# Generation Config
max_new_tokens=1024

python main.py \
    --tokenizer $tokenizer \
    --model $model \
    --output-dir $output_dir \
    --eval-strategy $eval_strategy \
    --logging-strategy $logging_strategy \
    --save-strategy $save_strategy \
    --max-new-tokens $max_new_tokens