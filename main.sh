#!/bin/bash
dataset=../raw_data/cookpad_data.tsv
tokenizer=cyberagent/open-calm-7b
model=cyberagent/open-calm-7b

# SFT Config
output_dir=tmp_trainer/`date '+%Y_%m_%d_%H_%M_%S'`
eval_strategy=epoch
logging_strategy=epoch
save_strategy=epoch

# Generation Config
max_new_tokens=1024

if [ $dataset = "../raw_data/cookpad_data.tsv" ]; then
    python ./run/cookpad.py \
        --dataset $dataset \
        --tokenizer $tokenizer \
        --model $model \
        --output-dir $output_dir \
        --eval-strategy $eval_strategy \
        --logging-strategy $logging_strategy \
        --save-strategy $save_strategy \
        --max-new-tokens $max_new_tokens

elif [ $dataset = "AWeirdDev/zh-tw-recipes-sm" ]; then
    python ./run/zh_tw_recipes_sm.py \
        --dataset $dataset \
        --tokenizer $tokenizer \
        --model $model \
        --output-dir $output_dir \
        --eval-strategy $eval_strategy \
        --logging-strategy $logging_strategy \
        --save-strategy $save_strategy \
        --max-new-tokens $max_new_tokens

elif [ $dataset = "Erik/data_recipes_instructor" ]; then
    python ./run/data_recipes_instructor.py \
        --dataset $dataset \
        --tokenizer $tokenizer \
        --model $model \
        --output-dir $output_dir \
        --eval-strategy $eval_strategy \
        --logging-strategy $logging_strategy \
        --save-strategy $save_strategy \
        --max-new-tokens $max_new_tokens

elif [ $dataset = "mertbozkurt/llama2-TR-recipe" ]; then
    python ./run/llama2_TR_recipe.py \
        --dataset $dataset \
        --tokenizer $tokenizer \
        --model $model \
        --output-dir $output_dir \
        --eval-strategy $eval_strategy \
        --logging-strategy $logging_strategy \
        --save-strategy $save_strategy \
        --max-new-tokens $max_new_tokens

elif [ $dataset = "pythainlp/thai_food_v1.0" ]; then
    python ./run/thai_food.py \
        --dataset $dataset \
        --tokenizer $tokenizer \
        --model $model \
        --output-dir $output_dir \
        --eval-strategy $eval_strategy \
        --logging-strategy $logging_strategy \
        --save-strategy $save_strategy \
        --max-new-tokens $max_new_tokens

fi