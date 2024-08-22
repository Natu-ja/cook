#!/bin/bash

dataset=../data/cookpad_data.csv

# SFT Config
eval_strategy=epoch
logging_strategy=epoch
save_strategy=epoch

# Generation Config
max_new_tokens=1024

if [ $dataset = "../data/cookpad_data.csv" ]; then

    output_dir=tmp_trainer/cookpad/`date '+%Y_%m_%d_%H_%M_%S'`

    python ./run/cookpad.py \
        --dataset $dataset \
        --output-dir $output_dir \
        --eval-strategy $eval_strategy \
        --logging-strategy $logging_strategy \
        --save-strategy $save_strategy \
        --max-new-tokens $max_new_tokens

elif [ $dataset = "AWeirdDev/zh-tw-recipes-sm" ]; then

    output_dir=tmp_trainer/zh_tw_recipes_sm/`date '+%Y_%m_%d_%H_%M_%S'`

    python ./run/zh_tw_recipes_sm.py \
        --dataset $dataset \
        --output-dir $output_dir \
        --logging-strategy $logging_strategy \
        --save-strategy $save_strategy \

elif [ $dataset = "Erik/data_recipes_instructor" ]; then

    output_dir=tmp_trainer/data_recipes_instructor/`date '+%Y_%m_%d_%H_%M_%S'`

    python ./run/data_recipes_instructor.py \
        --dataset $dataset \
        --output-dir $output_dir \
        --logging-strategy $logging_strategy \
        --save-strategy $save_strategy \

elif [ $dataset = "mertbozkurt/llama2-TR-recipe" ]; then

    output_dir=tmp_trainer/llama2_TR_recipe/`date '+%Y_%m_%d_%H_%M_%S'`

    python ./run/llama2_TR_recipe.py \
        --dataset $dataset \
        --output-dir $output_dir \
        --logging-strategy $logging_strategy \
        --save-strategy $save_strategy \

elif [ $dataset = "pythainlp/thai_food_v1.0" ]; then

    output_dir=tmp_trainer/thai_food/`date '+%Y_%m_%d_%H_%M_%S'`

    python ./run/thai_food.py \
        --dataset $dataset \
        --output-dir $output_dir \
        --logging-strategy $logging_strategy \
        --save-strategy $save_strategy \

elif [ $dataset = "SuryaKrishna02/aya-telugu-food-recipes" ]; then

    output_dir=tmp_trainer/aya_telugu_food_recipes/`date '+%Y_%m_%d_%H_%M_%S'`

    python ./run/aya_telugu_food_recipes.py \
        --dataset $dataset \
        --output-dir $output_dir \
        --logging-strategy $logging_strategy \
        --save-strategy $save_strategy \

else

    echo "Invalid dataset"

fi