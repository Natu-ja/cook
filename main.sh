#!/bin/bash

dataset=../data/cookpad_data.csv

# SFT Config
strategy=epoch

# Generation Config
max_new_tokens=1024

if [ $dataset = "../data/cookpad_data.csv" ]; then

    output_dir=tmp_trainer/cookpad/`date '+%Y_%m_%d_%H_%M_%S'`

    python ./run/cookpad.py \
        --dataset $dataset \
        --output-dir $output_dir \
        --eval-strategy $strategy \
        --logging-strategy $strategy \
        --save-strategy $strategy \
        --max-new-tokens $max_new_tokens

elif [ $dataset = "Erik/data_recipes_instructor" ]; then

    output_dir=tmp_trainer/data_recipes_instructor/`date '+%Y_%m_%d_%H_%M_%S'`

    python ./run/data_recipes_instructor.py \
        --dataset $dataset \
        --output-dir $output_dir \
        --logging-strategy $strategy \
        --save-strategy $strategy \

elif [ $dataset = "mertbozkurt/llama2-TR-recipe" ]; then

    output_dir=tmp_trainer/llama2_TR_recipe/`date '+%Y_%m_%d_%H_%M_%S'`

    python ./run/llama2_TR_recipe.py \
        --dataset $dataset \
        --output-dir $output_dir \
        --logging-strategy $strategy \
        --save-strategy $strategy \

elif [ $dataset = "AWeirdDev/all-recipes-sm" ]; then

    output_dir=tmp_trainer/all_recipes_sm/`date '+%Y_%m_%d_%H_%M_%S'`

    python ./run/all_recipes.py \
        --dataset $dataset \
        --output-dir $output_dir \
        --logging-strategy $strategy \
        --save-strategy $strategy \

elif [ $dataset = "AWeirdDev/zh-tw-recipes-sm" ]; then

    output_dir=tmp_trainer/zh_tw_recipes_sm/`date '+%Y_%m_%d_%H_%M_%S'`

    python ./run/zh_tw_recipes_sm.py \
        --dataset $dataset \
        --output-dir $output_dir \
        --logging-strategy $strategy \
        --save-strategy $strategy \

elif [ $dataset = "AWeirdDev/all-recipes-xs" ]; then

    output_dir=tmp_trainer/all_recipes_xs/`date '+%Y_%m_%d_%H_%M_%S'`

    python ./run/all_recipes.py \
        --dataset $dataset \
        --output-dir $output_dir \
        --logging-strategy $strategy \
        --save-strategy $strategy \

elif [ $dataset = "SuryaKrishna02/aya-telugu-food-recipes" ]; then

    output_dir=tmp_trainer/aya_telugu_food_recipes/`date '+%Y_%m_%d_%H_%M_%S'`

    python ./run/aya_telugu_food_recipes.py \
        --dataset $dataset \
        --output-dir $output_dir \
        --logging-strategy $strategy \
        --save-strategy $strategy \

elif [ $dataset = "pythainlp/thai_food_v1.0" ]; then

    output_dir=tmp_trainer/thai_food/`date '+%Y_%m_%d_%H_%M_%S'`

    python ./run/thai_food.py \
        --dataset $dataset \
        --output-dir $output_dir \
        --logging-strategy $strategy \
        --save-strategy $strategy \

else

    echo "Invalid dataset"

fi