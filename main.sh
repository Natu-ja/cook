tokenizer=cyberagent/open-calm-7b
model=cyberagent/open-calm-7b

# SFT Config
eval_strategy=steps

# Generation Config
max_new_tokens=1024

python main.py \
    --tokenizer $tokenizer \
    --model $model \
    --eval-strategy $eval_strategy \
    --max-new-tokens $max_new_tokens