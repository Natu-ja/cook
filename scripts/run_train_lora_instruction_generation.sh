tokenizer=TOKENIZER-NAME
model=MODEL-NAME
data=./data
input_max_len=128

# TrainingArguments
train_batch_size=8
eval_batch_size=8
gradients=1
lr=5e-5
weight_decay=0.0
max_grad_norm=1.0
epochs=3.0
scheduler=linear
warmup=0.0
seed=42
metric_for_best_model=eval_loss

# LoraConfig
rank=8
target_modules=(TARGET-MODULES1 TARGET-MODULES2 ...)
lora_alpha=8
lora_dropout=0.0
lora_bias=none

# Instruction tuning
system_message=以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。
instruction=以下の食材を使って作れる料理名を教えてください。

# generate
generation=yes
max_length=20
min-length=0
do_sample=False
num_beams=1
num_beam_groups=1
penalty_alpha=0.0
temperature=1.0
top-k=50
top_p=1.0
repetition_penalty=1.0

# Run
python run_train.py $tokenizer $model \
    --data $data \
    --input-max-len $input_max_len \
    --train-batch-size $train_batch_size \
    --eval-batch-size $eval_batch_size \
    --gradients $gradients \
    --lr $lr \
    --weight-decay $weight_decay \
    --max-grad-norm $max_grad_norm \
    --epochs $epochs \
    --scheduler $scheduler \
    --warmup $warmup \
    --seed $seed \
    --metric-for-best-model $metric_for_best_model \
    --rank $rank \
    --target-modules ${target_modules[@]} \
    --lora-alpha $lora_alpha \
    --lora-dropout $lora_dropout \
    --lora-bias $lora_bias \
    --system-message $system_message \
    --instruction $instruction\
    --generation $generation \
    --max-length $max_length \
    --min-length $min_length \
    --do-sample $do_sample \
    --num-beams $num_beams \
    --num-beam-groups $num_beam_groups \
    --penalty-alpha $penalty_alpha \
    --temperature $temperature \
    --top-k $top_k \
    --top-p $top_p \
    --repetition_penalty $repetition_penalty \