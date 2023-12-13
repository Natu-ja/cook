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