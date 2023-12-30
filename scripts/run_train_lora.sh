tokenizer=TOKENIZER-NAME
model=MODEL-NAME
data=./data
input_max_len=128
kfold=1

# TrainingArguments
train_batch_size=8
eval_batch_size=8
gradients=1
lr=5e-5
weight_decay=0.0
adam_beta1=0.9
adam_beta2=0.999
adam_epsilon=1e-8
max_grad_norm=1.0
epochs=3.0
scheduler=linear
warmup=0.0
seed=42
local_rank=-1
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
    --kfold $kfold \
    --train-batch-size $train_batch_size \
    --eval-batch-size $eval_batch_size \
    --gradients $gradients \
    --lr $lr \
    --weight-decay $weight_decay \
    --adam-beta1 $adam_beta1 \
    --adam-beta2 $adam_beta2 \
    --adam-epsilon $adam_epsilon \
    --max-grad-norm $max_grad_norm \
    --epochs $epochs \
    --scheduler $scheduler \
    --warmup $warmup \
    --seed $seed \
    --local_rank $local_rank \
    --metric-for-best-model $metric_for_best_model \
    --rank $rank \
    --target-modules ${target_modules[@]} \
    --lora-alpha $lora_alpha \
    --lora-dropout $lora_dropout \
    --lora-bias $lora_bias \