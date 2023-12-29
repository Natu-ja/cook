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

# generate
max_length=20
min_length=0
num_beams=1
num_beam_groups=1
penalty_alpha=0.0
temperature=1.0
top_k=50
top_p=1.0
typical_p=1.0
epsilon_cutoff=0.0
eta_cutoff=0.0
diversity_penalty=0.0
repetition_penalty=1.0
encoder_repetition_penalty=1.0
length_penalty=1.0
no_repeat_ngram_size=0

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
    --generation \
    --max-length $max_length \
    --min-length $min_length \
    --num-beams $num_beams \
    --num-beam-groups $num_beam_groups \
    --penalty-alpha $penalty_alpha \
    --temperature $temperature \
    --top-k $top_k \
    --top-p $top_p \
    --typical-p $typical_p \
    --epsilon-cutoff $epsilon_cutoff \
    --eta-cutoff $eta_cutoff \
    --diversity-penalty $diversity_penalty \
    --repetition_penalty $repetition_penalty \
    --encoder-repetition-penalty $encoder_repetition_penalty \
    --length-penalty $length_penalty \
    --no-repeat-ngram-size $no_repeat_ngram_size \