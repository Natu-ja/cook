tokenizer=llm-jp/llm-jp-1.3b-v1.0
model=llm-jp/llm-jp-1.3b-v1.0
data=./data/recipes.tsv
input=title
label=description

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
peft_method=lora
rank=8
target_modules=(q_proj v_proj)
lora_alpha=8
lora_dropout=0.0
peft_bias=none
target_r=8
init_r=12
tinit=0
tfinal=0
deltaT=1
peft_beta1=0.85
peft_beta2=0.85
orth_reg_weight=0.5

# Instruction tuning
system_message=以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。
instruction=以下の食材を使って作れる料理名を教えてください。

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
python main.py \
    --tokenizer $tokenizer \
    --model $model \
    --data $data \
    --input $input \
    --label $label \
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
    --peft-method $peft_method \
    --rank $rank \
    --target-modules ${target_modules[@]} \
    --lora-alpha $lora_alpha \
    --lora-dropout $lora_dropout \
    --peft-bias $peft_bias \
    --target-r $target_r \
    --init-r $init_r \
    --tinit $tinit \
    --tfinal $tfinal \
    --deltaT $deltaT \
    --peft-beta1 $peft_beta1 \
    --peft-beta2 $peft_beta2 \
    --orth-reg-weight $orth_reg_weight \
    --system-message $system_message \
    --instruction $instruction\
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