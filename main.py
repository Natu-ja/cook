import argparse
from typing import Any
from datasets import load_from_disk
import os
from peft import get_peft_model, LoraConfig 
from transformers import AutoModelForCausalLM, LlamaTokenizer, Trainer, TrainingArguments

def load_tokenize_data(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)
    print(f'Loaded tokenizer from {args.tokenizer}!!')

    dataset = load_from_disk(args.data)
    train_dataset = dataset['train']
    eval_dataset = dataset['eval']

    def preprocess(data):
        inputs = tokenizer(data['食材'], truncation=True, max_length=args.input_max_len, padding=True)
        labels = tokenizer(data['料理'], truncation=True, max_length=args.input_max_len, padding=True)

        inputs['input_ids'] = inputs['input_ids']
        inputs['attention_mask'] = inputs['attention_mask']
        inputs['labels'] = labels['input_ids']
        
        return inputs
    
    train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    print(f'train dataset: {len(train_dataset)} samples!!')
    eval_dataset = eval_dataset.map(
        preprocess,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    print(f'eval dataset: {len(eval_dataset)} samples!!')
    
    return train_dataset, eval_dataset
    
def adapt_peft(model):
    peft_config = LoraConfig(
        r=args.rank, 
        target_modules=args.target_modules, 
        lora_alpha=args.lora_alpha, 
        lora_dropout =args.lora_dropout, 
        bias=args.lora_bias
    )

    model = get_peft_model(model, peft_config)
    print('Adapt LoRA!!')
    model.print_trainable_parameters()
    
    return model

def run_training(args, model, train_dataset, eval_dataset):

    training_args=TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        evaluation_strategy=args.strategy,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        logging_strategy=args.strategy,
        save_strategy=args.strategy,
        save_total_limit=5,
        seed=args.seed,
        data_seed=args.seed,
        run_name=args.run_name,
        remove_unused_columns = True,
        load_best_model_at_end=True,
        report_to=args.report_to
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=args.tokenizer
    )
    
    print('Start fine-tuning!!')
    trainer.train()
    print('Finish fine-tuning!!')
    
def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model)
    print(f'Loaded model from {args.model}, model size {model.num_parameters()}!!')

    model = adapt_peft(model)
    
    train_dataset, eval_dataset = load_tokenize_data(args)

    run_training(model, train_dataset, eval_dataset)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--tokenizer', default='novelai/nerdstash-tokenizer-v1', type=str)
    parser.add_argument('--model', default='stabilityai/japanese-stablelm-instruct-alpha-7b', type=str)
    parser.add_argument('--data', default='./data/', type=str)
    parser.add_argument('--input-max-len', default=128, type=int)

    # TrainingArguments
    parser.add_argument('--save-dir', default='./output', type=str)
    parser.add_argument('--strategy', default='epoch', type=str, choices=['no', 'steps', 'epoch'])
    parser.add_argument('--per-device-train-batch-size', default=4, type=int)
    parser.add_argument('--per-device-eval-batch-size', default=1, type=int)
    parser.add_argument('--gradient-accumulation', default=16, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--weight-decay', default=0, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--scheduler', default='linear', type=str, choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'])
    parser.add_argument('--warmup', default=0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--run-name', type=str)
    parser.add_argument('--report-to', default='all', type=str, choices=['azure_ml', 'clearml', 'codecarbon', 'comet_ml', 'dagshub', 'flyte', 'mlflow', 'neptune', 'tensorboard', 'wandb'])

    #LoraConfig
    parser.add_argument('--rank', default=8, type=int)
    parser.add_argument('--target-modules', nargs='*', default=['embed_in', 'query_key_value', 'dense', 'packed_input_proj', 'out_proj', 'embed_out'], type=str)
    parser.add_argument('--lora-alpha', default=16, type=float)
    parser.add_argument('--lora-dropout', default=0, type=float)
    parser.add_argument('--lora-bias', default='none', type=str, choices=['none', 'all', 'lora_only'])

    args = parser.parse_args()

    os.makedirs(name=args.save_dir, exist_ok=True)
    
    main(args)