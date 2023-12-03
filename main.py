import argparse
from datasets import Dataset, load_from_disk
import datetime
import os
from peft import get_peft_model, LoraConfig, PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, Trainer, TrainingArguments
from typing import Any

def load_tokenize_data(args, tokenizer):

    dataset = load_from_disk(args.data)
    train_dataset = dataset['train']
    eval_dataset = dataset['eval']

    def instruction(args, dataset):
        dataset = dataset.to_pandas()
        dataset = args.instruction + dataset['食材']
        dataset = Dataset.from_pandas(dataset)
        return dataset
    
    if args.instruction != None:
        train_dataset = instruction(train_dataset)
        eval_dataset = instruction(eval_dataset)

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
        remove_columns=train_dataset.column_names,
        num_proc=8
    )
    print(f'train dataset: {len(train_dataset)} samples!!')
    eval_dataset = eval_dataset.map(
        preprocess,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    print(f'eval dataset: {len(eval_dataset)} samples!!')
    
    return train_dataset, eval_dataset
    
def load(args):
    try:
        config = PeftConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        print(f'Loaded tokenizer from {args.tokenizer}!!')

        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        print(f'Loaded model from {args.model}, model size {model.num_parameters()}!!')

        model = PeftModel.from_pretrained(model, args.model)
        print('Adapt Peft!!')
        model.print_trainable_parameters()
    
    except:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except:
            tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)
        print(f'Loaded tokenizer from {args.tokenizer}!!')

        model = AutoModelForCausalLM.from_pretrained(args.model)
        print(f'Loaded model from {args.model}, model size {model.num_parameters()}!!')

        if args.target_modules != None:
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
    
    return tokenizer, model

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
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        logging_strategy=args.strategy,
        save_strategy=args.strategy,
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        run_name=args.run_name,
        remove_unused_columns=True,
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
    tokenizer, model = load(args)
    train_dataset, eval_dataset = load_tokenize_data(args, tokenizer)
    run_training(model, train_dataset, eval_dataset)

if __name__ == "__main__":

    dt_now = datetime.datetime.now()
    parser = argparse.ArgumentParser()

    parser.add_argument('tokenizer', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('--data', default='./data', type=str)
    parser.add_argument('--input-max-len', default=128, type=int)
    parser.add_argument('--instruction', type=str)

    # TrainingArguments
    parser.add_argument('--save-dir', default='./output/'+dt_now.strftime('%Y_%m_%d_%H_%M_%S'), type=str)
    parser.add_argument('--strategy', default='epoch', type=str, choices=['no', 'steps', 'epoch'])
    parser.add_argument('--per-device-train-batch-size', default=8, type=int)
    parser.add_argument('--per-device-eval-batch-size', default=8, type=int)
    parser.add_argument('--gradient-accumulation', default=1, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--weight-decay', default=0, type=float)
    parser.add_argument('--max-grad-norm', default=1, type=1)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--scheduler', default='linear', type=str, choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'])
    parser.add_argument('--warmup', default=0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--run-name', type=str)
    parser.add_argument('--report-to', default='all', type=str, choices=['azure_ml', 'clearml', 'codecarbon', 'comet_ml', 'dagshub', 'flyte', 'mlflow', 'neptune', 'tensorboard', 'wandb'])

    #LoraConfig
    parser.add_argument('--rank', default=8, type=int)
    parser.add_argument('--target-modules', nargs='*', type=str)
    parser.add_argument('--lora-alpha', default=8, type=float)
    parser.add_argument('--lora-dropout', default=0, type=float)
    parser.add_argument('--lora-bias', default='none', type=str, choices=['none', 'all', 'lora_only'])

    args = parser.parse_args()

    os.makedirs(name=args.save_dir, exist_ok=True)
    
    main(args)