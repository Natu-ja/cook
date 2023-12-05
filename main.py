import argparse
from datasets import Dataset, load_from_disk
import datetime
import os
import pandas as pd
from peft import get_peft_model, LoraConfig, PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, Trainer, TrainingArguments
from typing import Any

def load_tokenize_data(args, tokenizer):

    dataset = load_from_disk(args.data)
    train_dataset = dataset['train']
    val_dataset = dataset['val']
    test_dataset = dataset['test'] if args.generation == 'yes' else None
    
    if args.instruction != None:

        def instruct(args, dataset):

            def build_prompt(args, inputs="", sep="\n\n### "):
                system_message = args.system_message
                roles = ["指示", "応答"]
                messages = [": \n" + args.instruction, ": "]
                if inputs:
                    roles.insert(1, "入力")
                    messages.insert(1, ": \n" + inputs)
                for role, message in zip(roles, messages):
                    system_message += sep + role, message
                return message

            dataset = dataset.to_pandas()
            for i in range(len(dataset)):
                user_inputs = {
                    "args": args,
                    "inputs": dataset['材料'][i]
                }
                dataset['材料'][i] = build_prompt(args, **user_inputs)
            dataset = Dataset.from_pandas(dataset)
            return dataset
        
        train_dataset = instruct(args, train_dataset)
        val_dataset = instruct(args, val_dataset)
        if test_dataset != None: test_dataset = instruct(args, test_dataset)

    def preprocess(data):

        inputs = tokenizer(data['材料'], truncation=True, max_length=args.input_max_len, padding=True)
        labels = tokenizer(data['タイトル'], truncation=True, max_length=args.input_max_len, padding=True)

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
    val_dataset = val_dataset.map(
        preprocess,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    print(f'eval dataset: {len(val_dataset)} samples!!')
    
    return train_dataset, val_dataset, test_dataset
    
def load(args):
    try:
        config = PeftConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        print(f'Loaded tokenizer from {args.tokenizer}!!')

        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        print(f'Loaded model from {args.model}, model size {model.num_parameters()}!!')

        model = PeftModel.from_pretrained(model, args.model)
        print(f'Applying LoRA to the model, the trainable parameters is {model.print_trainable_parameters()}!!')
    
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
            print(f'Applying LoRA to the model, the trainable parameters is {model.print_trainable_parameters()}!!')
    
    return tokenizer, model

def run_training(args, tokenizer, model, train_dataset, val_dataset, test_dataset):

    training_args=TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=False,
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
        metric_for_best_model=args.metric_for_best_model,
        report_to=args.report_to,
        auto_find_batch_size=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    print('Start training!!')
    trainer.train()
    print('Finish training!!')

    if args.generation == 'yes':
        print('Start generation!!')
        df = pd.DataFrame(columns=['材料', '正解タイトル', '予測タイトル'])
        for i in range(len(test_dataset)):
            df.loc[i, '材料'] = test_dataset['材料'][i]
            df.loc[i, '正解タイトル'] = test_dataset['正解タイトル'][i]
            inputs = tokenizer(test_dataset['材料'][i], add_special_tokens=False, return_tensors='pt')['input_ids'].cuda()
            for j in range(args.num_beams):
                outputs = model.generate(
                    **inputs,
                    max_length=args.max_length,
                    min_length=args.min_length,
                    do_sample=args.do_sample,
                    num_beams=j+1,
                    num_beams_group=args.num_beams_group,
                    penalty=args.penalty_alpha,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                df.loc[i, f'予測タイトル (num_beams={j+1})'] = tokenizer.decode(outputs[0].tolist())
        
        print('Finish generation!!')
        df.to_csv(args.save_dir+args.generation_file_name, index=False)
        print(f'Saved in {args.save_dir+args.generation_file_name}')
    
def main(args):
    tokenizer, model = load(args)
    train_dataset, val_dataset, test_dataset = load_tokenize_data(args, tokenizer)
    run_training(args, tokenizer, model, train_dataset, val_dataset, test_dataset)

if __name__ == "__main__":

    dt_now = datetime.datetime.now()
    parser = argparse.ArgumentParser()

    parser.add_argument('tokenizer', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('--data', default='./data', type=str)
    parser.add_argument('--input-max-len', default=128, type=int)

    # generate
    parser.add_argument('--generation', default='no', type=str, choices=['no', 'yes'])
    parser.add_argument('--generation-file-name', default='/generation.csv', type=str)
    parser.add_argument('--max-length', default=20, type=int)
    parser.add_argument('--min-length', default=0, type=int)
    parser.add_argument('--do-sample', default=False, type=bool)
    parser.add_argument('--num-beams', default=1, type=int)
    parser.add_argument('--num-beams-group', default=1, type=int)
    parser.add_argument('--penalty-alpha', default=0, type=float)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--top-k', default=50, type=int)
    parser.add_argument('--top-p', default=1, type=float, choices=[0, 1])

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
    parser.add_argument('--metric-for-best-model', default='eval_loss', type=str)
    parser.add_argument('--report-to', default='all', type=str, choices=['azure_ml', 'clearml', 'codecarbon', 'comet_ml', 'dagshub', 'flyte', 'mlflow', 'neptune', 'tensorboard', 'wandb'])

    # LoraConfig
    parser.add_argument('--rank', default=8, type=int)
    parser.add_argument('--target-modules', nargs='*', type=str)
    parser.add_argument('--lora-alpha', default=8, type=int)
    parser.add_argument('--lora-dropout', default=0, type=float)
    parser.add_argument('--lora-bias', default='none', type=str, choices=['none', 'all', 'lora_only'])

    # Instruction tuning
    parser.add_argument('--system-message', default='', type=str)
    parser.add_argument('--instruction', type=str)

    args = parser.parse_args()

    os.makedirs(name=args.save_dir, exist_ok=True)
    
    main(args)