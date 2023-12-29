import argparse
from sklearn.model_selection import KFold
# import pandas as pd
import fireducks.pandas as pd
import os
import datetime
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from src.data import *
from src.model import *

def run_training(args, tokenizer, model, train_dataset, val_dataset, test_dataset, fold=None):
    if args.kfold == 1:
        training_args=TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=False,
            do_train=True,
            do_eval=True,
            evaluation_strategy=args.strategy,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradients,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            num_train_epochs=args.epochs,
            lr_scheduler_type=args.scheduler,
            warmup_ratio=args.warmup,
            logging_strategy=args.strategy,
            save_strategy=args.strategy,
            save_total_limit=1,
            seed=args.seed,
            data_seed=args.seed,
            bf16=args.bf16,
            fp16=args.fp16,
            fp16_opt_level=args.fp16_opt_level,
            run_name=args.run_name,
            remove_unused_columns=True,
            label_names=['labels'],
            load_best_model_at_end=True,
            metric_for_best_model=args.metric_for_best_model,
            deepspeed=args.deepspeed,
            report_to=args.report_to,
            auto_find_batch_size=True
        )
    elif args.kfold > 1:
        training_args=TrainingArguments(
            output_dir=args.output_dir+f'/fold_{fold+1}',
            overwrite_output_dir=False,
            do_train=True,
            do_eval=True,
            evaluation_strategy=args.strategy,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradients,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            num_train_epochs=args.epochs,
            lr_scheduler_type=args.scheduler,
            warmup_ratio=args.warmup,
            logging_strategy=args.strategy,
            save_strategy=args.strategy,
            save_total_limit=1,
            seed=args.seed,
            data_seed=args.seed,
            bf16=args.bf16,
            fp16=args.fp16,
            fp16_opt_level=args.fp16_opt_level,
            run_name=args.run_name+f'_fold_{fold+1}',
            remove_unused_columns=True,
            label_names=['labels'],
            load_best_model_at_end=True,
            metric_for_best_model=args.metric_for_best_model,
            deepspeed=args.deepspeed,
            report_to=args.report_to,
            auto_find_batch_size=True
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    print('Start main loop!!')
    trainer.train()
    print('Finish main loop!!')

    if args.generation:

        print('Start generation!!')
        
        df = pd.DataFrame(columns=['材料', '正解タイトル', '予測タイトル'])
        for i, test_data in enumerate(test_dataset):
            df.loc[i, '材料'] = test_data['材料']
            df.loc[i, '正解タイトル'] = test_data['正解タイトル']
            inputs = tokenizer(test_data['材料'], add_special_tokens=False, return_tensors='pt')['input_ids'].cuda()
            outputs = model.generate(
                **inputs,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                min_length=args.min_length,
                min_new_tokens=args.min_new_tokens,
                early_stopping=args.early_stopping,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                num_beam_groups=args.num_beam_groups,
                penalty_alpha=args.penalty_alpha,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                typical_p=args.typical_p,
                epsilon_cutoff=args.epsilon_cutoff,
                eta_cutoff=args.eta_cutoff,
                diversity_penalty=args.diversity_penalty,
                repetition_penalty=args.repetition_penalty,
                encoder_repetition_penalty=args.encoder_repetition_penalty,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                renormalize_logits=args.renormalize_logits
            )
            df.loc[i, '予測タイトル'] = tokenizer.decode(outputs[0].tolist())
        
        print('Finish generation!!')
        df.to_csv(args.save_dir+args.generation_file_name, index=False)
        print(f'Saved in {args.save_dir+args.generation_file_name}')

def main(args):

    dataset = load_raw_dataset(args)
    train_val_dataset, test_dataset = train_test_data_split(args, dataset)

    if args.kfold == 1:
        tokenizer, model = load(args)
        train_dataset, val_dataset = train_val_data_split(args, train_val_dataset)
        train_dataset = load_tokenize_data(args, tokenizer, train_dataset)
        val_dataset = load_tokenize_data(args, tokenizer, val_dataset)
        print(f'train : val : test = {len(train_dataset)} : {len(val_dataset)} : {len(test_dataset)}!!')
        run_training(args, tokenizer, model, train_dataset, val_dataset, test_dataset)
        
    elif args.kfold > 1:
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
            print(f'Fold {fold+1} / {args.kfold}')
            tokenizer, model = load(args)
            train_dataset = train_dataset.select(train_idx)
            val_dataset = train_dataset.select(val_idx)
            train_dataset = load_tokenize_data(args, tokenizer, train_dataset)
            val_dataset = load_tokenize_data(args, tokenizer, val_dataset)
            print(f'train : val : test = {len(train_dataset)} : {len(val_dataset)} : {len(test_dataset)}!!')
            run_training(args, tokenizer, model, train_dataset, val_dataset, test_dataset, fold)

if __name__ == "__main__":

    dt_now = datetime.datetime.now()
    parser = argparse.ArgumentParser()

    parser.add_argument('tokenizer', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('--args-file-name', default='/args.json', type=str)
    parser.add_argument('--data', default='./data', type=str)
    parser.add_argument('--input-max-len', default=128, type=int)
    parser.add_argument('--kfold', default=1, type=int)

    # TrainingArguments
    parser.add_argument('--output-dir', default='./output/'+dt_now.strftime('%Y_%m_%d_%H_%M_%S'), type=str)
    parser.add_argument('--strategy', default='epoch', type=str, choices=['no', 'steps', 'epoch'])
    parser.add_argument('--train-batch-size', default=8, type=int)
    parser.add_argument('--eval-batch-size', default=8, type=int)
    parser.add_argument('--gradients', default=1, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--weight-decay', default=0.0, type=float)
    parser.add_argument('--max-grad-norm', default=1.0, type=float)
    parser.add_argument('--epochs', default=3.0, type=float)
    parser.add_argument('--scheduler', default='linear', type=str, choices=['linear', 'cosine', 'constant'])
    parser.add_argument('--warmup', default=0.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16-opt-level', default='O1', type=str, choices=['O0', 'O1', 'O2', 'O3'])
    parser.add_argument('--run-name', default=dt_now.strftime('%Y_%m_%d_%H_%M_%S'), type=str)
    parser.add_argument('--metric-for-best-model', default='eval_loss', type=str)
    parser.add_argument('--deepspeed', type=str)
    parser.add_argument('--report-to', default='all', type=str, choices=['azure_ml', 'clearml', 'codecarbon', 'comet_ml', 'dagshub', 'flyte', 'mlflow', 'neptune', 'tensorboard', 'wandb'])

    # LoraConfig
    parser.add_argument('--rank', default=8, type=int)
    parser.add_argument('--target-modules', nargs='*', type=str)
    parser.add_argument('--lora-alpha', default=8, type=int)
    parser.add_argument('--lora-dropout', default=0.0, type=float)
    parser.add_argument('--fan-in-fan-out', action='store_true')
    parser.add_argument('--lora-bias', default='none', type=str, choices=['none', 'all', 'lora_only'])

    # Instruction tuning
    parser.add_argument('--system-message', default='', type=str)
    parser.add_argument('--instruction', type=str)

    # generate
    parser.add_argument('--generation', action='store_true')
    parser.add_argument('--generation-file-name', default='/generation.csv', type=str)
    parser.add_argument('--max-length', default=20, type=int)
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--min-length', default=0, type=int)
    parser.add_argument('--min-new-tokens', type=int)
    parser.add_argument('--early-stopping', action='store_true')
    parser.add_argument('--do-sample', action='store_true')
    parser.add_argument('--num-beams', default=1, type=int)
    parser.add_argument('--num-beam-groups', default=1, type=int)
    parser.add_argument('--penalty-alpha', default=0.0, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top-k', default=50, type=int)
    parser.add_argument('--top-p', default=1.0, type=float, choices=[0, 1])
    parser.add_argument('--typical-p', default=1.0, type=float, choices=[0, 1])
    parser.add_argument('--epsilon-cutoff', default=0.0, type=float, choices=[0, 1])
    parser.add_argument('--eta-cutoff', default=0.0, type=float, choices=[0, 1])
    parser.add_argument('--diversity-penalty', default=0.0, type=float)
    parser.add_argument('--repetition-penalty', default=1.0, type=float)
    parser.add_argument('--encoder-repetition-penalty', default=1.0, type=float)
    parser.add_argument('--length-penalty', default=1.0, type=float)
    parser.add_argument('--no-repeat-ngram-size', default=0, type=int)
    parser.add_argument('--renormalize-logits', action='store_true')

    args = parser.parse_args()
    
    os.makedirs(name=args.save_dir, exist_ok=True)
    
    main(args)