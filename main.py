import argparse
from sklearn.model_selection import KFold
import pandas as pd
# import fireducks.pandas as pd
import os
import datetime
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.data import *
from src.model import *

def run_training(args, tokenizer, model, train_dataset, val_dataset, test_dataset, fold=None):
    training_args=TrainingArguments(
        output_dir=args.dir if fold is None else args.dir+f'/{fold}',
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy=args.strategy,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradients,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup,
        logging_dir=args.dir if fold is None else args.dir+f'/{fold}',
        logging_strategy=args.strategy,
        save_strategy=args.strategy,
        save_total_limit=1,
        seed=args.seed,
        data_seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
        bf16_full_eval=args.bf16,
        fp16_full_eval=args.fp16,
        local_rank=args.local_rank,
        dataloader_num_workers=0 if args.dataloader_num_workers else len(os.sched_getaffinity(0)),
        run_name=args.run_name,
        remove_unused_columns=True,
        label_names=['labels'],
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        deepspeed=args.deepspeed,
        group_by_length=args.group_by_length,
        report_to=args.report_to,
        auto_find_batch_size=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    print('Start main loop!!')
    trainer.train()
    print('Finish main loop!!')

    if args.generation:

        print('Start generation!!')
        
        df = pd.DataFrame(columns=[args.input, f'ground_truth({args.ouput})', f'prediction({args.ouput})'])
        for i, test_data in enumerate(test_dataset):
            df.loc[i, args.input] = test_data[args.input]
            df.loc[i, f'ground_truth({args.ouput})'] = test_data[args.ouput]
            inputs = tokenizer(test_data[args.input], add_special_tokens=True, return_tensors='pt')['input_ids'].cuda()
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
                bad_words_ids=[[tokenizer.unk_token_id]],
                renormalize_logits=args.renormalize_logits,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            df.loc[i, f'prediction({args.ouput})'] = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
        
        print('Finish generation!!')
        df.to_csv(args.dir+args.generation_file_name, index=False)
        print(f'Saved in {args.dir+args.generation_file_name}')

def main(args):

    dataset = load_raw_dataset(args)
    train_val_dataset, test_dataset = tv_test_data_split(args, dataset)

    if args.n_splits == 1:
        tokenizer, model = load(args)
        train_dataset, val_dataset = train_val_data_split(args, train_val_dataset)
        train_dataset = load_tokenize_data(args, tokenizer, train_dataset)
        val_dataset = load_tokenize_data(args, tokenizer, val_dataset)
        print(f'train : val : test = {len(train_dataset)} : {len(val_dataset)} : {len(test_dataset)}!!')
        run_training(args, tokenizer, model, train_dataset, val_dataset, test_dataset)
        
    elif args.n_splits > 1:
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
            print(f'Fold {fold+1} / {args.n_splits}')
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

    parser.add_argument('--tokenizer', default='tokyotech-llm/Swallow-7b-hf', type=str)
    parser.add_argument('--model', default='tokyotech-llm/Swallow-7b-hf', type=str)
    parser.add_argument('--input', default='title', type=str)
    parser.add_argument('--output', default='description', type=str)
    parser.add_argument('--args-file-name', default='/args.json', type=str)
    parser.add_argument('--data', default='./data/recipes.tsv', type=str)
    parser.add_argument('--input-max-len', default=128, type=int)
    parser.add_argument('--n-splits', default=1, type=int)
    parser.add_argument('--num-proc', action='store_false')

    # TrainingArguments
    parser.add_argument('--dir', default='./output/'+dt_now.strftime('%Y_%m_%d_%H_%M_%S'), type=str)
    parser.add_argument('--strategy', default='epoch', type=str, choices=['no', 'steps', 'epoch'])
    parser.add_argument('--train-batch-size', default=8, type=int)
    parser.add_argument('--eval-batch-size', default=8, type=int)
    parser.add_argument('--gradients', default=1, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--weight-decay', default=0.0, type=float)
    parser.add_argument('--adam-beta1', default=0.9, type=float)
    parser.add_argument('--adam-beta2', default=0.999, type=float)
    parser.add_argument('--adam-epsilon', default=1e-8, type=float)
    parser.add_argument('--max-grad-norm', default=1.0, type=float)
    parser.add_argument('--epochs', default=3.0, type=float)
    parser.add_argument('--scheduler', default='linear', type=str, choices=['linear', 'cosine', 'constant'])
    parser.add_argument('--warmup', default=0.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16-opt-level', default='O1', type=str, choices=['O0', 'O1', 'O2', 'O3'])
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dataloader-num-workers', action='store_false')
    parser.add_argument('--run-name', default=dt_now.strftime('%Y_%m_%d_%H_%M_%S'), type=str)
    parser.add_argument('--metric-for-best-model', default='eval_loss', type=str)
    parser.add_argument('--deepspeed', type=str)
    parser.add_argument('--group-by-length', action='store_true')
    parser.add_argument('--report-to', default='all', type=str, choices=['azure_ml', 'clearml', 'codecarbon', 'comet_ml', 'dagshub', 'flyte', 'mlflow', 'neptune', 'tensorboard', 'wandb'])

    # PEFT
    parser.add_argument('--peft-method', default='lora', type=str, choices=['lora', 'adalora'])
    parser.add_argument('--rank', default=8, type=int)
    parser.add_argument('--target-modules', nargs='*', type=str)
    parser.add_argument('--lora-alpha', default=8, type=int)
    parser.add_argument('--lora-dropout', default=0.0, type=float)
    parser.add_argument('--fan-in-fan-out', action='store_true')
    parser.add_argument('--peft-bias', default='none', type=str, choices=['none', 'all', 'lora_only'])
    parser.add_argument('--target-r', default=8, type=int)
    parser.add_argument('--init-r', default=12, type=int)
    parser.add_argument('--tinit', default=0, type=int)
    parser.add_argument('--tfinal', default=0, type=int)
    parser.add_argument('--deltaT', default=1, type=int)
    parser.add_argument('--peft-beta1', default=0.85, type=float)
    parser.add_argument('--peft-beta2', default=0.85, type=float)
    parser.add_argument('--orth-reg-weight', default=0.5, type=float)

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
    
    os.makedirs(name=args.dir, exist_ok=True)
    
    main(args)