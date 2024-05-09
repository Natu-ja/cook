import pandas as pd
import os
from datasets import Dataset

def load_raw_dataset(args):
    dataset = pd.read_csv(args.data, sep='\t')

    dataset = data_cleaning(args, dataset)

    dataset = Dataset.from_pandas(dataset)
    print(f'dataset: {len(dataset)} samples')

    return dataset

def data_cleaning(args, dataset):

    print('Start data cleaning')

    dataset = dataset.dropna(subset=[args.input, args.label])
    dataset = dataset.drop_duplicates(subset=[args.input, args.label])

    print('Finish data cleaning')

    return dataset

def tv_test_data_split(args, dataset):
    dataset = dataset.train_test_split(test_size=0.001, train_size=0.01, shuffle=True, seed=args.seed)
    train_val_dataset = dataset['train']
    test_dataset = dataset['test']
    return train_val_dataset, test_dataset

def train_val_data_split(args, dataset):
    dataset = dataset.train_test_split(test_size=0.25, train_size=0.75, shuffle=True, seed=args.seed)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    return train_dataset, val_dataset

def build_prompt(args, inputs="", sep="\n\n### "):
        
        system_message = args.system_message
        roles = ["指示", "応答"]
        messages = [": \n" + args.instruction, ": "]
        if inputs:
            roles.insert(1, "入力")
            messages.insert(1, ": \n" + inputs)
        for role, message in zip(roles, messages):
            system_message += sep + role + message
        return message

def instruct(args, dataset):

    dataset = dataset.to_pandas()
    for i in range(len(dataset)):
        user_inputs = {
            "args": args,
            "inputs": dataset[args.input][i]
        }
        dataset[args.input][i] = build_prompt(**user_inputs)
    dataset = Dataset.from_pandas(dataset)
    return dataset

def load_tokenize_data(args, tokenizer, dataset):
    
    if args.instruction is not None:
        dataset = instruct(args, dataset)

    def preprocess(data):
        inputs = tokenizer(data[args.input], add_special_tokens=True, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        labels = tokenizer(data[args.label], add_special_tokens=True, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

        inputs['input_ids'] = inputs['input_ids']
        inputs['attention_mask'] = inputs['attention_mask']
        inputs['labels'] = labels['input_ids']
        
        return inputs
    
    dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=None if args.num_proc else len(os.sched_getaffinity(0))
    )

    return dataset