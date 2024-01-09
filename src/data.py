import pandas as pd
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from datasets import Dataset

def load_raw_dataset(args):
    dataset = pd.read_csv(args.data, sep='\t')

    dataset = data_cleaning(args, dataset)

    dataset = Dataset.from_pandas(dataset)
    print(f'dataset: {len(dataset)} samples')

    return dataset

def data_cleaning(args, dataset):

    print('Start data cleaning')

    dataset = dataset.dropna(subset=[args.input, args.output])
    dataset = dataset.drop_duplicates(subset=[args.input, args.output])

    print('Finish data cleaning')

    return dataset

def tv_test_data_split(args, dataset):
    dataset = dataset.train_test_split(test_size=0.0001, train_size=0.01, shuffle=True, seed=args.seed)
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
            system_message += sep + role, message
        return message

def instruct(args, dataset):

    dataset = dataset.to_pandas()
    for i in range(len(dataset)):
        user_inputs = {
            "args": args,
            "inputs": dataset[args.input][i]
        }
        dataset[args.input][i] = build_prompt(args, **user_inputs)
    dataset = Dataset.from_pandas(dataset)
    return dataset

def load_tokenize_data(args, tokenizer, dataset):
    
    if args.instruction is not None:
        dataset = instruct(args, dataset)

    def preprocess(data):

        inputs = tokenizer(data[args.input], add_special_tokens=True)
        labels = tokenizer(data[args.output], add_special_tokens=True)

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

class CustomDataCollator(DataCollatorForLanguageModeling):
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = False

    def __call__(self, examples):
        input_ids = [example['input_ids'] for example in examples]
        labels = [example['labels'] for example in examples]

        max_length = max(max(len(ids) for ids in input_ids), max(len(ids) for ids in labels))

        padded_input_ids = pad_sequence([torch.tensor(ids + [self.tokenizer.pad_token_id] * (max_length - len(ids))) for ids in input_ids], batch_first=True)
        padded_labels = pad_sequence([torch.tensor(ids + [-100] * (max_length - len(ids))) for ids in labels], batch_first=True)

        return {"input_ids": padded_input_ids, "labels": padded_labels}