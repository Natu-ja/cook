from datasets import Dataset, load_from_disk

def load_raw_dataset(args):
    dataset = load_from_disk(args.data)
    print(f'dataset: {len(dataset)} samples!!')
    return dataset

def train_test_data_split(args, dataset):
    return dataset.train_test_split(test_size=0.25, train_size=0.75, shuffle=True, random_state=args.seed)

def train_val_data_split(args, dataset):
    return dataset.train_test_split(test_size=0.25, train_size=0.75, shuffle=True, random_state=args.seed)

def instruct(args, dataset):

    def build_prompt(inputs="", sep="\n\n### "):
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
        dataset['材料'][i] = build_prompt(**user_inputs)
    dataset = Dataset.from_pandas(dataset)
    return dataset

def load_tokenize_data(args, tokenizer, dataset):
    
    if args.instruction != None:
        dataset = instruct(args, dataset)

    def preprocess(data):

        inputs = tokenizer(data['材料'], truncation=True, max_length=args.input_max_len)
        labels = tokenizer(data['タイトル'], truncation=True, max_length=args.input_max_len)

        inputs['input_ids'] = inputs['input_ids']
        inputs['attention_mask'] = inputs['attention_mask']
        inputs['labels'] = labels['input_ids']
        
        return inputs
    
    dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names
    )
    return dataset