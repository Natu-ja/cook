from datasets import Dataset, load_from_disk

def load_tokenize_data(args, tokenizer):

    dataset = load_from_disk(args.data)
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']
    print(f'train : val : test = {len(train_dataset)} : {len(val_dataset)} : {len(test_dataset)}!!')
    
    if args.instruction != None:

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
        
        train_dataset = instruct(args, train_dataset)
        val_dataset = instruct(args, val_dataset)
        test_dataset = instruct(args, test_dataset)

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