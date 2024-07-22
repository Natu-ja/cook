import os
from argparse import Namespace
from datasets.arrow_dataset import Dataset
from datasets.formatting.formatting import LazyBatch

def load_raw_dataset(args: Namespace) -> tuple[Dataset, Dataset, Dataset] | tuple[Dataset, None, None]:

    if not os.path.exists(path=args.dataset):

        from datasets import load_dataset

        train_dataset = load_dataset(path=args.dataset, split="train")

        return train_dataset, None, None
    
    else:

        import pandas as pd

        dataset = pd.read_csv(filepath_or_buffer=args.dataset, sep="\t")
        dataset = Dataset.from_pandas(df=dataset)

        dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=args.seed)
        tv_dataset, test_dataset = dataset["train"], dataset["test"]
        tv_dataset = tv_dataset.train_test_split(test_size=0.2, shuffle=True, seed=args.seed)
        train_dataset, eval_dataset = tv_dataset["train"], tv_dataset["test"]

        return train_dataset, eval_dataset, test_dataset

def formatting_func_cookpad(example: LazyBatch) -> list[str]:
    output_texts = [f"# ユーザ\n{example['title'][i]}\n\n# アシスタント\n## 食材\n{example['name'][i]}\n## 作り方\n{example['position'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_zh_tw_recipes_sm(example: LazyBatch) -> list[str]:
    output_texts = [f"# 標題\n{example['title'][i]}\n\n# 腳步\n{example['steps'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_data_recipes_instructor(example: LazyBatch) -> list[str]:
    output_texts = [f"# Instruction\n{example['instruction'][i]}\n\n# Input\n{example['input'][i]}\n\n# Output\n{example['output'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_thai_food(example: LazyBatch) -> list[str]:
    output_texts = [f"# ชื่อ\n{example['name'][i]}\n\n# ข้อความ\n{example['text'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_aya_telugu_food_recipes(example: LazyBatch) -> list[str]:
    output_texts = [f"# ఇన్పుట్\n{example['inputs'][i]}\n\n# లక్ష్యం\n{example['target'][i]}" for i in range(len(example))]
    return output_texts