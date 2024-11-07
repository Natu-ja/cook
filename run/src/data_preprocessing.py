import os
from argparse import Namespace
from datasets.arrow_dataset import Dataset
from datasets.formatting.formatting import LazyBatch

def load_raw_dataset(args: Namespace) -> tuple[Dataset, Dataset] | tuple[Dataset, None]:

    """
    Reads the specified data set and splits it into train and validation dataset.

    Args:
        args (`Namespace`):
            Arguments containing settings such as the path and seed value of the dataset.

    Returns:
        `tuple[datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset]`:
            Returns dataset for train and validation.
        `tuple[datasets.arrow_dataset.Dataset, None]`:
            Returns dataset for train only.
    """

    if os.path.exists(path=args.dataset):

        import pickle
        import pandas as pd

        def text_preprocessing(example: Dataset) -> Dataset:
                
            """
            Preprocesses the text data.

            Args:
                example (`datasets.arrow_dataset.Dataset`):
                    A dataset containing text data.

            Returns:
                `datasets.arrow_dataset.Dataset`:
                    Returns the preprocessed dataset.
            """

            import re
            import neologdn
            import demoji

            processed_example = {}

            for key, value in example.items():

                if isinstance(value, str):

                    value = value.replace("\n", "").replace("\r", "")
                    value = re.sub(r"http?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", value)
                    value = re.sub(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", value)
                    value = demoji.replace(string=value, repl="")
                    value = re.sub(r'[!"#$%&\'\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]', "", value)
                    value = re.sub("[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]", "", value)
                    
                    value = neologdn.normalize(value)
                    value = value.lower()

                    processed_example[key] = value

                else:
                    processed_example[key] = value

            return processed_example
        
        dataset = pd.read_csv(filepath_or_buffer=args.dataset)

        dataset = Dataset.from_pandas(df=dataset)

        if args.text_normalizer:
            dataset = dataset.map(function=text_preprocessing, num_proc=args.num_proc)

        dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=args.seed)
        tv_dataset, test_dataset = dataset["train"], dataset["test"]
        tv_dataset = tv_dataset.train_test_split(test_size=0.2, shuffle=True, seed=args.seed)
        train_dataset, eval_dataset = tv_dataset["train"], tv_dataset["test"]

        with open(file=args.output_dir + "/test_dataset.pkl", mode="wb") as f:
            pickle.dump(test_dataset, f)

        return train_dataset, eval_dataset

    else:

        from datasets import load_dataset

        train_dataset = load_dataset(path=args.dataset, split="train")

        return train_dataset, None

def formatting_func_cookpad(example: LazyBatch) -> list[str]:

    """
    Formats a batch of recipe examples into a specific text format.

    Args:
        example (`datasets.formatting.formatting.LazyBatch`):
            A batch of recipe examples containing 'title' and 'name' fields.
    """

    output_texts = [f"# ユーザ\n## タイトル\n{example['title'][i]}\n\n# アシスタント\n## 食材\n{example['ingredients'][i]}\n## 作り方\n{example['steps'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_data_recipes_instructor(example: LazyBatch) -> list[str]:

    """
    Formats a batch of recipe examples into a specific text format.

    Args:
        example (`datasets.formatting.formatting.LazyBatch`):
            A batch of recipe examples containing 'instruction' and 'input' fields.
    """

    output_texts = [f"# User\n## Instruction\n{example['instruction'][i]}\n\n## Input\n{example['input'][i]}\n\n#Assistant\n## Output\n{example['output'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_Recipes_Greek(example: LazyBatch) -> list[str]:

    """
    Formats a batch of recipe examples into a specific text format.

    Args:
        example (`datasets.formatting.formatting.LazyBatch`):
            A batch of recipe examples containing 'name' 'Ingredients', and 'Instructions' fields.
    """

    output_texts = [f"# Μεταχειριζόμενος\n## όνομα\n{example['name'][i]}\n\n# Βοηθός\n## Συστατικά\n{example['Ingredients'][i]}\n## Οδηγίες\n{example['Instructions'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_all_recipes(example: LazyBatch) -> list[str]:

    """
    Formats a batch of recipe examples into a specific text format.

    Args:
        example (`datasets.formatting.formatting.LazyBatch`):
            A batch of recipe examples containing 'title', 'ingredients' and 'steps' fields.
    """

    output_texts = [f"# User\n## Name\n{example['name'][i]}\n\n#Assistant\n## Ingredients\n{example['ingredients'][i]}\n## Steps\n{example['steps'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_zh_tw_recipes_sm(example: LazyBatch) -> list[str]:

    """
    Formats a batch of recipe examples into a specific text format.

    Args:
        example (`datasets.formatting.formatting.LazyBatch`):
            A batch of recipe examples containing 'title' and 'steps' fields.
    """

    output_texts = [f"# 用户\n## 標題\n{example['title'][i]}\n\n# 助手\n## 腳步\n{example['steps'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_aya_telugu_food_recipes(example: LazyBatch) -> list[str]:

    """
    Formats a batch of recipe examples into a specific text format.

    Args:
        example (`datasets.formatting.formatting.LazyBatch`):
            A batch of recipe examples containing 'inputs' and 'target' fields.
    """

    output_texts = [f"# వినియోగదారు\n## ఇన్పుట్\n{example['inputs'][i]}\n\n# సహాయకుడు\n## లక్ష్యం\n{example['target'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_thai_food(example: LazyBatch) -> list[str]:

    """
    Formats a batch of recipe examples into a specific text format.

    Args:
        example (`datasets.formatting.formatting.LazyBatch`):
            A batch of recipe examples containing 'name' and 'text' fields.
    """

    output_texts = [f"# ผู้ใช้\n## ชื��อ\n{example['name'][i]}\n\n# ผู้ช่วย\n## ข้อความ\n{example['text'][i]}" for i in range(len(example))]
    return output_texts