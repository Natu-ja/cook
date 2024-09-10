import os
from argparse import Namespace
from datasets.arrow_dataset import Dataset
from datasets.formatting.formatting import LazyBatch

def load_raw_dataset(args: Namespace) -> tuple[Dataset, Dataset, Dataset] | tuple[Dataset, None, None]:

    """
    Reads the specified data set and splits it into train, validation, and test dataset.

    Args:
        args (`Namespace`):
            Arguments containing settings such as the path and seed value of the dataset.

    Returns:
        `tuple[datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset]`:
            Returns dataset for train, validation, and test.
        `tuple[datasets.arrow_dataset.Dataset, None, None]`:
            Returns dataset for train only.
    """
    
    if os.path.exists(path=args.dataset):

        import pandas as pd

        def text_normalization(df: pd.DataFrame) -> pd.DataFrame:

            """
            String normalization processing.

            Args:
                df (`pandas.core.frame.DataFrame`):
                    DataFrame containing text data.
                
            Returns:
                `pandas.core.frame.DataFrame`:
                    Returns a DataFrame with normalized text data.
            """

            import neologdn

            for i in df.columns:
                df[i] = df[i].apply(lambda x: neologdn.normalize(x))
            
            return df

        dataset = pd.read_csv(filepath_or_buffer=args.dataset, sep="\t")

        if args.text_normalizer:
            dataset = text_normalization(df=dataset)

        dataset = Dataset.from_pandas(df=dataset)

        dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=args.seed)
        tv_dataset, test_dataset = dataset["train"], dataset["test"]
        tv_dataset = tv_dataset.train_test_split(test_size=0.2, shuffle=True, seed=args.seed)
        train_dataset, eval_dataset = tv_dataset["train"], tv_dataset["test"]

        return train_dataset, eval_dataset, test_dataset

    else:

        from datasets import load_dataset

        train_dataset = load_dataset(path=args.dataset, split="train")

        return train_dataset, None, None

def formatting_func_cookpad(example: LazyBatch) -> list[str]:

    """
    Formats a batch of recipe examples into a specific text format.

    Args:
        example (`datasets.formatting.formatting.LazyBatch`):
            A batch of recipe examples containing 'title' and 'name' fields.
    """

    output_texts = [f"# ユーザ\n{example['title'][i]}\n\n# アシスタント\n## 食材\n{example['name'][i]}\n## 作り方\n{example['position'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_zh_tw_recipes_sm(example: LazyBatch) -> list[str]:

    """
    Formats a batch of recipe examples into a specific text format.

    Args:
        example (`datasets.formatting.formatting.LazyBatch`):
            A batch of recipe examples containing 'title' and 'steps' fields.
    """

    output_texts = [f"# 標題\n{example['title'][i]}\n\n# 腳步\n{example['steps'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_data_recipes_instructor(example: LazyBatch) -> list[str]:

    """
    Formats a batch of recipe examples into a specific text format.

    Args:
        example (`datasets.formatting.formatting.LazyBatch`):
            A batch of recipe examples containing 'instruction' and 'input' fields.
    """

    output_texts = [f"# Instruction\n{example['instruction'][i]}\n\n# Input\n{example['input'][i]}\n\n# Output\n{example['output'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_thai_food(example: LazyBatch) -> list[str]:

    """
    Formats a batch of recipe examples into a specific text format.

    Args:
        example (`datasets.formatting.formatting.LazyBatch`):
            A batch of recipe examples containing 'name' and 'text' fields.
    """

    output_texts = [f"# ชื่อ\n{example['name'][i]}\n\n# ข้อความ\n{example['text'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_aya_telugu_food_recipes(example: LazyBatch) -> list[str]:

    """
    Formats a batch of recipe examples into a specific text format.

    Args:
        example (`datasets.formatting.formatting.LazyBatch`):
            A batch of recipe examples containing 'inputs' and 'target' fields.
    """

    output_texts = [f"# ఇన్పుట్\n{example['inputs'][i]}\n\n# లక్ష్యం\n{example['target'][i]}" for i in range(len(example))]
    return output_texts