# cook

## Google Colaboratory
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ad8CKAOHuK5dnqvufVwrDTmIGH7SDTEw?usp=sharing)

## Datasets
- [COOKPAD Dataset](https://www.nii.ac.jp/dsc/idr/cookpad/cookpad.html)
    - Recipe Dataset

## Environments
Python 3.11.1
```
pip install -r requirements.txt
```

## Example of use
### Standard
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b
```
```
python main.py rinna/japanese-gpt2-medium rinna/japanese-gpt2-medium
```

### [LoRA](https://openreview.net/pdf?id=nZeVKeeFYf9)
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b --target-modules embed_in query_key_value dense packed_input_proj out_proj embed_out
```

### [Instruction tuning](https://openreview.net/pdf?id=gEZrGCozdqR)
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b --system-message 以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。 --instruction 以下の食材を使って作れる料理名を教えてください。
```

### Generation
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b --generation yes
```