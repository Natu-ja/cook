# cook

## 環境構築
Python 3.11.1
```
pip install -r requirements.txt
```

## 使い方の例
一番基本的な学習方法
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b
```
```
python main.py rinna/japanese-gpt2-medium rinna/japanese-gpt2-medium
```

[LoRA](https://openreview.net/pdf?id=nZeVKeeFYf9) を適応させた時の学習方法
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b --target-modules embed_in query_key_value dense packed_input_proj out_proj embed_out
```

[Instruction tuning](https://openreview.net/pdf?id=gEZrGCozdqR) をする時の学習方法
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b --system-message 以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。 --instruction 以下の食材を使って作れる料理名を教えてください。
```

学習後に，生成をする場合（デフォルトの場合，貪欲法 (Greedy search) を採用している．）
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b --generation yes
```
学習後に，生成する時に，Beam-Search Decoding をしたい場合
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b --generation yes --num-beams 2
```