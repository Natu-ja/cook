# cook

## 使い方の例
一番基本的な学習方法
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b
```
```
python main.py rinna/japanese-gpt2-medium rinna/japanese-gpt2-medium
```

LoRA を適応させた時の学習方法
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b --target-modules embed_in query_key_value dense packed_input_proj out_proj embed_out
```

Instruction tuning をする時の学習方法
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b --instruction 以下の食材を使って作れる料理名を教えてください。：
```