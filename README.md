# cook

## 使い方の例
一番基本的な実行の方法
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b
```

LoRA を適応させた時の実行
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b --target-modules embed_in query_key_value dense packed_input_proj out_proj embed_out
```

別のモデルを使った実行
```
python main.py rinna/japanese-gpt2-medium rinna/japanese-gpt2-medium
```