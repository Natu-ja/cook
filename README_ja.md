# cook

<h4 align="center">
    <p>
        <a href='https://github.com/Natu-ja/cook/'>English</a> | 
        <b>日本語</b>
    </p>
</h4>

## すること
- ストップワードの除去
    - 例. 塩, 水
- データ拡張
    - ノイズ
        - 削除
        - 挿入
        - 置換
        - 交換

## Google Colaboratory
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ad8CKAOHuK5dnqvufVwrDTmIGH7SDTEw?usp=sharing)

## データセット
- [クックパッドデータセット](https://www.nii.ac.jp/dsc/idr/cookpad/cookpad.html)
    - レシピデータ

## 環境
Python 3.11.1
```
pip install -r requirements.txt
```

## 使用例
### 基本
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b
```
```
python main.py rinna/japanese-gpt2-medium rinna/japanese-gpt2-medium
```
または
```
bash run_train.sh
```

### LoRA
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b --target-modules embed_in query_key_value dense packed_input_proj out_proj embed_out
```
または
```
bash run_train_lora.sh
```

### 指示チューニング
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b --system-message 以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。 --instruction 以下の食材を使って作れる料理名を教えてください。
```
または
```
bash run_train_instruction.sh
```

### 生成
```
python main.py novelai/nerdstash-tokenizer-v1 stabilityai/japanese-stablelm-instruct-alpha-7b --generation yes
```
または
```
bash run_train_generation.sh
```