# cook

<h4 align="center">
    <p>
        <a href='https://github.com/Natu-ja/cook/'>English</a> | 
        <b>日本語</b>
    </p>
</h4>

## すること
- テキスト前処理
    - データクリーニング
    - 単語の正規化
    - ストップワードの除去

## Google Colaboratory
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ad8CKAOHuK5dnqvufVwrDTmIGH7SDTEw?usp=sharing)

## データセット
- [クックパッドデータセット](https://www.nii.ac.jp/dsc/idr/cookpad/cookpad.html)
    - レシピデータ

## 環境
Python 3.11.1
```
pyenv install 3.11.1
pyenv local 3.11.1
```

```
pip install -r requirements.txt
```

## 使用例
### 基本
```
python main.py
```
```
python main.py  --tokenizer rinna/japanese-gpt2-medium --model rinna/japanese-gpt2-medium
```
または
```
bash run_train.sh
```

### LoRA
```
python main.py --target-modules q_proj v_proj
```
または
```
bash run_train_lora.sh
```

### 指示チューニング
```
python main.py --system-message 以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。 --instruction 以下の料理のレシピを教えてください。
```
または
```
bash run_train_instruction.sh
```

### 生成
```
python main.py --generation
```
または
```
bash run_train_generation.sh
```