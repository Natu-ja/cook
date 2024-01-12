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


## データセット
- [クックパッドデータセット](https://www.nii.ac.jp/dsc/idr/cookpad/cookpad.html)
    - レシピデータ

```
mysql -u root -p -e "select * from recipes" cookpad > recipes.tsv
```


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
python main.py --tokenizer rinna/japanese-gpt2-xsmall --model rinna/japanese-gpt2-xsmall
```
または
```
bash run_train.sh
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