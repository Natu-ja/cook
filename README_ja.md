# Cooking

<h4 align="center">
    <p>
        <a href='https://github.com/Natu-ja/cook/blob/main/README.md'>English</a> | 
        <b>日本語</b>
    </p>
</h4>

## データ
- [クックパッドデータセット](https://www.nii.ac.jp/dsc/idr/cookpad/)
    - レシピデータ

## 学習する

```
python main.py \
    --dataset dataset.tsv \
    --tokenizer rinna/japanese-gpt2-xsmall \
    --model rinna/japanese-gpt2-xsmall \
    --output-dir output_dir \
    --evaluation-strategy steps \
    --num-train-epochs 1.0 \
    --load-best-model-at-end \
    --group-by-length
```