# Cooking

<h4 align="center">
    <p>
        <b>English</b> | 
        <a href='https://github.com/Natu-ja/cook/blob/main/README_ja.md'>日本語</a>
    </p>
</h4>

## Data
- [Cookpad dataset](https://www.nii.ac.jp/dsc/idr/cookpad/)
    - Recipe data

## Train

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