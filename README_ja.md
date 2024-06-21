# Cooking

<h4 align="center">
    <p>
        <a href='https://github.com/Natu-ja/cook/blob/main/README.md'>English</a> | 
        <b>日本語</b>
    </p>
</h4>

<img src='image.webp' style="display: block; margin: auto;">

## データ
- [クックパッドデータセット（レシピデータ）](https://www.nii.ac.jp/dsc/idr/cookpad/)
---
- [AWeirdDev/zh-tw-recipes-sm](https://huggingface.co/datasets/AWeirdDev/zh-tw-recipes-sm)
    - データセットのサイズ
        - 訓練データ: $1,799$
    - 言語
        - 中国語
- [Erik/data_recipes_instructor](https://huggingface.co/datasets/Erik/data_recipes_instructor)
    - データセットのサイズ
        - 訓練データ: $20,000$
    - 言語
        - 英語
- [mertbozkurt/llama2-TR-recipe](https://huggingface.co/datasets/mertbozkurt/llama2-TR-recipe)
    - データセットのサイズ
        - 訓練データ: $10,504$
    - 言語
        - トルコ語
- [SuryaKrishna02/aya-telugu-food-recipes](https://huggingface.co/datasets/SuryaKrishna02/aya-telugu-food-recipes)
    - データセットのサイズ
        - 訓練データ: $441$
    - 言語
        - テルグ語

## 実装済み
- 量子化
- Flash Attention
- PEFT
    - Prompt Tuning
    - P-Tuning
    - Prefix Tuning
    - LoRA
    - AdaLoRA
    - BOFT
    - Llama-Adapter
    - IA3
    - LoHa
    - LoKr
    - OFT
    - Polytropon
    - Layernorm Tuning
- 生成戦略
    - 貪欲法
    - Multinomial Sampling
    - Beam-Search Decoding
    - Beam-Search Multinomial Sampling
    - Contrastive Search
    - Diverse Beam-Search Decoding
    - Assisted Decoding

## 学習する

```
python main.py \
    --eval-strategy steps \
    --load-best-model-at-end \
    --group-by-length \
    --auto-find-batch-size \
    --max-new-tokens 512
```