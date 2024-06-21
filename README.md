# Cooking

<h4 align="center">
    <p>
        <b>English</b> | 
        <a href='https://github.com/Natu-ja/cook/blob/main/README_ja.md'>日本語</a>
    </p>
</h4>

## Data
- [Cookpad dataset (Recipe data)](https://www.nii.ac.jp/dsc/idr/cookpad/)
---
- [AWeirdDev/zh-tw-recipes-sm](https://huggingface.co/datasets/AWeirdDev/zh-tw-recipes-sm)
    - dataset size
        - train data: $1,799$
    - language
        - Chinese
- [Erik/data_recipes_instructor](https://huggingface.co/datasets/Erik/data_recipes_instructor)
    - dataset size
        - train data: $20,000$
    - language
        - English
- [mertbozkurt/llama2-TR-recipe](https://huggingface.co/datasets/mertbozkurt/llama2-TR-recipe)
    - dataset size
        - train data: $10,504$
    - language
        - Turkish
- [SuryaKrishna02/aya-telugu-food-recipes](https://huggingface.co/datasets/SuryaKrishna02/aya-telugu-food-recipes)
    - dataset size
        - train data: $441$
    - language
        - Telugu

## Implemented
- Quantization
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
- Generation Strategy
    - Greedy Decoding
    - Multinomial Sampling
    - Beam-Search Decoding
    - Beam-Search Multinomial Sampling
    - Contrastive Search
    - Diverse Beam-Search Decoding
    - Assisted Decoding

## Train

```
python main.py \
    --load-best-model-at-end \
```