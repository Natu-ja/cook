# Cooking

<h4 align="center">
    <p>
        <b>English</b> | 
        <a href='https://github.com/Natu-ja/cook/blob/main/README_ja.md'>日本語</a>
    </p>
</h4>

<img src='image.webp' style="display: block; margin: auto; width: 100%;">

## Train

### Setup

```
pip install -r requirements.txt
```

### Data

| Dataset Name | Language | Train Dataset Size | Validation Dataset Size | Test Dataset Size | All Dataset Size | URL |
|:--:|:--:|:--:|:--:|:--:|:--:|:--|
| Cookpad dataset (Recipe data) | Japanese ||||| https://www.nii.ac.jp/dsc/idr/cookpad/ |
| zh-tw-recipes-sm | Chinese | $1,799$ ||| $1,799$ | https://huggingface.co/datasets/AWeirdDev/zh-tw-recipes-sm |
| data_recipes_instructor | English | $20,000$ ||| $20,000$ | https://huggingface.co/datasets/Erik/data_recipes_instructor |
| llama2-TR-recipe | Turkish | $10,504$ ||| $10,504$ | https://huggingface.co/datasets/mertbozkurt/llama2-TR-recipe |
| thai_food_v1.0 | Thai | $159$ ||| $159$ | https://huggingface.co/datasets/pythainlp/thai_food_v1.0 |
| aya-telugu-food-recipes | Telugu | $441$ ||| $441$ | https://huggingface.co/datasets/SuryaKrishna02/aya-telugu-food-recipes |

### Implemented

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

### Run

```
bash main.sh
```