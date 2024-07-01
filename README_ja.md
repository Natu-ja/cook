# Cooking

<h4 align="center">
    <p>
        <a href='https://github.com/Natu-ja/cook/blob/main/README.md'>English</a> | 
        <b>日本語</b>
    </p>
</h4>

<img src='image.webp' style="display: block; margin: auto; width: 100%;">

## 学習する

### セットアップ

```
pip install -r requirements.txt
```

### データ

|データセットの名前|言語|訓練データセットのサイズ|検証データセットのサイズ|評価データセットのサイズ|全てのデータセットのサイズ|URL|
|:--:|:--:|:--:|:--:|:--:|:--:|:--|
|クックパッドデータセット（レシピデータ）|日本語|||||https://www.nii.ac.jp/dsc/idr/cookpad/|
|AWeirdDev/zh-tw-recipes-sm|中国語|$1,799$|||$1,799$|https://huggingface.co/datasets/AWeirdDev/zh-tw-recipes-sm|
|Erik/data_recipes_instructor|英語|$20,000$|||$20,000$|https://huggingface.co/datasets/Erik/data_recipes_instructor|
|mertbozkurt/llama2-TR-recipe|トルコ語|$10,504$|||$10,504$|https://huggingface.co/datasets/mertbozkurt/llama2-TR-recipe|
|pythainlp/thai_food_v1.0|タイ語|$159$|||$159$|https://huggingface.co/datasets/pythainlp/thai_food_v1.0|
|SuryaKrishna02/aya-telugu-food-recipes|テルグ語|$441$|||$441$|https://huggingface.co/datasets/SuryaKrishna02/aya-telugu-food-recipes|

### 実装済み

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

### 実行する

```
bash main.sh
```