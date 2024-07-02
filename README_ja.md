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

| データセットの名前 | 言語 | 訓練データセットのサイズ | 検証データセットのサイズ | 評価データセットのサイズ | 全てのデータセットのサイズ | URL |
| :--: | :--: | :--: | :--: | :--: | :--: | :-- |
| クックパッドデータセット（レシピデータ） | 日本語 ||||| https://www.nii.ac.jp/dsc/idr/cookpad/ |
| zh-tw-recipes-sm | 中国語 | $1,799$ ||| $1,799$ | https://huggingface.co/datasets/AWeirdDev/zh-tw-recipes-sm |
| data_recipes_instructor | 英語 | $20,000$ ||| $20,000$ | https://huggingface.co/datasets/Erik/data_recipes_instructor |
| llama2-TR-recipe | トルコ語 | $10,504$ ||| $10,504$ | https://huggingface.co/datasets/mertbozkurt/llama2-TR-recipe |
| thai_food_v1.0 | タイ語 | $159$ ||| $159$ | https://huggingface.co/datasets/pythainlp/thai_food_v1.0 |
| aya-telugu-food-recipes | テルグ語 | $441$ ||| $441$ | https://huggingface.co/datasets/SuryaKrishna02/aya-telugu-food-recipes |

### 実装済み

| 大分類 | 中分類 | 小分類 | 論文 |
| :--: | :--: | :--: | :-- |
| 量子化 ||||
| Flash Attention || Flash Attention 2 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning |
| PEFT | Soft prompts | Prompt Tuning | The Power of Scale for Parameter-Efficient Prompt Tuning |
| PEFT | Soft prompts | P-Tuning | GPT Understands, Too |
| PEFT | Soft prompts | Prefix Tuning | Prefix-Tuning: Optimizing Continuous Prompts for Generation |
| PEFT | Adapters | LoRA | LoRA: Low-Rank Adaptation of Large Language Models |
| PEFT | Adapters | AdaLoRA | Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning |
| PEFT | Adapters | BOFT | Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization |
| PEFT | Adapters | Llama-Adapter | LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention |
| PEFT || IA3 | Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning |
| PEFT | Adapters | LoHa | FedPara: Low-Rank Hadamard Product for Communication-Efficient Federated Learning |
| PEFT | Adapters | LoKr | Navigating Text-To-Image Customization:From LyCORIS Fine-Tuning to Model Evaluation |
| PEFT | Adapters | OFT | Controlling Text-to-Image Diffusion by Orthogonal Finetuning |
| PEFT || Polytropon | Combining Modular Skills in Multitask Learning|
| PEFT || Layernorm Tuning | Tuning LayerNorm in Attention: Towards Efficient Multi-Modal LLM Finetuning |
| 生成戦略 || 貪欲法 ||
| 生成戦略 || Multinomial Sampling ||
| 生成戦略 || Beam-Search Decoding ||
| 生成戦略 || Beam-Search Multinomial Sampling ||
| 生成戦略 || Contrastive Search | A Contrastive Framework for Neural Text Generation |
| 生成戦略 || Diverse Beam-Search Decoding | Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models |
| 生成戦略 || Assisted Decoding ||

### 実行する

```
bash main.sh
```