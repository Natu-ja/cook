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
| :--: | :--: | :--: | :--: | :--: | :--: | :-- |
| Cookpad dataset (Recipe data) | Japanese ||||| https://www.nii.ac.jp/dsc/idr/cookpad/ |
| zh-tw-recipes-sm | Chinese | $1,799$ ||| $1,799$ | https://huggingface.co/datasets/AWeirdDev/zh-tw-recipes-sm |
| data_recipes_instructor | English | $20,000$ ||| $20,000$ | https://huggingface.co/datasets/Erik/data_recipes_instructor |
| llama2-TR-recipe | Turkish | $10,504$ ||| $10,504$ | https://huggingface.co/datasets/mertbozkurt/llama2-TR-recipe |
| thai_food_v1.0 | Thai | $159$ ||| $159$ | https://huggingface.co/datasets/pythainlp/thai_food_v1.0 |
| aya-telugu-food-recipes | Telugu | $441$ ||| $441$ | https://huggingface.co/datasets/SuryaKrishna02/aya-telugu-food-recipes |

### Implemented

| Major Category | Subcategory | Sub-subcategory | Paper |
| :--: | :--: | :--: | :-- |
| Quantization ||||
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
| PEFT || Polytropon | Combining Modular Skills in Multitask Learning |
| PEFT || Layernorm Tuning | Tuning LayerNorm in Attention: Towards Efficient Multi-Modal LLM Finetuning |
| Generation Strategy || Greedy Decoding ||
| Generation Strategy || Multinomial Sampling ||
| Generation Strategy || Beam-Search Decoding ||
| Generation Strategy || Beam-Search Multinomial Sampling ||
| Generation Strategy || Contrastive Search | A Contrastive Framework for Neural Text Generation |
| Generation Strategy || Diverse Beam-Search Decoding | Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models |
| Generation Strategy || Assisted Decoding ||

### Run

```
bash main.sh
```