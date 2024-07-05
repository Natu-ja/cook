# Cooking

<h4 align="center">
    <p>
        <b>English</b> | 
        <a href='https://github.com/Natu-ja/cook/blob/main/README_ja.md'>日本語</a>
    </p>
</h4>

![a girl who cooks Japanese cuisine](image.webp)

## Overview

This program uses cooking datasets, such as those from Cookpad, to generate ingredients and instructions from a recipe title.

## Train

### Setup

```bash
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

### Prompt

If you want to change the prompts used during training, please modify the `formatting_func_.+` function in [`data_preprocessing.py`](https://github.com/Natu-ja/cook/blob/main/run/src/data_preprocessing.py). The following function is a sample for Cookpad.

```python
def formatting_func_cookpad(example):
    output_texts = [f"# ユーザ\n{example['title'][i]}\n\n# アシスタント\n## 食材\n{example['name'][i]}\n## 作り方\n{example['position'][i]}" for i in range(len(example))]
    return output_texts
```

An example of a dataset with the [`formatting_func_cookpad`](https://github.com/Natu-ja/cook/blob/main/run/src/data_preprocessing.py#L30C-L32C) function applied is shown below.

```
# ユーザ
豚の角煮

# アシスタント
## 食材
しょうが（お好みで）、ニンニク（お好みで）、ねぎ（１本）、豚肉（バラのブロック２パック）、砂糖（小さじ１から２くらい）、酒（たくさん（安い日本酒でいい））、醤油（適量（味見しながらね））、みりん（大さじ３くらい）
## 作り方
鍋に、水とたっぷりのお酒、ねぎの使わない葉の部分、しょうがの皮、にんにくを入れて、２，３時間煮込みます。その間、あくや浮いてきた脂を丁寧に取りましょう。煮込んだお肉を、いったん水で洗いましょう。落とし蓋をして１時間。食べるちょっと前にねぎを入れて、味がついたらたべましょう。写真のは、ちんげん菜を入れてみました。鍋に、豚肉をいれて、酒、砂糖、みりん、醤油、しょうが（薄切り）、にんにくで煮込みます。
```

### Implemented

| Major Category | Subcategory | Sub-subcategory | Paper | Usage |
| :--: | :--: | :--: | :-- | :-- |
| Quantization || 8 bit || `python cookpad.py --load-in-8bit` |
| Quantization || 4 bit || `python cookpad.py --load-in-4bit` |
| Flash Attention || Flash Attention 2 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | `python cookpad.py --attn-implementation flash_attention_2 --torch-dtype float16` or `python cookpad.py --attn-implementation flash_attention_2 --torch-dtype bfloat16` |
| PEFT | Soft prompts | Prompt Tuning | The Power of Scale for Parameter-Efficient Prompt Tuning | `python cookpad.py --peft-type PROMPT_TUNING --prompt-tuning-init TEXT --prompt-tuning-init-text 料理のタイトルから料理の材料と手順を予測する。` |
| PEFT | Soft prompts | P-Tuning | GPT Understands, Too | `python cookpad.py --peft-type P_TUNING --encoder-hidden-size 768` |
| PEFT | Soft prompts | Prefix Tuning | Prefix-Tuning: Optimizing Continuous Prompts for Generation | `python cookpad.py --peft-type PREFIX_TUNING --encoder-hidden-size 768` |
| PEFT | Adapters | LoRA | LoRA: Low-Rank Adaptation of Large Language Models | `python cookpad.py --peft-type LORA --target-modules all-linear` |
| PEFT | Adapters | AdaLoRA | Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning | `python cookpad.py --peft-type ADALORA` |
| PEFT | Adapters | BOFT | Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization | `python cookpad.py --peft-type BOFT --target-modules all-linear` |
| PEFT | Adapters | Llama-Adapter | LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention | `python cookpad.py --peft-type ADAPTION_PROMPT` |
| PEFT || IA3 | Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning | `python cookpad.py --peft-type IA3 --target-modules all-linear --feedforward-modules all-linear` |
| PEFT | Adapters | LoHa | FedPara: Low-Rank Hadamard Product for Communication-Efficient Federated Learning | `python cookpad.py --peft-type LOHA --target-modules all-linear` |
| PEFT | Adapters | LoKr | Navigating Text-To-Image Customization:From LyCORIS Fine-Tuning to Model Evaluation | `python cookpad.py --target-modules all-linear` |
| PEFT | Adapters | OFT | Controlling Text-to-Image Diffusion by Orthogonal Finetuning | `python cookpad.py --peft-type OFT --target-modules all-linear` |
| PEFT || Polytropon | Combining Modular Skills in Multitask Learning | `python cookpad.py --peft-type POLY --target-modules all-linear` |
| PEFT || Layernorm Tuning | Tuning LayerNorm in Attention: Towards Efficient Multi-Modal LLM Finetuning | `python cookpad.py --peft-type LN_TUNING --target-modules all-linear` |
| Generation Strategy || Greedy Decoding || `python cookpad.py` |
| Generation Strategy || Multinomial Sampling || `python cookpad.py --do-sample` |
| Generation Strategy || Beam-Search Decoding || `python cookpad.py --num-beams 2` |
| Generation Strategy || Beam-Search Multinomial Sampling || `python cookpad.py --do-sample --num-beams 2` |
| Generation Strategy || Contrastive Search | A Contrastive Framework for Neural Text Generation | `python cookpad.py --penalty-alpha 0.5` |
| Generation Strategy || Diverse Beam-Search Decoding | Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models | `python cookpad.py --num-beams 2 --num-beam-groups 2` |
| Generation Strategy || Assisted Decoding || `python cookpad.py --prompt-lookup-num-tokens 2` |

### Run

```bash
bash main.sh
```