# Cooking

<h4 align="center">
    <p>
        <b>English</b> | 
        <a href='./README_ja.md'>日本語</a>
    </p>
</h4>

![a girl who cooks Japanese cuisine](./fig/image.webp)

## Overview

This program uses cooking datasets, such as those from Cookpad, to generate ingredients and instructions from a recipe title.

## Train

### Setup

```bash
pip install -r requirements.txt
```

If you would like to use Flash Attention, please refer to [this link](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) and follow the instructions for installation.

### Quick Start

For easy fine-tuning, you can use the Jupyter Notebook provided in the [`examples` folder](./examples/).

### Data

| Dataset Name | Language | Train Dataset Size | Validation Dataset Size | Test Dataset Size | All Dataset Size | URL | Seed |
| :--: | :--: | :--: | :--: | :--: | :--: | :-- | :--: |
| Cookpad dataset (Recipe data) | Japanese | $1,071,753$ | $267,939$ | $334,923$ | $1,674,615$ | https://www.nii.ac.jp/dsc/idr/cookpad/ | $42$ |
| data_recipes_instructor | English | $20,000$ ||| $20,000$ | https://huggingface.co/datasets/Erik/data_recipes_instructor ||
| llama2-TR-recipe | Turkish | $10,504$ ||| $10,504$ | https://huggingface.co/datasets/mertbozkurt/llama2-TR-recipe ||
| Recipes_Greek | Greek | $5,434$ ||| $5,434$ | https://huggingface.co/datasets/Depie/Recipes_Greek ||
| all-recipes-sm | English | $2,000$ ||| $2,000$ | https://huggingface.co/datasets/AWeirdDev/all-recipes-sm ||
| zh-tw-recipes-sm | Chinese | $1,799$ ||| $1,799$ | https://huggingface.co/datasets/AWeirdDev/zh-tw-recipes-sm ||
| all-recipes-xs | English | $500$ ||| $500$ | https://huggingface.co/datasets/AWeirdDev/all-recipes-xs ||
| aya-telugu-food-recipes | Telugu | $441$ ||| $441$ | https://huggingface.co/datasets/SuryaKrishna02/aya-telugu-food-recipes ||
| thai_food_v1.0 | Thai | $159$ ||| $159$ | https://huggingface.co/datasets/pythainlp/thai_food_v1.0 ||

Please save the obtained Cookpad dataset in the [`data` folder](./data/).

### Prompt

If you want to change the prompts used during training, please modify the [`formatting_func_.+` function](./run/src/data_preprocessing.py#L82-L171) in [`data_preprocessing.py`](./run/src/data_preprocessing.py). The following function is a sample for Cookpad.

```python:./run/src/data_preprocessing.py
def formatting_func_cookpad(example):
    output_texts = [f"# ユーザ\n## タイトル\n{example['title'][i]}\n\n# アシスタント\n## 食材\n{example['ingredients'][i]}\n## 作り方\n{example['steps'][i]}" for i in range(len(example))]
    return output_texts
```

An example of a dataset with the [`formatting_func_cookpad` function](./run/src/data_preprocessing.py#L82-L93) applied is shown below.

```text
# ユーザ
豚の角煮

# アシスタント
## 食材
しょうが（お好みで）、ニンニク（お好みで）、ねぎ（１本）、豚肉（バラのブロック２パック）、砂糖（小さじ１から２くらい）、酒（たくさん（安い日本酒でいい））、醤油（適量（味見しながらね））、みりん（大さじ３くらい）
## 作り方
鍋に、水とたっぷりのお酒、ねぎの使わない葉の部分、しょうがの皮、にんにくを入れて、２，３時間煮込みます。その間、あくや浮いてきた脂を丁寧に取りましょう。煮込んだお肉を、いったん水で洗いましょう。落とし蓋をして１時間。食べるちょっと前にねぎを入れて、味がついたらたべましょう。写真のは、ちんげん菜を入れてみました。鍋に、豚肉をいれて、酒、砂糖、みりん、醤油、しょうが（薄切り）、にんにくで煮込みます。
```

### Data Collator

#### Completion Only LM

> [!WARNING]
> Some tokenizers (e.g., LLaMA 2) tokenize sequences in ways that differ from the usual methods. As a result, the provided code may not work correctly for training.

For more details, please refer to [this website](https://huggingface.co/docs/trl/sft_trainer#using-tokenids-directly-for-responsetemplate).

> [!TIP]
> To resolve this issue, please use the following code.

```python:./run/cookpad.py
from trl import DataCollatorForCompletionOnlyLM

data_collator = DataCollatorForCompletionOnlyLM(
    response_template=tokenizer.encode("\n# アシスタント\n", add_special_tokens=False)[2:],
    instruction_template=tokenizer.encode("# ユーザ\n", add_special_tokens=False),
    mlm=False,
    tokenizer=tokenizer
)
```

### Model

This program operates using the Causal Language Model (CLM) available from [Hugging Face](https://huggingface.co/models). The CLM is widely used for text generation.

### Implemented

| Major Category | Subcategory | Sub-subcategory | Paper | Usage |
| :--: | :--: | :--: | :-- | :-- |
| Quantization || 8 bit || `python run/cookpad.py --load-in-8bit` |
| Quantization || 4 bit || `python run/cookpad.py --load-in-4bit` |
| Flash Attention || Flash Attention 2 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | `python run/cookpad.py --attn-implementation flash_attention_2 --torch-dtype float16` or `python run/cookpad.py --attn-implementation flash_attention_2 --torch-dtype bfloat16` |
| PEFT | Soft prompts | Prompt Tuning | The Power of Scale for Parameter-Efficient Prompt Tuning | `python run/cookpad.py --peft-type PROMPT_TUNING --prompt-tuning-init TEXT --prompt-tuning-init-text 料理のタイトルから料理の材料と手順を予測する。` |
| PEFT | Soft prompts | P-Tuning | GPT Understands, Too | `python run/cookpad.py --peft-type P_TUNING --encoder-hidden-size 768` |
| PEFT | Soft prompts | Prefix Tuning | Prefix-Tuning: Optimizing Continuous Prompts for Generation | `python run/cookpad.py --peft-type PREFIX_TUNING --encoder-hidden-size 768` |
| PEFT | Adapters | LoRA | LoRA: Low-Rank Adaptation of Large Language Models | `python run/cookpad.py --peft-type LORA --target-modules all-linear` |
| PEFT | Adapters | AdaLoRA | Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning | `python run/cookpad.py --peft-type ADALORA` |
| PEFT | Adapters | BOFT | Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization | `python run/cookpad.py --peft-type BOFT --target-modules all-linear` |
| PEFT | Adapters | Llama-Adapter | LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention | `python run/cookpad.py --peft-type ADAPTION_PROMPT` |
| PEFT || IA3 | Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning | `python run/cookpad.py --peft-type IA3 --target-modules all-linear --feedforward-modules all-linear` |
| PEFT | Adapters | LoHa | FedPara: Low-Rank Hadamard Product for Communication-Efficient Federated Learning | `python run/cookpad.py --peft-type LOHA --target-modules all-linear` |
| PEFT | Adapters | LoKr | Navigating Text-To-Image Customization:From LyCORIS Fine-Tuning to Model Evaluation | `python run/cookpad.py --target-modules all-linear` |
| PEFT | Adapters | OFT | Controlling Text-to-Image Diffusion by Orthogonal Finetuning | `python run/cookpad.py --peft-type OFT --target-modules all-linear` |
| PEFT || Polytropon | Combining Modular Skills in Multitask Learning | `python run/cookpad.py --peft-type POLY --target-modules all-linear` |
| PEFT || Layernorm Tuning | Tuning LayerNorm in Attention: Towards Efficient Multi-Modal LLM Finetuning | `python run/cookpad.py --peft-type LN_TUNING --target-modules all-linear` |
| PEFT|| FourierFT | Parameter-Efficient Fine-Tuning with Discrete Fourier Transform | `python run/cookpad.py --peft-type FOURIERFT --target-modules all-linear` |
| Generation Strategy || Greedy Decoding || `python run/cookpad.py` |
| Generation Strategy || Multinomial Sampling || `python run/cookpad.py --do-sample` |
| Generation Strategy || Beam-Search Decoding || `python run/cookpad.py --num-beams 2` |
| Generation Strategy || Beam-Search Multinomial Sampling || `python run/cookpad.py --do-sample --num-beams 2` |
| Generation Strategy || Contrastive Search | A Contrastive Framework for Neural Text Generation | `python run/cookpad.py --penalty-alpha 0.5` |
| Generation Strategy || Diverse Beam-Search Decoding | Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models | `python run/cookpad.py --num-beams 2 --num-beam-groups 2` |
| Generation Strategy || Assisted Decoding || `python run/cookpad.py --prompt-lookup-num-tokens 2` |
| Generation Strategy || DoLa Decoding | DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models | `python run/cookpad.py --dola-layers low` |

### Run

```bash
bash main.sh
```

## Infer

To perform inference using the fine-tuned model, execute the following code. Replace `checkpoint` in the code with your path, and `title` with the title of the dish you want to generate.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

checkpoint = "YOUR_PATH/checkpoint-X"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")

generation_config = GenerationConfig()

with torch.no_grad():
    title = "料理のタイトル"
    input_text = f"# ユーザ\n{title}\n\n# アシスタント\n"
    input_text = tokenizer(input_text, add_special_tokens=True, return_tensors="pt").to(model.device)
    output_text = model.generate(**input_text, generation_config=generation_config)
    output_list = [tokenizer.decode(output_text[i], skip_special_tokens=True) for i in range(len(output_text))]
```