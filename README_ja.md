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

| 大分類 | 中分類 | 小分類 | 論文 | 使用法 |
| :--: | :--: | :--: | :-- | :-- |
| 量子化 |||| `python cookpad.py --load-in-8bit` |
| Flash Attention || Flash Attention 2 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | `python cookpad.py --attn-implementation flash_attention_2 --torch-dtype float16` または `python cookpad.py --attn-implementation flash_attention_2 --torch-dtype bfloat16` |
| PEFT | Soft prompts | Prompt Tuning | The Power of Scale for Parameter-Efficient Prompt Tuning | `python cookpad.py --peft-type PROMPT_TUNING --prompt-tuning-init TEXT --prompt-tuning-init-text 料理のタイトルから料理の材料と手順を予測する。` |
| PEFT | Soft prompts | P-Tuning | GPT Understands, Too | `python cookpad.py --peft-type P_TUNING --encoder-hidden-size 768` |
| PEFT | Soft prompts | Prefix Tuning | Prefix-Tuning: Optimizing Continuous Prompts for Generation | `python cookpad.py --peft-type PREFIX_TUNING --encoder-hidden-size 768` |
| PEFT | Adapters | LoRA | LoRA: Low-Rank Adaptation of Large Language Models | `python cookpad.py --peft-type LORA --target-modules all-linear` |
| PEFT | Adapters | AdaLoRA | Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning | `python cookpad.py --peft-type ADALORA` |
| PEFT | Adapters | BOFT | Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization | `python cookpad.py --peft-type BOFT --target-modules all-linear` |
| PEFT | Adapters | Llama-Adapter | LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention | `python cookpad.py --peft-type ADAPTION_PROMPT` |
| PEFT || IA3 | Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning | `python cookpad.py --peft-type IA3 --target-modules all-linear --feedforward-modules all-linear` |
| PEFT | Adapters | LoHa | FedPara: Low-Rank Hadamard Product for Communication-Efficient Federated Learning | `python cookpad.py --peft-type LOHA --target-modules all-linear` |
| PEFT | Adapters | LoKr | Navigating Text-To-Image Customization:From LyCORIS Fine-Tuning to Model Evaluation | `python cookpad.py --peft-type LOKR --target-modules all-linear` |
| PEFT | Adapters | OFT | Controlling Text-to-Image Diffusion by Orthogonal Finetuning | `python cookpad.py --peft-type OFT --target-modules all-linear` |
| PEFT || Polytropon | Combining Modular Skills in Multitask Learning| `python cookpad.py --peft-type POLY --target-modules all-linear` |
| PEFT || Layernorm Tuning | Tuning LayerNorm in Attention: Towards Efficient Multi-Modal LLM Finetuning | `python cookpad.py --peft-type LN_TUNING --target-modules all-linear` |
| 生成戦略 || 貪欲法 || `python cookpad.py` |
| 生成戦略 || Multinomial Sampling || `python cookpad.py --do-sample` |
| 生成戦略 || Beam-Search Decoding || `python cookpad.py --num-beams 2` |
| 生成戦略 || Beam-Search Multinomial Sampling || `python cookpad.py --do-sample --num-beams 2` |
| 生成戦略 || Contrastive Search | A Contrastive Framework for Neural Text Generation | `python cookpad.py --penalty-alpha 0.5` |
| 生成戦略 || Diverse Beam-Search Decoding | Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models | `python cookpad.py --num-beams 2 --num-beam-groups 2` |
| 生成戦略 || Assisted Decoding || `python cookpad.py --prompt-lookup-num-tokens 2` |

### 実行する

```
bash main.sh
```