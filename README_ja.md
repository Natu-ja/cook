# Cooking

<h4 align="center">
    <p>
        <a href='./README.md'>English</a> | 
        <b>日本語</b>
    </p>
</h4>

![日本の料理を作る少女](./fig/image.webp)

## 概要

Cookpadなどの料理データセットを用いて、料理のタイトルから材料や手順を生成するプログラムです。

## 学習する

### セットアップ

```bash
pip install -r requirements.txt
```

Flash Attention を使用したい場合は、[こちらのリンク](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features)を参照し、指示に従ってインストールしてください。

### クイックスタート

簡単にファインチューニングするには、[`examples` フォルダ](./examples/) にある Jupyter Notebook を使用することで行えます。

### データ

| データセットの名前 | 言語 | 訓練データセットのサイズ | 検証データセットのサイズ | 評価データセットのサイズ | 全てのデータセットのサイズ | URL | シード |
| :--: | :--: | :--: | :--: | :--: | :--: | :-- | :--: |
| クックパッドデータセット（レシピデータ） | 日本語 | $1,071,753$ | $267,939$ | $334,923$ | $1,674,615$ | https://www.nii.ac.jp/dsc/idr/cookpad/ | $42$ |
| data_recipes_instructor | 英語 | $20,000$ ||| $20,000$ | https://huggingface.co/datasets/Erik/data_recipes_instructor ||
| llama2-TR-recipe | トルコ語 | $10,504$ ||| $10,504$ | https://huggingface.co/datasets/mertbozkurt/llama2-TR-recipe ||
| Recipes_Greek | ギリシャ語 | $5,434$ ||| $5,434$ | https://huggingface.co/datasets/Depie/Recipes_Greek ||
| all-recipes-sm | 英語 | $2,000$ ||| $2,000$ | https://huggingface.co/datasets/AWeirdDev/all-recipes-sm ||
| zh-tw-recipes-sm | 中国語 | $1,799$ ||| $1,799$ | https://huggingface.co/datasets/AWeirdDev/zh-tw-recipes-sm ||
| all-recipes-xs | 英語 | $500$ ||| $500$ | https://huggingface.co/datasets/AWeirdDev/all-recipes-xs ||
| aya-telugu-food-recipes | テルグ語 | $441$ ||| $441$ | https://huggingface.co/datasets/SuryaKrishna02/aya-telugu-food-recipes ||
| thai_food_v1.0 | タイ語 | $159$ ||| $159$ | https://huggingface.co/datasets/pythainlp/thai_food_v1.0 ||

取得したクックパッドデータセットは、[`data` フォルダ](./data/)に保存してください。

### プロンプト

学習時のプロンプトを変更したい場合は、[`data_preprocessing.py`](./run/src/data_preprocessing.py) 内の [`formatting_func_.+` 関数](./run/src/data_preprocessing.py#L70-L159)を変更してください。以下の関数は、Cookpad 用のサンプルです。

```python:./run/src/data_preprocessing.py
def formatting_func_cookpad(example):
    output_texts = [f"# ユーザ\n{example['title'][i]}\n\n# アシスタント\n## 食材\n{example['name'][i]}\n## 作り方\n{example['position'][i]}" for i in range(len(example))]
    return output_texts
```

[`formatting_func_cookpad` 関数](./run/src/data_preprocessing.py#L70-L81) を適用したデータセットの例を示します．

```text
# ユーザ
豚の角煮

# アシスタント
## 食材
しょうが（お好みで）、ニンニク（お好みで）、ねぎ（１本）、豚肉（バラのブロック２パック）、砂糖（小さじ１から２くらい）、酒（たくさん（安い日本酒でいい））、醤油（適量（味見しながらね））、みりん（大さじ３くらい）
## 作り方
鍋に、水とたっぷりのお酒、ねぎの使わない葉の部分、しょうがの皮、にんにくを入れて、２，３時間煮込みます。その間、あくや浮いてきた脂を丁寧に取りましょう。煮込んだお肉を、いったん水で洗いましょう。落とし蓋をして１時間。食べるちょっと前にねぎを入れて、味がついたらたべましょう。写真のは、ちんげん菜を入れてみました。鍋に、豚肉をいれて、酒、砂糖、みりん、醤油、しょうが（薄切り）、にんにくで煮込みます。
```

### データコレーター

#### Completion Only LM

> [!WARNING]
> いくつかのトークナイザ（例：LLaMA 2）では、シーケンスを通常とは異なる方法でトークン化します。そのため、提供したコードでは学習がうまく進まない場合があります。

詳細は、[こちらのサイト](https://huggingface.co/docs/trl/sft_trainer#using-tokenids-directly-for-responsetemplate)をご参照ください。

> [!TIP]
> この問題を解決するには、次のようなコードを使用してください。

```python:./run/cookpad.py
from trl import DataCollatorForCompletionOnlyLM

data_collator = DataCollatorForCompletionOnlyLM(
    response_template=tokenizer.encode("\n# アシスタント\n", add_special_tokens=False)[2:],
    instruction_template=tokenizer.encode("# ユーザ\n", add_special_tokens=False),
    mlm=False,
    tokenizer=tokenizer
)
```

### モデル

このプログラムは、[Huggingface](https://huggingface.co/models) で公開されている Causal Language Model (CLM) を使用して動作します。CLM は、テキスト生成に広く使用されているモデルです。

### 実装済み

| 大分類 | 中分類 | 小分類 | 論文 | 使用法 |
| :--: | :--: | :--: | :-- | :-- |
| 量子化 || 8 bit || `python run/cookpad.py --load-in-8bit` |
| 量子化 || 4 bit || `python run/cookpad.py --load-in-4bit` |
| Flash Attention || Flash Attention 2 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | `python run/cookpad.py --attn-implementation flash_attention_2 --torch-dtype float16` または `python run/cookpad.py --attn-implementation flash_attention_2 --torch-dtype bfloat16` |
| PEFT | Soft prompts | Prompt Tuning | The Power of Scale for Parameter-Efficient Prompt Tuning | `python run/cookpad.py --peft-type PROMPT_TUNING --prompt-tuning-init TEXT --prompt-tuning-init-text 料理のタイトルから料理の材料と手順を予測する。` |
| PEFT | Soft prompts | P-Tuning | GPT Understands, Too | `python run/cookpad.py --peft-type P_TUNING --encoder-hidden-size 768` |
| PEFT | Soft prompts | Prefix Tuning | Prefix-Tuning: Optimizing Continuous Prompts for Generation | `python run/cookpad.py --peft-type PREFIX_TUNING --encoder-hidden-size 768` |
| PEFT | Adapters | LoRA | LoRA: Low-Rank Adaptation of Large Language Models | `python run/cookpad.py --peft-type LORA --target-modules all-linear` |
| PEFT | Adapters | AdaLoRA | Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning | `python run/cookpad.py --peft-type ADALORA` |
| PEFT | Adapters | BOFT | Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization | `python run/cookpad.py --peft-type BOFT --target-modules all-linear` |
| PEFT | Adapters | Llama-Adapter | LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention | `python run/cookpad.py --peft-type ADAPTION_PROMPT` |
| PEFT || IA3 | Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning | `python run/cookpad.py --peft-type IA3 --target-modules all-linear --feedforward-modules all-linear` |
| PEFT | Adapters | LoHa | FedPara: Low-Rank Hadamard Product for Communication-Efficient Federated Learning | `python run/cookpad.py --peft-type LOHA --target-modules all-linear` |
| PEFT | Adapters | LoKr | Navigating Text-To-Image Customization:From LyCORIS Fine-Tuning to Model Evaluation | `python run/cookpad.py --peft-type LOKR --target-modules all-linear` |
| PEFT | Adapters | OFT | Controlling Text-to-Image Diffusion by Orthogonal Finetuning | `python run/cookpad.py --peft-type OFT --target-modules all-linear` |
| PEFT || Polytropon | Combining Modular Skills in Multitask Learning| `python run/cookpad.py --peft-type POLY --target-modules all-linear` |
| PEFT || Layernorm Tuning | Tuning LayerNorm in Attention: Towards Efficient Multi-Modal LLM Finetuning | `python run/cookpad.py --peft-type LN_TUNING --target-modules all-linear` |
| PEFT|| FourierFT | Parameter-Efficient Fine-Tuning with Discrete Fourier Transform | `python run/cookpad.py --peft-type FOURIERFT --target-modules all-linear` |
| 生成戦略 || 貪欲法 || `python run/cookpad.py` |
| 生成戦略 || Multinomial Sampling || `python run/cookpad.py --do-sample` |
| 生成戦略 || Beam-Search Decoding || `python run/cookpad.py --num-beams 2` |
| 生成戦略 || Beam-Search Multinomial Sampling || `python run/cookpad.py --do-sample --num-beams 2` |
| 生成戦略 || Contrastive Search | A Contrastive Framework for Neural Text Generation | `python run/cookpad.py --penalty-alpha 0.5` |
| 生成戦略 || Diverse Beam-Search Decoding | Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models | `python run/cookpad.py --num-beams 2 --num-beam-groups 2` |
| 生成戦略 || Assisted Decoding || `python run/cookpad.py --prompt-lookup-num-tokens 2` |
| 生成戦略 || DoLa Decoding | DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models | `python run/cookpad.py --dola-layers low` |

### 実行する

```bash
bash main.sh
```

## 推論する

ファインチューニングしたモデルを使って推論を行うには、以下のコードを実行してください。コード内の `checkpoint` にはご自身のパスを、`title` には生成したい料理のタイトルを入力してください。

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