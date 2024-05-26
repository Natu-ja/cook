import argparse
from tqdm.contrib import tzip
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

def load_dataset(args):
    dataset = pd.read_csv(args.dataset, sep="\t")
    dataset = Dataset.from_pandas(dataset)

    dataset = dataset.train_test_split(test_size=0.25, shuffle=True, seed=args.seed)
    tv_dataset, test_dataset = dataset["train"], dataset["test"]
    tv_dataset = tv_dataset.train_test_split(test_size=0.25, shuffle=False, seed=args.seed)
    train_dataset, eval_dataset = tv_dataset["train"], tv_dataset["test"]
    
    print(f"Train size : validation size : test size = {len(train_dataset)} : {len(eval_dataset)} : {len(test_dataset)}.")

    return train_dataset, eval_dataset, test_dataset

def load_checkpoint(args):

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.tokenizer,
    )

    if args.torch_dtype=="float16":
        args.torch_dtype = torch.float16
    elif args.torch_dtype=="bfloat16":
        args.torch_dtype = torch.bfloat16
    elif args.torch_dtype=="float32":
        args.torch_dtype = torch.float32

    if args.load_in_8bit or args.load_in_4bit:

        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit
        )

        if args.attn_implementation is not None:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model,
                attn_implementation=args.attn_implementation,
                torch_dtype=args.torch_dtype,
                device_map="auto",
                quantization_config=quantization_config
            )

        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model,
                torch_dtype=args.torch_dtype,
                device_map="auto",
                quantization_config=quantization_config
            )

    else:

        if args.attn_implementation is not None:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model,
                attn_implementation=args.attn_implementation,
                torch_dtype=args.torch_dtype,
                device_map="auto"
            )

        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model,
                torch_dtype=args.torch_dtype,
                device_map="auto"
            )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print(f"Tokenizer and model loaded from {args.tokenizer} and {args.model}.")

    return tokenizer, model

def formatting_func(example):
    output_texts = [f"# ユーザ\n{example[i]['title']}\n\n# アシスタント\n## 食材\n{example[i]['name']}\n## 作り方\n{example[i]['position']}" for i in range(len(example))]
    return output_texts

def run_training(args, train_dataset, eval_dataset):

    tokenizer, model = load_checkpoint(args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=1,
        seed=args.seed,
        data_seed=args.data_seed,
        dataloader_num_workers=args.dataloader_num_workers,
        load_best_model_at_end=args.load_best_model_at_end,
        optim=args.optim,
        group_by_length=args.group_by_length,
        report_to=args.report_to,
        neftune_noise_alpha=args.neftune_noise_alpha,
        optim_target_modules=args.optim_target_modules,
    )

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer.encode("# アシスタント\n", add_special_tokens=False),
        instruction_template=tokenizer.encode("# ユーザ\n", add_special_tokens=False),
        mlm=False
    )

    if args.peft_type is not None:

        if args.init_lora_weights is None or args.init_lora_weights=="true":
            args.init_lora_weights = True
        elif args.init_lora_weights=="false":
            args.init_lora_weights = False

        if args.peft_type=="PROMPT_TUNING":
            from peft import PromptTuningConfig
            peft_config = PromptTuningConfig(
                peft_type=args.peft_type,
                task_type="CAUSAL_LM",
                inference_mode=False,
                num_virtual_tokens=args.num_virtual_tokens,
                token_dim=args.token_dim,
                num_transformer_submodules=args.num_transformer_submodules,
                num_attention_heads=args.num_attention_heads,
                num_layers=args.num_layers,
                prompt_tuning_init=args.prompt_tuning_init,
                prompt_tuning_init_text=args.prompt_tuning_init_text,
                tokenizer_name_or_path=args.tokenizer
            )
        elif args.peft_type=="P_TUNING":
            from peft import PromptEncoderConfig
            peft_config = PromptEncoderConfig(
                peft_type=args.peft_type,
                task_type="CAUSAL_LM",
                inference_mode=False,
                num_virtual_tokens=args.num_virtual_tokens,
                token_dim=args.token_dim,
                num_transformer_submodules=args.num_transformer_submodules,
                num_attention_heads=args.num_attention_heads,
                num_layers=args.num_layers,
                encoder_reparameterization_type=args.encoder_reparameterization_type,
                encoder_hidden_size=args.encoder_hidden_size,
                encoder_num_layers=args.encoder_num_layers,
                encoder_dropout=args.encoder_dropout
            )
        elif args.peft_type=="PREFIX_TUNING":
            from peft import PrefixTuningConfig
            peft_config = PrefixTuningConfig(
                peft_type=args.peft_type,
                task_type="CAUSAL_LM",
                inference_mode=False,
                num_virtual_tokens=args.num_virtual_tokens,
                token_dim=args.token_dim,
                num_transformer_submodules=args.num_transformer_submodules,
                num_attention_heads=args.num_attention_heads,
                num_layers=args.num_layers,
                encoder_hidden_size=args.encoder_hidden_size,
                prefix_projection=args.prefix_projection
            )
        elif args.peft_type=="LORA":
            from peft import LoraConfig
            peft_config = LoraConfig(
                peft_type=args.peft_type,
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=args.r,
                target_modules=args.target_modules,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                fan_in_fan_out=args.fan_in_fan_out,
                bias=args.bias,
                use_rslora=args.use_rslora,
                init_lora_weights=args.init_lora_weights,
                use_dora=args.use_dora
            )
        elif args.peft_type=="ADALORA":
            from peft import AdaLoraConfig
            peft_config = AdaLoraConfig(
                peft_type=args.peft_type,
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=args.r,
                target_modules=args.target_modules,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                fan_in_fan_out=args.fan_in_fan_out,
                bias=args.bias,
                use_rslora=args.use_rslora,
                init_lora_weights=args.init_lora_weights,
                use_dora=args.use_dora,
                target_r=args.target_r,
                init_r=args.init_r,
                tinit=args.tinit,
                tfinal=args.tfinal,
                deltaT=args.deltaT,
                beta1=args.beta1,
                beta2=args.beta2,
                orth_reg_weight=args.orth_reg_weight
            )
        elif args.peft_type=="LOHA":
            from peft import LoHaConfig
            peft_config = LoHaConfig(
                peft_type=args.peft_type,
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=args.r,
                alpha=args.alpha,
                rank_dropout=args.rank_dropout,
                module_dropout=args.module_dropout,
                use_effective_conv2d=args.use_effective_conv2d,
                target_modules=args.target_modules,
                init_weights=args.init_weights
            )
        elif args.peft_type=="LOKR":
            from peft import LoKrConfig
            peft_config = LoKrConfig(
                peft_type=args.peft_type,
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=args.r,
                alpha=args.alpha,
                rank_dropout=args.rank_dropout,
                module_dropout=args.module_dropout,
                use_effective_conv2d=args.use_effective_conv2d,
                decompose_both=args.decompose_both,
                decompose_factor=args.decompose_factor,
                target_modules=args.target_modules,
                init_weights=args.init_weights
            )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            formatting_func=formatting_func,
            max_seq_length=args.max_seq_length,
            dataset_batch_size=args.dataset_batch_size
        )
    
    else:

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            formatting_func=formatting_func,
            max_seq_length=args.max_seq_length,
            dataset_batch_size=args.dataset_batch_size
        )

    print("Training started.")
    trainer.train()

    return tokenizer, model

def generation(args, tokenizer, model, test_dataset):

    if args.num_beams==1 and not args.do_sample:
        print("Text generation strategy is greedy decoding.")
    if args.num_beams==1 and args.do_sample:
        print("Text generation strategy is multinomial sampling.")
    if args.num_beams>1 and not args.do_sample:
        print("Text generation strategy is beam-search decoding.")
    if args.num_beams>1 and args.do_sample:
        print("Text generation strategy is beam-search multinomial sampling.")
    if args.penalty_alpha>0 and args.top_k>1:
        print("Text generation strategy is contrastive search.")
    if args.num_beams>1 and args.num_beam_groups>1:
        print("Text generation strategy is diverse beam-search decoding.")
    if args.assistant_model is not None or args.prompt_lookup_num_tokens is not None:
        print("Text generation strategy is assisted decoding.")

    generation_config = {
        "max_length" : args.max_length,
        "max_new_tokens" : args.max_new_tokens,
        "min_length" : args.min_length,
        "min_new_tokens" : args.min_new_tokens,
        "early_stopping" : args.early_stopping,
        "do_sample" : args.do_sample,
        "num_beams" : args.num_beams,
        "num_beam_groups" : args.num_beam_groups,
        "penalty_alpha" : args.penalty_alpha,
        "use_cache" : args.use_cache,
        "temperature" : args.temperature,
        "top_k" : args.top_k,
        "top_p" : args.top_p,
        "min_p" : args.min_p,
        "typical_p": args.typical_p,
        "epsilon_cutoff" : args.epsilon_cutoff,
        "eta_cutoff" : args.eta_cutoff,
        "diversity_penalty" : args.diversity_penalty,
        "repetition_penalty" : args.repetition_penalty,
        "encoder_repetition_penalty" : args.encoder_repetition_penalty,
        "length_penalty" : args.length_penalty,
        "no_repeat_ngram_size" : args.no_repeat_ngram_size,
        "renormalize_logits" : args.renormalize_logits,
        "guidance_scale" : args.guidance_scale,
        "num_return_sequences" : args.num_return_sequences,
        "output_attentions" : args.output_attentions,
        "pad_token_id" : tokenizer.pad_token_id,
        "bos_token_id" : tokenizer.bos_token_id,
        "eos_token_id" : tokenizer.eos_token_id,
        "prompt_lookup_num_tokens" : args.prompt_lookup_num_tokens,
        "max_matching_ngram_size" : args.max_matching_ngram_size
    }

    assistant_model = load_checkpoint(args)[1] if args.assistant_model is not None else None

    output_lists = []
    for title, name, position in tzip(test_dataset["title"]), test_dataset["name"], test_dataset["position"]:
        input_text = f"# ユーザ\n{title}\n\n# アシスタント\n## 食材\n{name}\n## 作り方\n{position}"
        input_text = tokenizer(input_text, add_special_tokens=True, return_tensors="pt").to(model.device)
        output_text = model.generate(
            **input_text,
            **generation_config,
            assistant_model=assistant_model
        )
        output_list = [tokenizer.decode(output_text[i], skip_special_tokens=True) for i in range(len(output_text))]
        output_lists.append(output_list)

    output_lists = pd.DataFrame(output_lists)
    output_lists.to_csv(args.dir+"/results.csv", header=False, index=False)
        

def main(args):
    train_dataset, eval_dataset, test_dataset = load_dataset(args)
    tokenizer, model = run_training(args, train_dataset, eval_dataset)
    generation(args, tokenizer, model, test_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--model", type=str)
    
    # Bits And Bytes Config
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")

    # From Pretrained
    parser.add_argument("--torch-dtype", default="auto", type=str, choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", type=str, choices=["eager", "sdpa", "flash_attention_2"])

    # Training Arguments
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--evaluation-strategy", default="no", type=str, choices=["no", "steps", "epoch"])
    parser.add_argument("--per-device-train-batch-size", default=8, type=int)
    parser.add_argument("--per-device-eval-batch-size", default=8, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--learning-rate", default=5e-5, type=float)
    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--adam-beta1", default=0.9, type=float)
    parser.add_argument("--adam-beta2", default=0.999, type=float)
    parser.add_argument("--adam-epsilon", default=1e-8, type=float)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--num-train-epochs", default=3.0, type=float)
    parser.add_argument("--lr-scheduler-type", default="linear", type=str, choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup-ratio", default=0.0, type=float)
    parser.add_argument("--logging-strategy", default="steps", type=str, choices=["no", "steps", "epoch"])
    parser.add_argument("--logging-steps", default=500, type=int)
    parser.add_argument("--save-strategy", default="steps", type=str, choices=["no", "steps", "epoch"])
    parser.add_argument("--save-steps", default=500, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--dataloader-num-workers", default=0, type=int)
    parser.add_argument("--load-best-model-at-end", action="store_true")
    parser.add_argument("--optim", default="adamw_torch", type=str, choices=["adamw_torch", "adamw_hf", "adamw_torch", "adamw_torch_fused", "adamw_apex_fused", "adamw_anyprecision", "adafactor", "adamw_bnb_8bit", "galore_adamw", "galore_adamw_8bit", "galore_adafactor"])
    parser.add_argument("--group-by-length", action="store_true")
    parser.add_argument("--report-to", nargs="*", default="all", type=str, choices=["azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "flyte", "mlflow", "neptune", "tensorboard", "wandb"])
    parser.add_argument("--neftune-noise-alpha", type=float)
    parser.add_argument("--optim-target-modules", nargs="*", type=str)

    # Peft Config
    parser.add_argument("--peft-type", type=str, choices=["PROMPT_TUNING", "P_TUNING", "PREFIX_TUNING", "LORA", "ADALORA", "LOHA", "LOKR"])
    ## Lora Config, Adalora Config, Loha Config and Lokr Config
    parser.add_argument("--r", default=8, type=int)
    parser.add_argument("--target-modules", nargs="*", type=str)
    ## Prompt Tuning Config, Prompt Encoder Config and Prefix Tuning Config
    parser.add_argument("--num-virtual-tokens", type=int)
    parser.add_argument("--token-dim", type=int)
    parser.add_argument("--num-transformer-submodules", type=int)
    parser.add_argument("--num-attention-heads", type=int)
    parser.add_argument("--num-layers", type=int)
    ## Prompt Encoder Config and Prefix Tuning Config
    parser.add_argument("--encoder-hidden-size", type=int)
    ## Lora Config and Adalora Config
    parser.add_argument("--lora-alpha", default=8, type=int)
    parser.add_argument("--lora-dropout", default=0.0, type=float)
    parser.add_argument("--fan-in-fan-out", action="store_true")
    parser.add_argument("--bias", type=str, choices=["none", "all", "lora_only"])
    parser.add_argument("--use-rslora", action="store_true")
    parser.add_argument("--init-lora-weights", type=str, choices=["true", "false", "gaussian", "pissa", "pissa_niter_[number of iters]", "loftq"])
    parser.add_argument("--use-dora", action="store_true")
    ## Prompt Tuning Config
    parser.add_argument("--prompt-tuning-init", default="RANDOM", type=str, choices=["RANDOM", "TEXT"])
    parser.add_argument("--prompt-tuning-init-text", type=str)
    ## Prompt Encoder Config
    parser.add_argument("--encoder-reparameterization-type", default="MLP", type=str, choices=["MLP", "LSTM"])
    parser.add_argument("--encoder-num-layers", default=2, type=int)
    parser.add_argument("--encoder-dropout", default=0.0, type=float)
    ## Prefix Tuning Config
    parser.add_argument("--prefix-projection", action="store_true")
    ## Adalora Config
    parser.add_argument("--target-r", default=8, type=int)
    parser.add_argument("--init-r", default=12, type=int)
    parser.add_argument("--tinit", default=0, type=int)
    parser.add_argument("--tfinal", default=0, type=int)
    parser.add_argument("--deltaT", default=1, type=int)
    parser.add_argument("--beta1", default=0.85, type=float)
    parser.add_argument("--beta2", default=0.85, type=float)
    parser.add_argument("--orth-reg-weight", default=0.5, type=float)
    ## Loha Config and Lokr Config
    parser.add_argument("--alpha", default=8, type=int)
    parser.add_argument("--rank-dropout", default=0.0, type=float)
    parser.add_argument("--module-dropout", default=0.0, type=float)
    parser.add_argument("--use-effective-conv2d", action="store_true")
    parser.add_argument("--init-weights", action="store_false")
    ## Lokr Config
    parser.add_argument("--decompose-both", action="store_true")
    parser.add_argument("--decompose-factor", default=-1, type=int)

    # SFT Trainer
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--dataset-batch-size", type=int)

    # Generate
    parser.add_argument("--assistant-model", type=str)

    # Generation Config
    parser.add_argument("--max-length", default=20, type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--min-length", default=0, type=int)
    parser.add_argument("--min-new-tokens", type=int)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--num-beams", default=1, type=int)
    parser.add_argument("--num-beam-groups", default=1, type=int)
    parser.add_argument("--penalty-alpha", type=float)
    parser.add_argument("--use-cache", action="store_false")
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top-k", default=50, type=int)
    parser.add_argument("--top-p", default=1.0, type=float)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--typical-p", default=1.0, type=float)
    parser.add_argument("--epsilon-cutoff", default=0.0, type=float)
    parser.add_argument("--eta-cutoff", default=0.0, type=float)
    parser.add_argument("--diversity-penalty", default=0.0, type=float)
    parser.add_argument("--repetition-penalty", default=1.0, type=float)
    parser.add_argument("--encoder-repetition-penalty", default=1.0, type=float)
    parser.add_argument("--length-penalty", default=1.0, type=float)
    parser.add_argument("--no-repeat-ngram-size", default=0, type=int)
    parser.add_argument("--renormalize-logits", action="store_true")
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--num-return-sequences", default=1, type=int)
    parser.add_argument("--output-attentions", action="store_true")
    parser.add_argument("--prompt-lookup-num-tokens", type=int)
    parser.add_argument("--max-matching-ngram-size", type=int)

    args = parser.parse_args()

    main(args)