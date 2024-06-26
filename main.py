import os
import argparse
from argparse import Namespace
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
from datasets.arrow_dataset import Dataset
from datasets.formatting.formatting import LazyBatch
from trl import SFTConfig, SFTTrainer

def load_raw_dataset(args: Namespace) -> tuple[Dataset, Dataset, Dataset] | tuple[Dataset, None, None]:

    if not os.path.exists(path=args.dataset):

        from datasets import load_dataset

        train_dataset = load_dataset(path=args.dataset, split="train")

        return train_dataset, None, None
    
    else:

        import pandas as pd

        dataset = pd.read_csv(filepath_or_buffer=args.dataset, sep="\t")
        dataset = Dataset.from_pandas(df=dataset)

        dataset = dataset.train_test_split(test_size=0.25, shuffle=True, seed=args.seed)
        tv_dataset, test_dataset = dataset["train"], dataset["test"]
        tv_dataset = tv_dataset.train_test_split(test_size=0.25, shuffle=False, seed=args.seed)
        train_dataset, eval_dataset = tv_dataset["train"], tv_dataset["test"]

        return train_dataset, eval_dataset, test_dataset

def load_checkpoint(args: Namespace) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.tokenizer,
    )

    if args.torch_dtype=="float16":
        args.torch_dtype = torch.float16
    elif args.torch_dtype=="bfloat16":
        args.torch_dtype = torch.bfloat16
    elif args.torch_dtype=="float32":
        args.torch_dtype = torch.float32
    
    if args.bnb_4bit_compute_dtype=="float32":
        args.bnb_4bit_compute_dtype = torch.float32
    elif args.bnb_4bit_compute_dtype=="bfloat16":
        args.bnb_4bit_compute_dtype = torch.bfloat16

    if args.load_in_8bit or args.load_in_4bit:

        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            llm_int8_threshold=args.llm_int8_threshold,
            llm_int8_skip_modules=args.llm_int8_skip_modules,
            llm_int8_has_fp16_weight=args.llm_int8_has_fp16_weight,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_quant_storage=args.bnb_4bit_quant_storage
        )

        if args.attn_implementation is not None:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model,
                attn_implementation=args.attn_implementation,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
                quantization_config=quantization_config
            )

        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
                quantization_config=quantization_config
            )

    else:

        if args.attn_implementation is not None:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model,
                attn_implementation=args.attn_implementation,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
            )

        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
            )

    if tokenizer.pad_token_id is None and model.config.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return tokenizer, model

def formatting_func_zh_tw_recipes_sm(example: LazyBatch) -> list[str]:
    output_texts = [f"# 標題\n{example['title'][i]}\n\n# 腳步\n{example['steps'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_data_recipes_instructor(example: LazyBatch) -> list[str]:
    output_texts = [f"# Instruction\n{example['instruction'][i]}\n\n# Input\n{example['input'][i]}\n\n# Output\n{example['output'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_thai_food(example: LazyBatch) -> list[str]:
    output_texts = [f"# ชื่อ\n{example['name'][i]}\n\n# ข้อความ\n{example['text'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_aya_telugu_food_recipes(example: LazyBatch) -> list[str]:
    output_texts = [f"# ఇన్పుట్\n{example['inputs'][i]}\n\n# లక్ష్యం\n{example['target'][i]}" for i in range(len(example))]
    return output_texts

def formatting_func_cookpad(example: LazyBatch) -> list[str]:
    output_texts = [f"# ユーザ\n{example['title'][i]}\n\n# アシスタント\n## 食材\n{example['name'][i]}\n## 作り方\n{example['position'][i]}" for i in range(len(example))]
    return output_texts

def run_training(args: Namespace, train_dataset: Dataset, eval_dataset: Dataset | None) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:

    tokenizer, model = load_checkpoint(args)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        eval_strategy=args.eval_strategy,
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
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_dir=args.logging_dir,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=1,
        seed=args.seed,
        data_seed=args.data_seed,
        fp16_opt_level=args.fp16_opt_level,
        half_precision_backend=args.half_precision_backend,
        dataloader_num_workers=args.dataloader_num_workers,
        run_name=args.run_name,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        optim=args.optim,
        group_by_length=args.group_by_length,
        report_to=args.report_to,
        auto_find_batch_size=args.auto_find_batch_size,
        split_batches=args.split_batches,
        include_tokens_per_second=args.include_tokens_per_second,
        include_num_input_tokens_seen=args.include_num_input_tokens_seen,
        neftune_noise_alpha=args.neftune_noise_alpha,
        optim_target_modules=args.optim_target_modules,
        packing=args.packing,
        max_seq_length=min(tokenizer.model_max_length, args.max_seq_length),
        dataset_num_proc=args.dataset_num_proc,
        dataset_batch_size=args.dataset_batch_size,
        num_of_sequences=args.num_of_sequences,
        chars_per_token=args.chars_per_token
    )

    if args.data_collator=="CompletionOnlyLM":
        from trl import DataCollatorForCompletionOnlyLM
        if args.dataset=="AWeirdDev/zh-tw-recipes-sm":
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=tokenizer.encode("# 腳步歩\n", add_special_tokens=False),
                instruction_template=tokenizer.encode("# 標題\n", add_special_tokens=False),
                mlm=False,
                tokenizer=tokenizer
            )
        elif args.dataset=="Erik/data_recipes_instructor":
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=tokenizer.encode("# Output\n", add_special_tokens=False),
                instruction_template=tokenizer.encode("# Instruction\n", add_special_tokens=False),
                mlm=False,
                tokenizer=tokenizer
            )
        elif args.dataset=="mertbozkurt/llama2-TR-recipe":
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=tokenizer.encode("[/INST] ", add_special_tokens=False),
                instruction_template=tokenizer.encode("[INST] ", add_special_tokens=False),
                mlm=False,
                tokenizer=tokenizer
            )
        elif args.dataset=="pythainlp/thai_food_v1.0":
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=tokenizer.encode("# ข้อความ\n", add_special_tokens=False),
                instruction_template=tokenizer.encode("# ชื่อ\n", add_special_tokens=False),
                mlm=False,
                tokenizer=tokenizer
            )
        elif args.dataset=="SuryaKrishna02/aya-telugu-food-recipes":
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=tokenizer.encode("# లక్ష్యం\n", add_special_tokens=False),
                instruction_template=tokenizer.encode("# ఇన్పుట్\n", add_special_tokens=False),
                mlm=False,
                tokenizer=tokenizer
            )
        else:
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=tokenizer.encode("# アシスタント\n", add_special_tokens=False),
                instruction_template=tokenizer.encode("# ユーザ\n", add_special_tokens=False),
                mlm=False,
                tokenizer=tokenizer
            )
    else:
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors="pt"
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
        elif args.peft_type=="BOFT":
            from peft import BOFTConfig
            peft_config = BOFTConfig(
                peft_type=args.peft_type,
                task_type="CAUSAL_LM",
                inference_mode=False,
                boft_block_size=args.boft_block_size,
                boft_block_num=args.boft_block_num,
                boft_n_butterfly_factor=args.boft_n_butterfly_factor,
                target_modules=args.target_modules,
                boft_dropout=args.boft_dropout,
                fan_in_fan_out=args.fan_in_fan_out,
                bias=args.bias,
                init_weights=args.init_weights
            )
        elif args.peft_type=="ADAPTION_PROMPT":
            from peft import AdaptionPromptConfig
            peft_config = AdaptionPromptConfig(
                peft_type=args.peft_type,
                task_type="CAUSAL_LM",
                inference_mode=False,
                target_modules=args.target_modules,
                adapter_len=args.adapter_len,
                adapter_layers=args.adapter_layers
            )
        elif args.peft_type=="IA3":
            from peft import IA3Config
            peft_config = IA3Config(
                peft_type=args.peft_type,
                task_type="CAUSAL_LM",
                inference_mode=False,
                target_modules=args.target_modules,
                feedforward_modules=args.feedforward_modules,
                fan_in_fan_out=args.fan_in_fan_out,
                init_weights=args.init_weights
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
        elif args.peft_type=="OFT":
            from peft import OFTConfig
            peft_config = OFTConfig(
                peft_type=args.peft_type,
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=args.r,
                module_dropout=args.module_dropout,
                target_modules=args.target_modules,
                init_weights=args.init_weights,
                coft=args.coft,
                eps=args.eps,
                block_share=args.block_share
            )
        elif args.peft_type=="POLY":
            from peft import PolyConfig
            peft_config = PolyConfig(
                peft_type=args.peft_type,
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=args.r,
                target_modules=args.target_modules,
                init_weights=args.init_weights,
                n_tasks=args.n_tasks,
                n_skills=args.n_skills,
                n_splits=args.n_splits
            )
        elif args.peft_type=="LN_TUNING":
            from peft import LNTuningConfig
            peft_config = LNTuningConfig(
                peft_type=args.peft_type,
                task_type="CAUSAL_LM",
                inference_mode=False,
                target_modules=args.target_modules
            )

        if args.dataset=="AWeirdDev/zh-tw-recipes-sm":
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                data_collator=data_collator,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                peft_config=peft_config,
                formatting_func=formatting_func_zh_tw_recipes_sm,
                infinite=args.infinite
            )
        elif args.dataset=="Erik/data_recipes_instructor":
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                data_collator=data_collator,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                peft_config=peft_config,
                formatting_func=formatting_func_data_recipes_instructor,
                infinite=args.infinite
            )
        elif args.dataset=="mertbozkurt/llama2-TR-recipe":
            sft_config.dataset_text_field="text"
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                data_collator=data_collator,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                peft_config=peft_config,
                infinite=args.infinite
            )
        elif args.dataset=="pythainlp/thai_food_v1.0":
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                data_collator=data_collator,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                peft_config=peft_config,
                formatting_func=formatting_func_thai_food,
                infinite=args.infinite
            )
        elif args.dataset=="SuryaKrishna02/aya-telugu-food-recipes":
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                data_collator=data_collator,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                peft_config=peft_config,
                formatting_func=formatting_func_aya_telugu_food_recipes,
                infinite=args.infinite
            )
        else:
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                peft_config=peft_config,
                formatting_func=formatting_func_cookpad,
                infinite=args.infinite
            )
    
    else:
        if args.dataset=="AWeirdDev/zh-tw-recipes-sm":
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                data_collator=data_collator,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                formatting_func=formatting_func_zh_tw_recipes_sm,
                infinite=args.infinite
            )
        elif args.dataset=="Erik/data_recipes_instructor":
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                data_collator=data_collator,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                formatting_func=formatting_func_data_recipes_instructor,
                infinite=args.infinite
            )
        elif args.dataset=="mertbozkurt/llama2-TR-recipe":
            sft_config.dataset_text_field="text"
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                data_collator=data_collator,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                infinite=args.infinite
            )
        elif args.dataset=="pythainlp/thai_food_v1.0":
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                data_collator=data_collator,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                formatting_func=formatting_func_thai_food,
                infinite=args.infinite
            )
        elif args.dataset=="SuryaKrishna02/aya-telugu-food-recipes":
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                data_collator=data_collator,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                formatting_func=formatting_func_aya_telugu_food_recipes,
                infinite=args.infinite
            )
        else:
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                formatting_func=formatting_func_cookpad,
                infinite=args.infinite
            )

    trainer.train()

    return tokenizer, model

def generation(args: Namespace, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, test_dataset: Dataset) -> None:

    import pandas as pd
    from tqdm import tqdm
    from transformers import GenerationConfig

    generation_config = GenerationConfig(
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        min_length=args.min_length,
        min_new_tokens=args.min_new_tokens,
        early_stopping=args.early_stopping,
        max_time=args.max_time,
        stop_strings=args.stop_strings,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        num_beam_groups=args.num_beam_groups,
        penalty_alpha=args.penalty_alpha,
        use_cache=args.use_cache,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        typical_p=args.typical_p,
        epsilon_cutoff=args.epsilon_cutoff,
        eta_cutoff=args.eta_cutoff,
        diversity_penalty=args.diversity_penalty,
        repetition_penalty=args.repetition_penalty,
        encoder_repetition_penalty=args.encoder_repetition_penalty,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        renormalize_logits=args.renormalize_logits,
        forced_bos_token_id= model.config.forced_bos_token_id,
        forced_eos_token_id=model.config.forced_eos_token_id,
        remove_invalid_values=model.config.remove_invalid_values if args.remove_invalid_values else False,
        guidance_scale=args.guidance_scale,
        low_memory=args.low_memory,
        num_return_sequences=args.num_return_sequences,
        output_attentions=args.output_attentions,
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        num_assistant_tokens=args.num_assistant_tokens,
        num_assistant_tokens_schedule=args.num_assistant_tokens_schedule,
        prompt_lookup_num_tokens=args.prompt_lookup_num_tokens,
        max_matching_ngram_size=args.max_matching_ngram_size
    )

    output_lists = []
    with torch.no_grad():

        if args.assistant_model is None:  
            for title in tqdm(test_dataset["title"]):
                input_text = f"# ユーザ\n{title}\n\n# アシスタント\n"
                input_text = tokenizer(input_text, add_special_tokens=True, return_tensors="pt").to(model.device)
                output_text = model.generate(**input_text, generation_config=generation_config)
                output_list = [tokenizer.decode(output_text[i], skip_special_tokens=True) for i in range(len(output_text))]
                output_lists.append(output_list)
        
        else:
            assistant_model = load_checkpoint(args)[1]

            for title in tqdm(test_dataset["title"]):
                input_text = f"# ユーザ\n{title}\n\n# アシスタント\n"
                input_text = tokenizer(input_text, add_special_tokens=True, return_tensors="pt").to(model.device)
                output_text = model.generate(**input_text, generation_config=generation_config, assistant_model=assistant_model)
                output_list = [tokenizer.decode(output_text[i], skip_special_tokens=True) for i in range(len(output_text))]
                output_lists.append(output_list)

    output_lists = pd.DataFrame(output_lists)
    output_lists.to_csv(path_or_buf=args.output_dir+"/outputs.csv", header=False, index=False)

def main(args: Namespace) -> None:

    train_dataset, eval_dataset, test_dataset = load_raw_dataset(args)
    tokenizer, model = run_training(args, train_dataset, eval_dataset)
    
    if test_dataset is not None:
        generation(args, tokenizer, model, test_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the Cookpad dataset and generate outputs.")

    parser.add_argument("--dataset", default="./cookpad_data.tsv", type=str, help="Path to the dataset file.", choices=["./cookpad_data.tsv", "AWeirdDev/zh-tw-recipes-sm", "Erik/data_recipes_instructor", "mertbozkurt/llama2-TR-recipe", "pythainlp/thai_food_v1.0", "SuryaKrishna02/aya-telugu-food-recipes"])
    parser.add_argument("--tokenizer", default="ai-forever/mGPT", type=str, help="Tokenizer name or path.")
    parser.add_argument("--model", default="ai-forever/mGPT", type=str, help="Model name or path.")
    parser.add_argument("--data-collator", type=str, default="LanguageModeling", choices=["LanguageModeling", "CompletionOnlyLM"], help="Data collator type.")
    
    # Bits And Bytes Config
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--llm-int8-threshold", default=6.0, type=float)
    parser.add_argument("--llm-int8-skip-modules", nargs="*", type=str)
    parser.add_argument("--llm-int8-has-fp16-weight", action="store_true")
    parser.add_argument("--bnb-4bit-compute-dtype", default="float32", type=str,choices=["float32", "bfloat16"])
    parser.add_argument("--bnb-4bit-quant-type", default="fp4", type=str, choices=["fp4", "nf4"])
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument("--bnb-4bit-quant-storage", default="uint8", type=str, choices=["uint8"])

    # From Pretrained
    parser.add_argument("--torch-dtype", default="auto", type=str, choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", type=str, choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--device-map", default="auto", type=str, choices=["auto", "cpu", "cuda"])

    # SFT Config
    parser.add_argument("--output-dir", default="tmp_trainer", type=str)
    parser.add_argument("--eval-strategy", default="no", type=str, choices=["no", "steps", "epoch"])
    parser.add_argument("--per-device-train-batch-size", default=8, type=int)
    parser.add_argument("--per-device-eval-batch-size", default=8, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--learning-rate", default=5e-5, type=float)
    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--adam-beta1", default=0.9, type=float)
    parser.add_argument("--adam-beta2", default=0.999,type=float)
    parser.add_argument("--adam-epsilon", default=1e-8, type=float)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--num-train-epochs", default=3.0, type=float)
    parser.add_argument("--max-steps", default=-1, type=int)
    parser.add_argument("--lr-scheduler-type", default="linear", type=str, choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup-ratio", default=0.0, type=float)
    parser.add_argument("--logging-dir", type=str)
    parser.add_argument("--logging-strategy", default="steps", type=str, choices=["no", "steps", "epoch"])
    parser.add_argument("--logging-steps", default=500, type=int)
    parser.add_argument("--save-strategy", default="steps", type=str, choices=["no", "steps", "epoch"])
    parser.add_argument("--save-steps", default=500, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--fp16-opt-level", default="O1", type=str, choices=["O0", "O1", "O2", "O3"])
    parser.add_argument("--half-precision-backend", default="auto", type=str, choices=["auto", "apex", "cpu_amp"])
    parser.add_argument("--dataloader-num-workers", default=0, type=int)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--load-best-model-at-end", action="store_true")
    parser.add_argument("--metric-for-best-model", type=str)
    parser.add_argument("--optim", default="adamw_torch", type=str, choices=["adamw_torch", "adamw_hf", "adamw_torch", "adamw_torch_fused", "adamw_apex_fused", "adamw_anyprecision", "adafactor"])
    parser.add_argument("--group-by-length", action="store_true")
    parser.add_argument("--report-to", nargs="*", default="all", type=str, choices=["azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "flyte", "mlflow", "neptune", "tensorboard", "wandb"])
    parser.add_argument("--auto-find-batch-size", action="store_true")
    parser.add_argument("--split-batches", action="store_true")
    parser.add_argument("--include-tokens-per-second", action="store_true")
    parser.add_argument("--include-num-input-tokens-seen", action="store_true")
    parser.add_argument("--neftune-noise-alpha", type=float)
    parser.add_argument("--optim-target-modules", nargs="*", type=str)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--max-seq-length", default=1024, type=int)
    parser.add_argument("--dataset-num-proc", type=int)
    parser.add_argument("--dataset-batch-size", type=int)
    parser.add_argument("--num-of-sequences", default=1024, type=int)
    parser.add_argument("--chars-per-token", default=3.6, type=float)

    # SFT Trainer
    parser.add_argument("--infinite", action="store_true")

    # Peft Config
    parser.add_argument("--peft-type", type=str, choices=["PROMPT_TUNING", "P_TUNING", "PREFIX_TUNING", "LORA", "ADALORA", "BOFT", "ADAPTION_PROMPT", "IA3", "LOHA", "LOKR", "OFT", "POLY", "LN_TUNING"])
    ## Lora Config, Adalora Config, Boft Config, Adaption Prompt Config, IA3 Config, Loha Config, Lokr Config, OFT Config, Poly Config and LN Tuning Config
    parser.add_argument("--target-modules", nargs="*", type=str)
    ## Lora Config, Adalora Config, Loha Config, Lokr Config, OFT Config and Poly Config
    parser.add_argument("--r", default=8, type=int)
    ## Loha Config, Lokr Config, OFT Config, Boft Config and Poly Config
    parser.add_argument("--init-weights", action="store_false")
    ## Loha Config, Lokr Config, OFT Config and Boft Config
    parser.add_argument("--module-dropout", default=0.0, type=float)
    ## Lora Config, Adalora Config, Boft Config and IA3 Config
    parser.add_argument("--fan-in-fan-out", action="store_true")
    ## Prompt Tuning Config, Prompt Encoder Config and Prefix Tuning Config
    parser.add_argument("--num-virtual-tokens", type=int)
    parser.add_argument("--token-dim", type=int)
    parser.add_argument("--num-transformer-submodules", type=int)
    parser.add_argument("--num-attention-heads", type=int)
    parser.add_argument("--num-layers", type=int)
    ## Lora Config, Adalora Config and Boft Config
    parser.add_argument("--bias", type=str, choices=["none", "all", "lora_only", "boft_only"])
    ## Prompt Encoder Config and Prefix Tuning Config
    parser.add_argument("--encoder-hidden-size", type=int)
    ## Lora Config and Adalora Config
    parser.add_argument("--lora-alpha", default=8, type=int)
    parser.add_argument("--lora-dropout", default=0.0, type=float)
    parser.add_argument("--use-rslora", action="store_true")
    parser.add_argument("--init-lora-weights", type=str, choices=["true", "false", "gaussian", "pissa", "pissa_niter_[number of iters]", "loftq"])
    parser.add_argument("--use-dora", action="store_true")
    ## Loha Config and Lokr Config
    parser.add_argument("--alpha", default=8, type=int)
    parser.add_argument("--rank-dropout", default=0.0, type=float)
    parser.add_argument("--use-effective-conv2d", action="store_true")
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
    ## Boft Config
    parser.add_argument("--boft-block-size", default=4, type=int)
    parser.add_argument("--boft-block-num", default=0, type=int)
    parser.add_argument("--boft-n-butterfly-factor", default=1, type=int)
    parser.add_argument("--boft-dropout", default=0.0, type=float)
    ## Adaption Prompt Config
    parser.add_argument("--adapter-len", type=int)
    parser.add_argument("--adapter-layers", type=int)
    ## IA3 Config
    parser.add_argument("--feedforward-modules", nargs="*", type=str)
    parser.add_argument("--init-ia3-weights", action="store_false")
    ## Lokr Config
    parser.add_argument("--decompose-both", action="store_true")
    parser.add_argument("--decompose-factor", default=-1, type=int)
    ## OFT Config
    parser.add_argument("--coft", action="store_true")
    parser.add_argument("--eps", default=6e-05, type=float)
    parser.add_argument("--block-share", action="store_true")
    ## Poly Config
    parser.add_argument("--poly-type", default="poly", type=str, choices=["poly"])
    parser.add_argument("--n-tasks", default=1, type=int)
    parser.add_argument("--n-skills", default=4, type=int)
    parser.add_argument("--n-splits", default=1, type=int)

    # Generate
    parser.add_argument("--assistant-model", type=str)

    # Generation Config
    parser.add_argument("--max-length", default=20, type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--min-length", default=0, type=int)
    parser.add_argument("--min-new-tokens", type=int)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--max-time", type=float)
    parser.add_argument("--stop-strings", nargs="*", type=str)
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
    parser.add_argument("--remove-invalid-values", action="store_true")
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--num-return-sequences", default=1, type=int)
    parser.add_argument("--output-attentions", action="store_true")
    parser.add_argument("--num-assistant-tokens", default=5, type=int)
    parser.add_argument("--num-assistant-tokens-schedule", default="heuristic", type=str, choices=["heuristic", "heuristic_transient", "constant"])
    parser.add_argument("--prompt-lookup-num-tokens", type=int)
    parser.add_argument("--max-matching-ngram-size", type=int)

    args = parser.parse_args()

    main(args)