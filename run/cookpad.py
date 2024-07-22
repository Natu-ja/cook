import argparse
from argparse import Namespace
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets.arrow_dataset import Dataset
from trl import SFTConfig, SFTTrainer

from src.data_preprocessing import load_raw_dataset, formatting_func_cookpad
from src.models import load_checkpoint
from src.generation import generation

def run_training(args: Namespace, train_dataset: Dataset, eval_dataset: Dataset) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:

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

    if args.data_collator=="LanguageModeling":
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors="pt"
        )
    else:
        from trl import DataCollatorForCompletionOnlyLM
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=tokenizer.encode("# アシスタント\n", add_special_tokens=False),
            instruction_template=tokenizer.encode("# ユーザ\n", add_special_tokens=False),
            mlm=False,
            tokenizer=tokenizer
        )
    
    if args.peft_type is not None:

        from src.models import get_peft_config
        peft_config = get_peft_config(args)

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

def main(args: Namespace):

    train_dataset, eval_dataset, test_dataset = load_raw_dataset(args)
    tokenizer, model = run_training(args, train_dataset, eval_dataset)
    generation(args, tokenizer, model, test_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the Cookpad dataset and generate outputs.")

    parser.add_argument("--dataset", default="../data/cookpad_data.csv", type=str, help="https://www.nii.ac.jp/dsc/idr/cookpad/")
    parser.add_argument("--tokenizer", default="cyberagent/open-calm-7b", type=str, help="Tokenizer name or path.")
    parser.add_argument("--model", default="cyberagent/open-calm-7b", type=str, help="Model name or path.")
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
    parser.add_argument("--attn-implementation", type=str, choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--torch-dtype", default="auto", type=str, choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto", type=str, choices=["auto", "cpu", "cuda"])

    # SFT Config
    parser.add_argument("--output-dir", default="tmp_trainer/cookpad", type=str)
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
    parser.add_argument("--bias", default="none", type=str, choices=["none", "all", "lora_only", "boft_only"])
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