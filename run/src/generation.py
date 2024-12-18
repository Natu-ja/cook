from argparse import Namespace
import pickle
import pandas as pd
from tqdm import tqdm
import torch
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

@torch.no_grad()
def generation_cookpad(args: Namespace, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel):

    """
    Using the test dataset, text generation is performed based on the specified model and tokenizer.

    Args:
        args (`argparse.Namespace`):
            Arguments for generation settings, model paths, output destinations, etc.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            Tokenizer.
        model (`transformers.PreTrainedModel`):
            Model.
    """

    test_dataset = pickle.load(open(file=args.output_dir + "/test_dataset.pkl", mode="rb"))

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
        exponential_decay_length_penalty=(args.exponential_decay_length_penalty_start_index, args.exponential_decay_length_penalty_decay_factor) if args.exponential_decay_length_penalty else None,
        token_healing=args.token_healing,
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
        max_matching_ngram_size=args.max_matching_ngram_size,
        dola_layers=args.dola_layers
    )

    output_lists = []

    if args.assistant_model is None:  
        for title in tqdm(test_dataset["title"]):
            input_text = f"# ユーザ\n## タイトル\n{title}\n\n# アシスタント\n"
            input_text = tokenizer(input_text, add_special_tokens=True, return_tensors="pt").to(model.device)
            output_text = model.generate(**input_text, generation_config=generation_config)
            output_list = tokenizer.batch_decode(output_text, skip_special_tokens=True)
            output_lists.append(output_list)
    
    else:

        from models import load_checkpoint
        assistant_model = load_checkpoint(args)[1]

        for title in tqdm(test_dataset["title"]):
            input_text = f"# ユーザ\n{title}\n\n# アシスタント\n"
            input_text = tokenizer(input_text, add_special_tokens=True, return_tensors="pt").to(model.device)
            output_text = model.generate(**input_text, generation_config=generation_config, assistant_model=assistant_model)
            output_list = tokenizer.batch_decode(output_text, skip_special_tokens=True)
            output_lists.append(output_list)

    output_lists = pd.DataFrame(output_lists)
    output_lists.to_csv(path_or_buf=args.output_dir+"/outputs.csv", header=False, index=False)