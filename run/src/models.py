from argparse import Namespace
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
from peft import PeftConfig

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

def get_peft_config(args: Namespace) -> PeftConfig:
    
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
    
    return peft_config