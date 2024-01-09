from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from peft import AdaLoraConfig, get_peft_model, LoraConfig, PeftConfig, PeftModel

def load(args):
    try:
        config = PeftConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        print(f'Loaded tokenizer from {args.tokenizer}')

        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
        print(f'Loaded model from {args.model}, model size {model.num_parameters()}')

        model = PeftModel.from_pretrained(model, args.model)
        model.print_trainable_parameters()
        print(f'model size {model.num_parameters()}')
    
    except:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except:
            tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)
        print(f'Loaded tokenizer from {args.tokenizer}')

        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
        print(f'Loaded model from {args.model}, model size {model.num_parameters()}')

        if args.target_modules is not None:
            peft_config = get_lora_config(args) if args.peft_method == 'lora' else get_adalora_config(args)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            print(f'model size {model.num_parameters()}')
    
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) if '[PAD]' not in tokenizer.all_special_tokens else None

    return tokenizer, model

def get_lora_config(args):
    peft_config = LoraConfig(
        r=args.rank, 
        target_modules=args.target_modules, 
        lora_alpha=args.lora_alpha, 
        lora_dropout =args.lora_dropout, 
        fan_in_fan_out=args.fan_in_fan_out, 
        bias=args.peft_bias
    )
    return peft_config

def get_adalora_config(args):
    peft_config = AdaLoraConfig(
        r=args.rank, 
        target_modules=args.target_modules, 
        lora_alpha=args.lora_alpha, 
        lora_dropout =args.lora_dropout, 
        fan_in_fan_out=args.fan_in_fan_out, 
        bias=args.peft_bias,
        target_r=args.target_r,
        init_r=args.init_r,
        tinit=args.tinit,
        tfinal=args.tfinal,
        deltaT=args.deltaT,
        beta1=args.peft_beta1,
        beta2=args.peft_beta2,
        orth_reg_weight=args.orth_reg_weight
    )
    return peft_config