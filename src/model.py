from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from peft import get_peft_model, LoraConfig, PeftConfig, PeftModel

def load(args):
    try:
        config = PeftConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        print(f'Loaded tokenizer from {args.tokenizer}!!')

        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        print(f'Loaded model from {args.model}, model size {model.num_parameters()}!!')

        model = PeftModel.from_pretrained(model, args.model)
        model.print_trainable_parameters()
        print(f'model size {model.num_parameters()}!!')
    
    except:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except:
            tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)
        print(f'Loaded tokenizer from {args.tokenizer}!!')

        model = AutoModelForCausalLM.from_pretrained(args.model)
        print(f'Loaded model from {args.model}, model size {model.num_parameters()}!!')

        if args.target_modules != None:
            peft_config = LoraConfig(
                r=args.rank, 
                target_modules=args.target_modules, 
                lora_alpha=args.lora_alpha, 
                lora_dropout =args.lora_dropout, 
                bias=args.lora_bias
            )

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            print(f'model size {model.num_parameters()}!!')
    
    return tokenizer, model