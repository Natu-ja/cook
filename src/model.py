from transformers import AutoModelForCausalLM, AutoTokenizer

def load(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    print(f'Loaded model from {args.model}, model size {model.num_parameters()}')

    return tokenizer, model