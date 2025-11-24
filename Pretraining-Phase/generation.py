import torch
from gptv1.GPT import TinyGPT
from tokenizer import byteTokenizer
from config import load_config

cfg = load_config("config.yaml")

MODEL_CKPT_PATH = cfg.model.best_model
DATA_PATH = cfg.data.path


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = cfg.generation.max_new_token
TEMPERATURE = cfg.generation.temperature
TOP_K = cfg.generation.top_k

def load_best_model(ckpt_path, device):
    """Same loading logic as before to ensure config matches"""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint['config']

    model = TinyGPT(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        n_block=config['n_block'],
        n_head=config['n_head'],
        d_model=config['d_model'],
        dropout=0.0
    )

    state_dict = checkpoint['model_state_dict']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def stream_generate(prompt:str):
    """
    Generates text and prints it to stdout immediately as tokens are created.
    """

    try:
        #  Load the model state
        model = load_best_model(MODEL_CKPT_PATH, DEVICE)
        tokenizer = byteTokenizer(DATA_PATH)
        
        prompt_encoded = tokenizer.encode(prompt)
        print(f"Prompt: {prompt}", end="", flush=True)
        
        collected_tokens = prompt_encoded
        len_printed = len(tokenizer.decode(collected_tokens))
        
        print(f"\n--- Streaming Generation (Temp: {TEMPERATURE}) ---\n")

        # generate method with stream=True
        generator = model.generate(
            prompt=prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            tokenizer=tokenizer,
            stream=True
        )   

        for new_token_id in generator:
            collected_tokens.append(new_token_id)
            full_text = tokenizer.decode(collected_tokens)
            new_text = full_text[len_printed:]
            print(new_text, end="", flush=True)
            
            # Update counter
            len_printed += len(new_text)

        print("\n\n--- End of Generation ---")
        
    except Exception as e:
        print(f"An error occurred: {e}")