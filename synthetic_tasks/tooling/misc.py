import os
import torch
import sentencepiece as spm
import wandb
from omegaconf import OmegaConf

def get_relative_path(path): #get relative path from current file
    return os.path.join(os.path.dirname(__file__), path)

def load_tokenizer(model_path:str = get_relative_path('../tokenizer/tokenizer.model')) -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=model_path)

def global_main(config, model_class):
    tokenizer = load_tokenizer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model_class(vocab_size = tokenizer.get_piece_size(), **config.model)
    model.to(device)
    model.train()
    model.device = device

    print(f'Total Parameters (M): {model.total_params() / 1e6}')
    optimizer = torch.optim.AdamW(model.parameters(), **config.optim)
    return model, optimizer, tokenizer

def init_wandb(args, config, project_name):
    if not args.no_wandb:
        wandb.init(
            project=project_name, 
            config=OmegaConf.to_container(config, resolve=True),
            dir="./" if (config.get('wandb', {}).get('log_dir', None) is None) else config.wandb.log_dir
        )