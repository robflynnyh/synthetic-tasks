from omegaconf import OmegaConf
import argparse
from ..modelling.transformer import Model
from ..tooling.misc import get_relative_path
import wandb
import torch
import sentencepiece as spm
import data_gen
from functools import partial
from tqdm import tqdm


def load_tokenizer(model_path:str = get_relative_path('../tokenizer/tokenizer.model')) -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=model_path)

def train_loop(config, model, optimizer, tokenizer, receive_batch, logger=wandb):
    period_id = tokenizer.piece_to_id('.')
    loss = torch.nn.MSELoss()
    shift_by = 2
    cur_max_len = 2
    max_max_len = config.data_gen.max_len
    
    pbar = tqdm(range(config.train.num_steps), total=config.train.num_steps)
    for step in pbar:
        optimizer.zero_grad()
        text, word_lengths, token_lenghts = receive_batch(max_len=cur_max_len)

        y_p = model(text.to(model.device))
        # get locations of period id
        p_id_idx = (text == period_id).nonzero(as_tuple=True)[1]
        
        y_p = y_p[0, p_id_idx].squeeze()
        word_lengths = word_lengths.squeeze()
        
        if shift_by > 0:
            shift_fwd = torch.cat([torch.zeros(shift_by, dtype=torch.long), word_lengths[:-shift_by]])
            shift_bwd = torch.cat([word_lengths[shift_by:], torch.zeros(shift_by, dtype=torch.long)])
            word_lengths = (word_lengths + shift_fwd + shift_bwd) 

        word_lengths = word_lengths.to(model.device)
        
        loss_val = loss(y_p, word_lengths)
    
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        accuracy = round(((y_p).round() == word_lengths).float().mean().item() * 100, ndigits=2)
        if accuracy > 99:
            cur_max_len = min(cur_max_len + 1, max_max_len)

        pbar.set_description(f'Loss: {round(loss_val.item(), ndigits=2)}, Accuracy: {accuracy}, Max Len: {cur_max_len}')
        wandb.log({'loss': loss_val.item(), 'accuracy': accuracy, 'max_len': cur_max_len}) if logger else None

def format_batch(config, generate_batch, tokenizer, max_len):
    sentences = generate_batch(max_len=max_len)
    period_id = tokenizer.piece_to_id('.')
    sentence_lengths = []
    token_lengths = []
    encoded_sentences = []
    for sentence in sentences:
        encoded = tokenizer.encode(sentence['text'])
        encoded_sentences += encoded
        encoded_sentences.append(period_id)
        token_lengths.append(len(encoded))
        sentence_lengths.append(sentence['len'])

    encoded_sentences = torch.tensor(encoded_sentences, dtype=torch.long)[None]
    sentence_lengths = torch.tensor(sentence_lengths, dtype=torch.float32)[None]
    token_lengths = torch.tensor(token_lengths, dtype=torch.float32)[None]
    
    return encoded_sentences, sentence_lengths, token_lengths

def main(config):
    tokenizer = load_tokenizer(config.tokenizer_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Model(vocab_size = tokenizer.get_piece_size(), **config.model)
    model.to(device)
    model.train()
    model.device = device

    print(f'Total Parameters (M): {model.total_params() / 1e6}')
    optimizer = torch.optim.AdamW(model.parameters(), **config.optim)
    
    assert config.data_gen.examples_per_batch == 1, 'Only 1 example per batch is supported (for now) to avoid padding'
    words_list = data_gen.load_words()
    generate_batch = partial(
        data_gen.generate_batch, 
        words_list=words_list, 
        batch_size=config.data_gen.sentences_per_example * config.data_gen.examples_per_batch,
        min_len=config.data_gen.min_len,
    )
    receive_batch = partial(format_batch, config, generate_batch, tokenizer)
    
    if not args.no_wandb:
        wandb.init(project='word-count', config=OmegaConf.to_container(config, resolve=True))
    train_loop(config, model, optimizer, tokenizer, receive_batch, logger=wandb if not args.no_wandb else None)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='settings.yaml')
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()  
    main(OmegaConf.load(args.config))
