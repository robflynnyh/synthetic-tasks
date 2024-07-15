from omegaconf import OmegaConf
import argparse
import sys
from synthetic_tasks.modelling.transformer import Model
from synthetic_tasks.tooling.misc import global_main
import wandb
import torch
import data_gen
from functools import partial
from tqdm import tqdm
import random

start_word_token = '▁'



def train_loop(config, model, optimizer, tokenizer, receive_batch, logger=wandb):
    loss = torch.nn.MSELoss()

    pbar = tqdm(range(config.train.num_steps), total=config.train.num_steps)
    for step in pbar:
        optimizer.zero_grad()
        text, culm_scores = receive_batch(max_len = config.data_gen.max_len)
        text, culm_scores = text.to(model.device), culm_scores.to(model.device)
        y_p = model(text).squeeze(2)
        loss_val = loss(y_p, culm_scores)

        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        accuracy = round(((y_p).round() == culm_scores).float().mean().item() * 100, ndigits=2)

        pbar.set_description(f'Loss: {round(loss_val.item(), ndigits=2)}, Accuracy: {accuracy}')
        logger.log({'loss': loss_val.item(), 'accuracy': accuracy}) if logger else None

       
        

def format_batch(config, generate_batch, tokenizer, word_scores_dict, token_scores_dict, max_len):
    sentence = generate_batch(max_len = max_len)[0]
    encoded = tokenizer.encode(sentence['text'])

    # Assign scores to tokens
    pieces = tokenizer.encode_as_pieces(sentence['text'])
    token_scores = [token_scores_dict['forward'][token] for token in pieces]
    token_scores = torch.tensor(token_scores)[None]
    culm_token_scores = torch.cumsum(token_scores, dim=1)

    # assign scores to words
    
    words = sentence['text'].split()
    word_scores = [word_scores_dict['forward'][word] for word in words]
    culm_word_score = []
    cur_idx = -2
    for i, piece in enumerate(pieces):
        if piece.startswith(start_word_token):
            cur_idx += 1
            if cur_idx >= 0:
                culm_word_score.append(culm_word_score[-1] + word_scores[cur_idx])
            else: culm_word_score.append(0)
        else: culm_word_score.append(culm_word_score[-1])

    culm_word_score = torch.tensor(culm_word_score)[None]

    culm_scores = (culm_token_scores + culm_word_score).to(torch.float32)
    encoded = torch.tensor(encoded)[None]
    
    return encoded, culm_scores


def assign_word_scores(word_list, random_seed=1234, min_score=-5, max_score=5): 
    random.seed(random_seed)
    return {'forward': {word: random.randint(min_score, max_score) for word in word_list}}

def assign_token_scores(tokenizer, random_seed=1234, min_score=-5, max_score=5):
    tokens = [tokenizer.id_to_piece(i) for i in range(tokenizer.get_piece_size())]
    return assign_word_scores(tokens, random_seed=random_seed, min_score=min_score, max_score=max_score)

def main(config):
    
    model, optimizer, tokenizer = global_main(config, model_class = Model)
    
    assert config.data_gen.examples_per_batch == 1, 'Only 1 example per batch is supported (for now) to avoid padding'
    words_list = data_gen.load_words()
    word_scores_dict, token_scores_dict = assign_word_scores(words_list), assign_token_scores(tokenizer)
    generate_batch = partial(
        data_gen.generate_batch, 
        words_list=words_list, 
        min_len=config.data_gen.min_len,
        #max_len = config.data_gen.max_len,
        batch_size = 1,
    )
    receive_batch = partial(format_batch, config, generate_batch, tokenizer, word_scores_dict, token_scores_dict)
    
    if not args.no_wandb:
        wandb.init(project='score-tracker', config=OmegaConf.to_container(config, resolve=True))
    train_loop(config, model, optimizer, tokenizer, receive_batch, logger=wandb if not args.no_wandb else None)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='settings.yaml')
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()  
    main(OmegaConf.load(args.config))
