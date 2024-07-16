from omegaconf import OmegaConf
import argparse
import sys
from synthetic_tasks.modelling.transformer import Model
from synthetic_tasks.tooling.misc import global_main, init_wandb
import wandb
import torch
import data_gen
from functools import partial
from tqdm import tqdm
import random

start_word_token = '▁'



def train_loop(config, model, optimizer, tokenizer, receive_batch, logger=wandb):
    period_id = tokenizer.piece_to_id('.')
    loss = torch.nn.MSELoss()
    shift_by = 2
    max_sentences_per_example = config.data_gen.sentences_per_example
    cur_sentences_per_example = 1
    

    pbar = tqdm(range(config.train.num_steps), total=config.train.num_steps)
    for step in pbar:
        optimizer.zero_grad()
        text, targets, token_lenghts = receive_batch(batch_size=cur_sentences_per_example)
        text, targets = text.to(model.device), targets.to(model.device)
        y_p = model(text).squeeze(2)
        loss_val = loss(y_p, targets)

        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        accuracy = round(((y_p).round() == targets).float().mean().item() * 100, ndigits=2)
        if accuracy > 99 and cur_sentences_per_example < max_sentences_per_example:
            cur_sentences_per_example += 1
        elif accuracy < 90 and cur_sentences_per_example > 2:
            cur_sentences_per_example -= 1

        pbar.set_description(f'Loss: {round(loss_val.item(), ndigits=2)}, Accuracy: {accuracy}')
        logger.log({'loss': loss_val.item(), 'accuracy': accuracy, 'sentences_per_example': cur_sentences_per_example}) if logger else None
        

        

def format_batch(config, generate_batch, tokenizer, batch_size):
    sentences = generate_batch(batch_size=batch_size)
    period_id = tokenizer.piece_to_id('.')
    token_lengths = []
    encoded_sentences = []
    word_positions_collection = []
    offset = 0

    words_in_sentences = {}
    for sentence in sentences:
        encoded = tokenizer.encode(sentence['text'])
        words = sentence['text'].split()
        as_pieces = tokenizer.encode_as_pieces(sentence['text'])
        word_positions = {}
        
        words_in_sentence = set(words)
        for word in words_in_sentence:
            if word not in words_in_sentences: words_in_sentences[word] = 0
            words_in_sentences[word] += 1
        
        cur_word_i = -1
        for idx, piece in enumerate(as_pieces):
            if piece.startswith(start_word_token): cur_word_i += 1
            if words[cur_word_i] not in word_positions: word_positions[words[cur_word_i]] = []
            word_positions[words[cur_word_i]].append(idx + offset)

        encoded_sentences += encoded
        encoded_sentences.append(period_id)
        token_lengths.append(len(encoded))
        offset += len(encoded) + 1
        word_positions_collection.append(word_positions)

    encoded_sentences = torch.tensor(encoded_sentences, dtype=torch.long)[None]
    token_lengths = torch.tensor(token_lengths, dtype=torch.float32)[None]

    targets = torch.zeros(1, encoded_sentences.shape[1], dtype=torch.float32)
    for word_positions in word_positions_collection:
        for word_key in word_positions.keys():
            for pos in word_positions[word_key]:
                targets[0, pos] = words_in_sentences[word_key] - 1
    
    

    return encoded_sentences, targets, token_lengths

def main(config):
    
    model, optimizer, tokenizer = global_main(config, model_class = Model)
    
    assert config.data_gen.examples_per_batch == 1, 'Only 1 example per batch is supported (for now) to avoid padding'
    words_list = data_gen.load_words()
    generate_batch = partial(
        data_gen.generate_batch, 
        words_list=words_list, 
        min_len=config.data_gen.min_len,
        max_len = config.data_gen.max_len,
    )
    receive_batch = partial(format_batch, config, generate_batch, tokenizer)

    init_wandb(args, config, project_name = 'word-occurrrence')

    train_loop(config, model, optimizer, tokenizer, receive_batch, logger=wandb if not args.no_wandb else None)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='settings.yaml')
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()  
    main(OmegaConf.load(args.config))
