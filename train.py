from omegaconf import OmegaConf
import argparse
from transformer import Model
import wandb
import torch
import sentencepiece as spm
import data_gen
from functools import partial
from tqdm import tqdm

def load_tokenizer(model_path:str = './tokenizer/tokenizer.model') -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=model_path)

def train_loop(config, model, optimizer, tokenizer, receive_batch):
    period_id = tokenizer.piece_to_id('.')
    loss = torch.nn.MSELoss()
    pbar = tqdm(range(config.train.num_steps), total=config.train.num_steps)
    for step in pbar:
        optimizer.zero_grad()
        text, word_lengths = receive_batch()
        y_p = model(text.to(model.device))
        # get locations of period id
        p_id_idx = (text == period_id).nonzero(as_tuple=True)[1]
        
        y_p = y_p[0, p_id_idx].squeeze()
        word_lengths = word_lengths.squeeze().to(model.device)
        loss_val = loss(y_p, word_lengths)
    
        loss_val.backward()
        optimizer.step()
        
        accuracy = round((y_p.round() == word_lengths).float().mean().item() * 100, ndigits=2)

        pbar.set_description(f'Loss: {round(loss_val.item(), ndigits=2)}, Accuracy: {accuracy}')

def format_batch(config, generate_batch, tokenizer):
    sentences = generate_batch()
    period_id = tokenizer.piece_to_id('.')
    sentence_lengths = []
    encoded_sentences = []
    for sentence in sentences:
        encoded_sentences += tokenizer.encode(sentence['text'])
        encoded_sentences.append(period_id)
        sentence_lengths.append(sentence['len'])

    encoded_sentences = torch.tensor(encoded_sentences, dtype=torch.long)[None]
    sentence_lengths = torch.tensor(sentence_lengths, dtype=torch.float32)[None]
    
    return encoded_sentences, sentence_lengths

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
        max_len=config.data_gen.max_len
    )
    receive_batch = partial(format_batch, config, generate_batch, tokenizer)

    train_loop(config, model, optimizer, tokenizer, receive_batch)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='settings.yaml')
    args = parser.parse_args()  
    main(OmegaConf.load(args.config))
