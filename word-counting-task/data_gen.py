import random

def load_words(path='./words_alpha.txt'): # each line is a word taken from: https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt
    with open(path, 'r') as f:
        words = f.read().split('\n')
    return [el.strip() for el in words if el.strip() != '']


def generate_random_text(words_list, min_len=1, max_len=10):
    return ' '.join(random.choices(words_list, k=random.randint(min_len, max_len)))

def generate_batch(batch_size, words_list, min_len=1, max_len=10):
    batch = []
    for _ in range(batch_size):
        rnd_text = generate_random_text(words_list, min_len, max_len)
        txt_len = len(rnd_text.split()) # split by space
        batch.append({
            'text': rnd_text,
            'len': txt_len
        })
    return batch



