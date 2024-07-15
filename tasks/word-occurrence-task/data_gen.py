import random
from synthetic_tasks.tooling.misc import get_relative_path

def load_words(path=get_relative_path('../artifacts/words_alpha.txt'), random_k=128, seed=12351): # each line is a word taken from: https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt
    with open(path, 'r') as f:
        words = f.read().split('\n')
    words_list = [el.strip() for el in words if el.strip() != '']
    random.seed(seed)
    return random.choices(words_list, k=random_k)


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



