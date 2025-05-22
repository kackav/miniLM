import os
import random
import requests

import torch


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder


def train_bpe(data, vocab_size=1024, special_tokens=None, output_dir='data', num_sentences=100000):
    if special_tokens is None:
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"]
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_json = f'{output_dir}/bpe_tokenizer_{vocab_size}.json'

    if not os.path.isfile(tokenizer_json):
        data_list = [_['text'] for _ in data]
        data_list_strip = [_.strip() for _ in data_list if len(_) > 0]
        if num_sentences > len(data_list_strip):
            num_sentences = len(data_list_strip)
        data_list_sub = random.sample(data_list_strip, num_sentences)
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
        tokenizer.train_from_iterator(data_list_sub, trainer)
        tokenizer.decoder = BPEDecoder()
        tokenizer.save(tokenizer_json)
    else:
        tokenizer = load_tokenizer(tokenizer_json)
    return tokenizer


def load_tokenizer(tokenizer_json):
    with open(tokenizer_json, 'r', encoding='utf-8') as f:
        tokenizer_str = f.read()
    tokenizer = Tokenizer.from_str(tokenizer_str)
    return tokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        inp = '[BOS]' + text
        label = text + '[EOS]'
        return {'input_ids': self.tokenizer.encode(inp).ids,
                'labels': self.tokenizer.encode(label).ids,
                }

    # def __iter__(self):
    #     for i in range(len(self.data)):
    #         yield self[i]

def collate_fn(batch, tokenizer=None):
    input_ids = [torch.tensor(_['input_ids'], dtype=torch.long) for _ in batch]
    labels = [torch.tensor(_['labels'], dtype=torch.long) for _ in batch]
    # pad and concatenate input_ids and labels
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.token_to_id('[PAD]') if tokenizer is not None else 0
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {'input_ids': input_ids, 'labels': labels}


def load_data(working_directory='data',
                          name='librispeech',
                          data_url='/mnt/matylda6/ivendrame/miniLM/data/librispeech_shuf.txt',
                          train_split_proportion=0.9999,
                          ):

    with open(data_url, 'r', encoding='utf-8') as f:
        data = f.read()
        data = [{'text': f'{_}\n'} for _ in data.split('\n') if _ ]
        #    random.shuffle(data)
    n = len(data)
    train_data = data[:int(n * train_split_proportion)]
    val_data = data[int(n * train_split_proportion):]

    return train_data, val_data


def load_from_textfile(data_url='/mnt/matylda6/ivendrame/miniLM/data/librispeech_shuf.txt'):

    with open(data_url, 'r', encoding='utf-8') as f:
        data = f.read()
        data = [{'text': f'{_}\n'} for _ in data.split('\n') if _ and len(_.split())< 231]
        return data