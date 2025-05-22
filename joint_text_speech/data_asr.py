import os
import random
import requests

import torch
import torchaudio
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder


def train_bpe(data, vocab_size=1024, special_tokens=None, output_dir='tokenizer', num_sentences=100000):
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

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, bos_token):
        self.data = data
        self.tokenizer = tokenizer
        if bos_token is None:
            self.bos_token = [self.tokenizer.eos_token_id]
        else:
            self.bos_token = self.tokenizer.encode(bos_token)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        inp =  text
        label = text
        
        return {"text_trans" : text,
                "input_ids" : self.bos + self.tokenizer.encode(inp),
                "input_len" : len(self.tokenizer.encode(inp)) + 1,
                "labels" : self.tokenizer.encode(label) + [self.tokenizer.eos_token_id],
                "labels_len" : len(self.tokenizer.encode(label)) + 1
                }


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, audio_dir, bos_token, fs = 16_000):
        self.data = data
        self.tokenizer = tokenizer
        self.audio_dir = audio_dir
        self.sample_rate = fs

        if bos_token is None:
            self.bos = [self.tokenizer.eos_token_id]
        else:
            self.bos = self.tokenizer.encode(bos_token)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text'][0]
        audio_path = self.data[idx]['text'][1]
        audio, fs = torchaudio.load(audio_path)
        if fs != self.sample_rate:
            audio = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(audio)
        inp = text
        label = text
        
        return {"audio" : audio,
                "audio_len" : audio.shape[1],
                "text_trans" : text,
                "input_ids" : self.bos + self.tokenizer.encode(inp),
                "input_len" : len(self.tokenizer.encode(inp)) + 1,
                "labels" : self.tokenizer.encode(label) + [self.tokenizer.eos_token_id],
                "labels_len" : len(self.tokenizer.encode(label)) + 1
                }
        
    def __iter__(self):
        for i in range(len(self.data)):
            yield self[i]

def collate_fn_asr(batch, tokenizer=None):
    audio = [_['audio'][0] for _ in batch]
    audio_len = torch.tensor([_['audio_len'] for _ in batch], dtype=torch.int)

    input_ids = [torch.tensor(_['input_ids'], dtype=torch.long) for _ in batch]
    labels = [torch.tensor(_['labels'], dtype=torch.long) for _ in batch]
    # pad and concatenate input_ids and labels
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer is not None else 0
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0)

    text_trans = [_['text_trans'] for _ in batch]
    input_len = torch.tensor([_['input_len'] for _ in batch], dtype=torch.int)
    labels_len = torch.tensor([_['labels_len'] for _ in batch], dtype=torch.int)

    return {"audio": audio,
            "audio_len": audio_len,
            "text_trans" : text_trans,
            "input_ids": input_ids,
            "input_len": input_len,
            "labels": labels,
            "labels_len": labels_len
            }

def collate_fn_text(batch, tokenizer=None):
    input_ids = [torch.tensor(_['input_ids'], dtype=torch.long) for _ in batch]
    labels = [torch.tensor(_['labels'], dtype=torch.long) for _ in batch]
    # pad and concatenate input_ids and labels
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer is not None else 0
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    input_len = torch.tensor([_['input_len'] for _ in batch], dtype=torch.int)
    labels_len = torch.tensor([_['labels_len'] for _ in batch], dtype=torch.int)

    return {"input_ids": input_ids,
            "input_len": input_len,
            "labels": labels,
            "labels_len": labels_len
            }


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


def load_from_textfile(textfile_path='/mnt/matylda6/ivendrame/miniLM/data/librispeech_shuf.txt'):

    with open(textfile_path, 'r', encoding='utf-8') as f:
        data = f.read()
        data = [{'text': f'{_}\n'} for _ in data.split('\n') if _ and len(_.split())< 231]

        return data
    

def load_from_transcript(trans_path, audio_dir):
    data_dict = []
    with open(trans_path, 'r', encoding='utf-8') as f:
        data = f.read().split('\n')
        
        for line in data:
            splt = line.split(maxsplit =  1)
            if len(splt)>1:
                audio_path = os.path.join(audio_dir, splt[0] + ".flac")
                trans = splt[1].lower()
                dict = {'track': f'{splt[0]}', 'text' : (trans, audio_path)}
                data_dict.append(dict)

        return data_dict
