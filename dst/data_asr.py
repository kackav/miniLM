import os
import random
import requests
import json
import functools


import torch
import torchaudio
import librosa
import soundfile
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder
import datasets
import yaml

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
        self.bos_token = bos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        inp =  text
        label = text
        return {"text_trans" : text,
                "input_ids" : [self.bos_token] + self.tokenizer.encode(inp),
                "input_len" : len(self.tokenizer.encode(inp)) + 1,
                "labels" : self.tokenizer.encode(label) + [self.tokenizer.eos_token_id],
                "labels_len" : len(self.tokenizer.encode(label)) + 1
                }
class AudioDatasetHF(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, bos_token):
        self.data = data
        self.is_fake = False
        self.fake_audio_len = 30
        self.fake_text_len = 20
        self.bos_token = bos_token
        
        #librispeech: iterable(dataset), ...
        self.data_lengths ={k: len(v) for k,v in data.items()}
        # self.cumulative_lengths = torch.sum(self.data_lengths)
        # print(self.cumulative_lengths)
        self.tokenizer = tokenizer

    def __len__(self):
        #print(self.data_lengths.values())
        return sum(list(self.data_lengths.values()))
    
    def __getitem__(self, idx):
        if self.is_fake:
            text = "This is a fake text for resume."
            audio = torch.zeros(int(self.fake_audio_len * 16_000))  # shape (1, fake_audio_len * fs)
            audio = audio.unsqueeze(0)  # add channel dimension
            inp = [0] * self.fake_text_len
            label = [0] * self.fake_text_len
            return {"audio": audio,
                "audio_len": audio.shape[1],
                "text_trans": text,
                "input_ids": inp,
                "input_len": self.fake_text_len,
                "labels": label,
                "labels_len" : self.fake_text_len
                }
        
        #print(f"__getitem__ called with idx={idx}")
        #print(self.data_lengths)
        total_length = 0
        for key, length in self.data_lengths.items():
            previous_length = total_length
            total_length += length
            if idx < total_length:
                idx -= previous_length
                break

         #normalized data structure - if dataset doesn't have it, prep it according to commonvoice_prep.py
        data_key = self.data[key][idx]
        text = data_key['normalized_text'] #if "normalized_text" in data_key else data_key["text"].lower()
        # print(data_key["audio"])
        audio = torch.tensor(data_key['audio']['array'], dtype=torch.float32)
        sample_rate = data_key['audio']['sampling_rate']
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)        
        audio = audio.unsqueeze(0)  # add channel dimension
        inp = text
        label = text
        encoded_inp = self.tokenizer.encode(inp, add_special_tokens = False)
        encoded_lab = self.tokenizer.encode(label, add_special_tokens = False)
        
        return {"audio" : audio,
                "audio_len" : audio.shape[1],
                "text_trans" : text,
                "input_ids" : [self.bos_token] + encoded_inp,
                "input_len" : len(encoded_inp) + 1,
                "labels" : encoded_lab + [self.tokenizer.eos_token_id],
                "labels_len" : len(encoded_lab) + 1
                }
        

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, audio_dir, bos_token, fs = 16_000):
        self.data = data
        self.tokenizer = tokenizer
        self.audio_dir = audio_dir
        self.sample_rate = fs
        self.bos_token = bos_token

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
                "input_ids" : [self.bos_token] + self.tokenizer.encode(inp),
                "input_len" : len(self.tokenizer.encode(inp)) + 1,
                "labels" : self.tokenizer.encode(label) + [self.tokenizer.eos_token_id],
                "labels_len" : len(self.tokenizer.encode(label)) + 1
                }
        
    def __iter__(self):
        for i in range(len(self.data)):
            yield self[i]

class TextDatasetHF(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, bos_token):
        self.data = data
        self.is_fake = False
        self.data_lengths = {key: len(v) for key,v in data.items()}
        self.tokenizer = tokenizer
        self.bos_token = bos_token
        self.fake_text_len = 30

    def __len__(self):
        return sum(list(self.data_lengths.values()))

    def __getitem__(self, idx):
        if self.is_fake:
            text = "This is a fake text for resume."
            inp = [0] * self.fake_text_len
            label = [0] * self.fake_text_len
            return {"text_trans": text,
                "input_ids": inp,
                "input_len": self.fake_text_len,
                "labels": label,
                "labels_len" : self.fake_text_len
                }
        
        total_length = 0
        for key, length in self.data_lengths.items():
            previous_length = total_length
            total_length += length
            if idx < total_length:
                idx -= previous_length
                break

        data_key = self.data[key][idx]
        text = data_key['text'].lower()
        # cut text to max 230 characters
        if len(text) > 180:
            text = text[:180]
        inp = text
        label = text
        inp_encoded = self.tokenizer.encode(inp, add_special_tokens = False)
        lab_encoded = self.tokenizer.encode(label, add_special_tokens = False)
        
        return {"text_trans" : text,
                "input_ids" : [self.bos_token] + inp_encoded,
                "input_len" : len(inp_encoded) + 1,
                "labels" : lab_encoded + [self.tokenizer.eos_token_id],
                "labels_len" : len(lab_encoded) + 1
                }

class DialogDatasetHF(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, bos_token):
        self.data = data
        self.is_fake = False
        self.fake_audio_len = 30
        self.fake_text_len = 20
        self.bos_token = bos_token
        self.fake_context_len = 10
        #librispeech: iterable(dataset), ...
        self.data_lengths ={k: len(v) for k,v in data.items()}
        # self.cumulative_lengths = torch.sum(self.data_lengths)
        # print(self.cumulative_lengths)
        self.tokenizer = tokenizer

    def __len__(self):
        #print(self.data_lengths.values())
        return sum(list(self.data_lengths.values()))
    
    def __getitem__(self, idx):
        if self.is_fake:
            text = "This is a fake text for resume."
            context = "This is a fake context for resume."
            audio = torch.zeros(int(self.fake_audio_len * 16_000))  # shape (1, fake_audio_len * fs)
            audio = audio.unsqueeze(0)  # add channel dimension
            inp = [0] * self.fake_text_len
            label = [0] * self.fake_text_len
            encoded_context = [0] * self.fake_context_len
            
            return {"audio" : audio,
                "audio_len" : audio.shape[1],
                "text_trans" : text,
                "context_trans" : context,
                "context" : encoded_context,
                "context_len" : len(encoded_context),
                "input_ids" : inp,
                "input_len" : len(inp),
                "labels" : label,
                "labels_len" : len(label),
                }
        
        #print(f"__getitem__ called with idx={idx}")
        #print(self.data_lengths)
        total_length = 0
        for key, length in self.data_lengths.items():
            previous_length = total_length
            total_length += length
            if idx < total_length:
                idx -= previous_length
                break
 

        data_key = self.data[key][idx]
        text = data_key['text'] #if "normalized_text" in data_key else data_key["text"].lower()
        context = data_key['context']
        # print(data_key["audio"])
        audio = torch.tensor(data_key['audio']['array'], dtype=torch.float32)
        sample_rate = data_key['audio']['sampling_rate']
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)        
        audio = audio.unsqueeze(0)  # add channel dimension
        inp = text
        label = text
        text_decoded = json.loads(text)
        pre_text = text_decoded['label']

        encoded_inp = self.tokenizer.encode(inp, add_special_tokens = False)
        encoded_lab = self.tokenizer.encode(label, add_special_tokens = False)
        encoded_context = self.tokenizer.encode(context, add_special_tokens = False)
        encoded_pre_text = self.tokenizer.encode(pre_text, add_special_tokens = False)

        return {"audio" : audio,
                "audio_len" : audio.shape[1],
                "text_trans" : text,
                "context_trans" : context,
                "context" : [self.bos_token] + encoded_context,
                "context_len" : len(encoded_context) +1,
                "pre_text_trans" : pre_text,
                "pre_text" : encoded_pre_text,
                "pre_text_len" : len(encoded_pre_text),
                "input_ids" : [self.bos_token] + encoded_inp,
                "input_len" : len(encoded_inp) + 1,
                "labels" : encoded_lab + [self.tokenizer.eos_token_id],
                "labels_len" : len(encoded_lab) + 1,
                "wav_id": data_key["wav_id"],
                "turn_index": data_key["turn_index"],
                "agent_text": data_key["agent_text"],
                "domains": data_key["domains"],
                "slots": data_key["slots"],
                }


class InferenceDialogDatasetHF(torch.utils.data.Dataset):
    def __init__(self, non_inference_dialog_dataset: DialogDatasetHF):
        self.non_inference_dialog_dataset = non_inference_dialog_dataset      
        wavids = {k : v["wav_id"] for k, v in self.non_inference_dialog_dataset.data.items()}
        wav_id_to_dataset = {}
        assert len(self.non_inference_dialog_dataset.data) == 1, "only one dataset supported for inference"
        for k, v in wavids.items():
            for w in v:
                if w in wav_id_to_dataset and wav_id_to_dataset[w] != k:
                    raise ValueError(f"wav_id {w} is present in multiple datasets, cannot do inference per wav_id")
                wav_id_to_dataset[w] = k
        self.wav_id_to_dataset = wav_id_to_dataset
        wavids = list(wavids.values())
        wavids = sum(wavids, [])
        wavids_to_inds = {}
        for i, w in enumerate(wavids):
            if w not in wavids_to_inds.keys():
                wavids_to_inds[w] = []
            wavids_to_inds[w].append(i)
        self.wavids_to_inds = wavids_to_inds
        self.wavids = list(self.wavids_to_inds.keys())

    def __len__(self):
        return len(self.wavids_to_inds)

    def __getitem__(self, ind):
        wavid = self.wavids[ind]
        dataset_of_dialog = self.wav_id_to_dataset[wavid]
        data_indices = self.wavids_to_inds[wavid]
        # the problem is that these items are from originial dataset not as returned by getitem of DialogHF . They should not be
        items = [self.non_inference_dialog_dataset[i] for i in data_indices]
        agent_text = [self.non_inference_dialog_dataset.data[dataset_of_dialog][i]["agent_text"] for i in data_indices]
        # sort based on turn_index
        for i in range(len(items)):
            items[i]["agent_text"] = agent_text[i]
            items[i]["wav_id"] = wavid
        items = sorted(items, key=lambda x: x["turn_index"])
      
        # agent_text = [self.non_inference_dialog_dataset[i]["agent_text"] for i in data_indices]
        return items


class TextDialogDatasetHF(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, bos_token):
        self.data = data
        self.is_fake = False
        self.data_lengths = {key: len(v) for key,v in data.items()}
        self.tokenizer = tokenizer
        self.bos_token = bos_token
        self.fake_text_len = 30

    def __len__(self):
        return sum(list(self.data_lengths.values()))

    def __getitem__(self, idx):
        if self.is_fake:
            text = "This is a fake text for resume."
            inp = [0] * self.fake_text_len
            label = [0] * self.fake_text_len
            return {"text_trans": text,
                "input_ids": inp,
                "input_len": self.fake_text_len,
                "labels": label,
                "labels_len" : self.fake_text_len
                }

        total_length = 0
        for key, length in self.data_lengths.items():
            previous_length = total_length
            total_length += length
            if idx < total_length:
                idx -= previous_length
                break

        data_key = self.data[key][idx]
        text = data_key['text'].lower()
        #cut text to max 230 characters
        # if len(text) > 180:
        #    text = text[:180]
        inp = text
        label = text
        text_decoded = json.loads(text)
        pre_text = text_decoded['label']
        inp_encoded = self.tokenizer.encode(inp, add_special_tokens = False)
        inp_encoded = [self.bos_token] + inp_encoded
        if len(inp_encoded) > 512:
            inp_encoded = inp_encoded[:512]
        lab_encoded = self.tokenizer.encode(label, add_special_tokens = False)
        lab_encoded = lab_encoded + [self.tokenizer.eos_token_id]
        if len(lab_encoded) > 512:
            lab_encoded = lab_encoded[:512]
            
        context = data_key['context']
        encoded_context = self.tokenizer.encode(context, add_special_tokens = False)
        encoded_pre_text = self.tokenizer.encode(pre_text, add_special_tokens = False)
        if len(encoded_pre_text) > 512:
            encoded_pre_text[:512]

        return {
                "context_trans" : context,
                "context" : [self.bos_token] + encoded_context,
                "context_len" : len(encoded_context) +1,
                "pre_text_trans" : pre_text,
                "pre_text" : encoded_pre_text,
                "pre_text_len" : len(encoded_pre_text),
                "text_trans" : text,
                "input_ids" : inp_encoded,
                "input_len" : len(inp_encoded),
                "labels" : lab_encoded,
                "labels_len" : len(lab_encoded),
                }

def load_from_config(ds_type, config_datasets, hf_dataset_path, accelerator=None):
    loaded_datasets = {}
    #hf_dataset_path = "/mnt/scratch/tmp/ivendrame/huggingface/modules/datasets_modules/datasets/"
    for k,v in config_datasets[ds_type].items():
        dataset_path = os.path.join(hf_dataset_path, f"prep_{k}_{ds_type}")
        if "path_to_subsets" in v:
            #dataset_path = os.path.join(dataset_path, v["path_to_subsets"]
            print(f"loading prepared subsets of dataset {k} from {dataset_path}")
            #dataset_path = os.path.join(dataset_path, f"prep_{k}_{split}_")
            for ds in os.listdir(dataset_path):
                loaded_dataset = datasets.load_from_disk(os.path.join(dataset_path, ds))
                loaded_datasets[ds]= loaded_dataset
        
        elif os.path.exists(dataset_path):
            print(f"loading prepared dataset {k} from {dataset_path}")
            loaded_datasets[k] = datasets.load_from_disk(dataset_path)

            if 'audio_len' in loaded_datasets[k].features:
                if accelerator is not None or accelerator.is_main_process:
                    loaded_datasets[k] = loaded_datasets[k].filter(lambda item: item['audio_len']<(30*16_000), num_proc=4)
                    loaded_datasets[k] = loaded_datasets[k].filter(lambda item: item['audio_len']>(0.5*16_000), num_proc=4)
        else:
            print(f"prepared dataset not found {k} in {dataset_path}, taking original dataset and thus filtering audio discarding audios longer then 30s")
            ds_name = v["name"].split(":")[0] if ":" in v["name"] else v["name"]
            ds_subset = v["name"].split(":")[1] if ":" in v["name"] else None
            ds_split = v["split"]
            #ds_text_column = v["text_column"]
            if ds_subset:
                loaded_datasets[k] = datasets.load_dataset(ds_name, ds_subset, split = ds_split,  num_proc=2) #trust_remote_code=True,
            else:
                loaded_datasets[k] = datasets.load_dataset(ds_name, split = ds_split,  num_proc=2) # trust_remote_code=True,
            if 'audio' in loaded_datasets[k].features:
                if accelerator is not None or accelerator.is_main_process:
                    loaded_datasets[k] = loaded_datasets[k].filter(lambda item: len(item['audio']["array"])<(30*16_000), num_proc=4)
                    loaded_datasets[k] = loaded_datasets[k].filter(lambda item: len(item['audio']["array"])>(0.5*16_000), num_proc=4)          
        
    return loaded_datasets

def filter_long_text(example, tokenizer, max_text_len=200):
    encoded = tokenizer.encode(json.loads(example["text"])["label"], add_special_tokens = False)
    return len(encoded) < max_text_len

def load_dial_from_config(ds_type, config_datasets, hf_dataset_path, tokenizer=None, accelerator=None):
    loaded_datasets = {}
    #hf_dataset_path = "/mnt/scratch/tmp/ivendrame/huggingface/modules/datasets_modules/datasets/"
    for k,v in config_datasets[ds_type].items():
        dataset_path = os.path.join(hf_dataset_path, f"prep_dial_{k}_{ds_type}")
        
            #datasets.load_from_disk(dataset_path)
        if "path_to_subsets" in v:
            #dataset_path = os.path.join(dataset_path, v["path_to_subsets"]
            print(f"loading prepared subsets of dataset {k} from {dataset_path}")
            #dataset_path = os.path.join(dataset_path, f"prep_{k}_{split}_")
            for ds in os.listdir(dataset_path):
                loaded_dataset = datasets.load_from_disk(os.path.join(dataset_path, ds))
                if accelerator is not None or accelerator.is_main_process:
                    loaded_dataset = loaded_dataset.filter(functools.partial(filter_long_text, tokenizer=tokenizer, max_text_len=200), num_proc=4)   
                loaded_datasets[ds]= loaded_dataset
        
        elif os.path.exists(dataset_path):
            print(f"loading prepared dataset {k} from {dataset_path}")
            if k == "spokenwoz":
                dataset_path = os.path.join(hf_dataset_path, f"prep_dial_{k}_allColumns_{ds_type}")
            loaded_datasets[k] = datasets.load_from_disk(dataset_path)
            if 'audio_len' in loaded_datasets[k].features:
                if accelerator is not None or accelerator.is_main_process:
                    loaded_datasets[k] = loaded_datasets[k].filter(lambda item: item['audio_len']<(30*16_000), num_proc=4)
                    loaded_datasets[k] = loaded_datasets[k].filter(lambda item: item['audio_len']>(0.5*16_000), num_proc=4)
            if accelerator is not None or accelerator.is_main_process:
                loaded_datasets[k] = loaded_datasets[k].filter(functools.partial(filter_long_text, tokenizer=tokenizer, max_text_len=200), num_proc=4)   
        

        else:
            print(f"prepared dataset not found {k} in {dataset_path}, taking original dataset and thus filtering audio discarding audios longer then 30s")
            ds_name = v["name"].split(":")[0] if ":" in v["name"] else v["name"]
            ds_subset = v["name"].split(":")[1] if ":" in v["name"] else None
            ds_split = v["split"]
            #ds_text_column = v["text_column"]
            if ds_subset:
                loaded_datasets[k] = datasets.load_dataset(ds_name, ds_subset, split = ds_split,  num_proc=2) #trust_remote_code=True,
            else:
                loaded_datasets[k] = datasets.load_dataset(ds_name, split = ds_split,  num_proc=2) # trust_remote_code=True,
            if 'audio' in loaded_datasets[k].features:
                if accelerator is not None or accelerator.is_main_process:
                    loaded_datasets[k] = loaded_datasets[k].filter(lambda item: len(item['audio']["array"])<(30*16_000), num_proc=4)
                    loaded_datasets[k] = loaded_datasets[k].filter(lambda item: len(item['audio']["array"])>(0.5*16_000), num_proc=4)
            if accelerator is not None or accelerator.is_main_process:
                loaded_datasets[k] = loaded_datasets[k].filter(functools.partial(filter_long_text, tokenizer=tokenizer, max_text_len=200), num_proc=4)   
        
    return loaded_datasets

class CollateFnSpeech:
    #collate_fn = CollateFnASR(tokenizer)
    #dataloader = DataLoader(dataset, batch_size=..., collate_fn=collate_fn)
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        audio = [_['audio'][0] for _ in batch]
        audio_len = torch.tensor([_['audio_len'] for _ in batch], dtype=torch.int)

        input_ids = [torch.tensor(_['input_ids'], dtype=torch.long) for _ in batch]
        labels = [torch.tensor(_['labels'], dtype=torch.long) for _ in batch]
        # pad and concatenate input_ids and labels
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0
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
    
class CollateFnText:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [torch.tensor(_['input_ids'], dtype=torch.long) for _ in batch]
        labels = [torch.tensor(_['labels'], dtype=torch.long) for _ in batch]
        # pad and concatenate input_ids and labels
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        input_len = torch.tensor([_['input_len'] for _ in batch], dtype=torch.int)
        labels_len = torch.tensor([_['labels_len'] for _ in batch], dtype=torch.int)
        text_trans = [_['text_trans'] for _ in batch]
        
        return {"text_trans" : text_trans,
                "input_ids": input_ids,
                "input_len": input_len,
                "labels": labels,
                "labels_len": labels_len
                }
    
class CollateFnDialogInference(CollateFnSpeech):
    def __init__(self, non_inference_collate_fn, tokenizer=None):
        super().__init__(tokenizer)
        self.non_inference_collate_fn = non_inference_collate_fn
    #batch, non_inference_collate_fn, tokenizer):
    
    def __call__(self, batch):#List[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[List[Optional[int]]]]:
        #import pdb; breakpoint()
        lengths = [len(x) for x in batch]
        max_length = max(lengths)
        out_batch = []
        pointers = []
        initial_context = ""
        initial_context = self.tokenizer.encode(initial_context, add_special_tokens=False)
        initial_context_length = len(initial_context)

        for i in range(max_length):
            out_batch.append([])
            pointers.append([])
            agent_text = []
            wav_ids = []
            for batch_idx, conversation in enumerate(batch):
                #if k >= i:
                if i >= len(conversation):
                    pointers[-1].append(None)
                    continue
                turn = conversation[i]
                # turn.pop("context")
                # turn.pop("context_len")
                # turn["context"] = initial_context
                # turn["context_len"] = initial_context_length
                # turn["context_trans"] = ""
                out_batch[-1].append(turn)
                pointers[-1].append(batch_idx)
                agent_text.append(turn["agent_text"])
                wav_ids.append(turn["wav_id"])
            
            out_batch[-1] = self.non_inference_collate_fn(out_batch[-1])
            out_batch[-1]["agent_text"] = agent_text
            out_batch[-1]["wav_id"] = wav_ids
        for i, p_out in enumerate(pointers):
            for j, p_in in enumerate(p_out):
                if p_in is None:
                    continue
                count_none = len([_ for _ in p_out[:j] if _ is None])
                pointers[i][j] -= count_none
                assert pointers[i][j] >= 0, ("all pointers should have non-negative values")
        return out_batch, pointers
    

def collate_all_text(batch_s, batch_t, tokenizer=None):
    input_ids_s = torch.nn.utils.rnn.unpad_sequence(
        batch_s['input_ids'], lengths = batch_s['input_len'], batch_first=True
    )
    input_ids_t = torch.nn.utils.rnn.unpad_sequence(
        batch_t['input_ids'], lengths = batch_t['input_len'], batch_first=True
    )
    input_ids = input_ids_s + input_ids_t

    labels_s = torch.nn.utils.rnn.unpad_sequence(batch_s['labels'], batch_first=True, lengths = batch_s['labels_len'])
    labels_t = torch.nn.utils.rnn.unpad_sequence(batch_t['labels'], batch_first=True, lengths = batch_t['labels_len'])
    labels = labels_s + labels_t
    
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer is not None else 0
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    input_len = torch.tensor(batch_s['input_len'].tolist()+ batch_t['input_len'].tolist())
    labels_len = torch.tensor(batch_s['labels_len'].tolist() + batch_t['labels_len'].tolist())
    text_trans = batch_s['text_trans'] + batch_t['text_trans']
    
    
    return {"text_trans" : text_trans,
            "input_ids": input_ids,
            "input_len": input_len,
            "labels": labels,
            "labels_len": labels_len
            }
            
class CollateFnDialog:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        audio = [_['audio'][0] for _ in batch]
        audio_len = torch.tensor([_['audio_len'] for _ in batch], dtype=torch.int)

        input_ids = [torch.tensor(_['input_ids'], dtype=torch.long) for _ in batch]
        labels = [torch.tensor(_['labels'], dtype=torch.long) for _ in batch]
        context = [torch.tensor(_['context'], dtype = torch.long) for _ in batch]
        # pad and concatenate input_ids and labels
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0)
        context = torch.nn.utils.rnn.pad_sequence(context, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0)

        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0)

        text_trans = [_['text_trans'] for _ in batch]
        context_trans = [_['context_trans'] for _ in batch]

        input_len = torch.tensor([_['input_len'] for _ in batch], dtype=torch.int)
        labels_len = torch.tensor([_['labels_len'] for _ in batch], dtype=torch.int)
        context_len = torch.tensor([_['context_len'] for _ in batch], dtype=torch.int)

        pre_text = [torch.tensor(_['pre_text'], dtype=torch.long) for _ in batch]
        pre_text = torch.nn.utils.rnn.pad_sequence(pre_text, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0)
        pre_text_len = torch.tensor([_['pre_text_len'] for _ in batch], dtype=torch.int)

        return {"audio": audio,
                "audio_len": audio_len,
                "text_trans" : text_trans,
                "context_trans" : context_trans,
                "context" : context,
                "context_len" : context_len,
                "input_ids": input_ids,
                "input_len": input_len,
                "labels": labels,
                "labels_len": labels_len,
                "pre_text": pre_text,
                "pre_text_len": pre_text_len,
                }

class CollateFnDialogText:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [torch.tensor(_['input_ids'], dtype=torch.long) for _ in batch]
        labels = [torch.tensor(_['labels'], dtype=torch.long) for _ in batch]
        # pad and concatenate input_ids and labels
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        input_len = torch.tensor([_['input_len'] for _ in batch], dtype=torch.int)
        labels_len = torch.tensor([_['labels_len'] for _ in batch], dtype=torch.int)
        text_trans = [_['text_trans'] for _ in batch]

        context = [torch.tensor(_['context'], dtype=torch.long) for _ in batch]
        pre_text = [torch.tensor(_['pre_text'], dtype=torch.long) for _ in batch]
        context = torch.nn.utils.rnn.pad_sequence(context, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0)
        pre_text = torch.nn.utils.rnn.pad_sequence(pre_text, batch_first=True, padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 0)
        context_len = torch.tensor([_['context_len'] for _ in batch], dtype=torch.int)
        pre_text_len = torch.tensor([_['pre_text_len'] for _ in batch], dtype=torch.int)


        return {"text_trans" : text_trans,
                "input_ids": input_ids,
                "input_len": input_len,
                "labels": labels,
                "labels_len": labels_len,
                "context": context,
                "context_len": context_len,
                "pre_text": pre_text,
                "pre_text_len": pre_text_len,
                }

def collate_all_dialog_text(batch_s, batch_t, tokenizer=None):
    input_ids_s = torch.nn.utils.rnn.unpad_sequence(
        batch_s['input_ids'], lengths = batch_s['input_len'], batch_first=True
    )
    input_ids_t = torch.nn.utils.rnn.unpad_sequence(
        batch_t['input_ids'], lengths = batch_t['input_len'], batch_first=True
    )
    input_ids = input_ids_s + input_ids_t

    labels_s = torch.nn.utils.rnn.unpad_sequence(batch_s['labels'], batch_first=True, lengths = batch_s['labels_len'])
    labels_t = torch.nn.utils.rnn.unpad_sequence(batch_t['labels'], batch_first=True, lengths = batch_t['labels_len'])
    labels = labels_s + labels_t

    context_s = torch.nn.utils.rnn.unpad_sequence(batch_s['context'], batch_first=True, lengths = batch_s['context_len'])
    context_t = torch.nn.utils.rnn.unpad_sequence(batch_t['context'], batch_first=True, lengths = batch_t['context_len'])
    context = context_s + context_t

    pre_text_s = torch.nn.utils.rnn.unpad_sequence(batch_s['pre_text'], batch_first=True, lengths = batch_s['pre_text_len'])
    pre_text_t = torch.nn.utils.rnn.unpad_sequence(batch_t['pre_text'], batch_first=True, lengths = batch_t['pre_text_len'])
    pre_text = pre_text_s + pre_text_t


    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer is not None else 0
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    context = torch.nn.utils.rnn.pad_sequence(context, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer is not None else 0)
    pre_text = torch.nn.utils.rnn.pad_sequence(pre_text, batch_first=True, padding_value=tokenizer.pad_token_id if tokenizer is not None else 0)

    input_len = torch.tensor(batch_s['input_len'].tolist()+ batch_t['input_len'].tolist())
    labels_len = torch.tensor(batch_s['labels_len'].tolist() + batch_t['labels_len'].tolist())
    pre_text_len = torch.tensor(batch_s['pre_text_len'].tolist() + batch_t['pre_text_len'].tolist())
    context_len = torch.tensor(batch_s['context_len'].tolist() + batch_t['context_len'].tolist())
    text_trans = batch_s['text_trans'] + batch_t['text_trans']


    return {"text_trans" : text_trans,
            "input_ids": input_ids,
            "input_len": input_len,
            "labels": labels,
            "labels_len": labels_len,
            "context": context,
            "context_len": context_len,
            "pre_text": pre_text,
            "pre_text_len": pre_text_len,
            }


def load_from_textfile(textfile_path='/mnt/matylda6/ivendrame/miniLM/data/librispeech_shuf.txt'):

    with open(textfile_path, 'r', encoding='utf-8') as f:
        data = [{'text': f'{_.strip().lower()}'} for _ in f.readlines() if _.strip() and len(_.split())< 231]

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
