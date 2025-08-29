import torch
import datasets
import os
import yaml
import torchaudio
import functools
import re
import data_asr
from whisper_normalizer.english import EnglishSpellingNormalizer
import random

def modify_commonvoice(item, normalizer=None):
    new_item = {}
    new_item["text"] = item["sentence"]
    #if ds_text_column == "normalized_text":
    new_item["normalized_text"] = normalize_text(new_item["text"], normalizer=normalizer)
    new_item["audio"] = item["audio"]
    audio = item['audio']['array']
    audio = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)(torch.tensor(audio, dtype=torch.float32))
    new_item["audio"]["sampling_rate"] = 16_000
    new_item["audio"]["array"] = audio.numpy()
    new_item["audio_len"] = len(new_item["audio"]["array"])
    return new_item

def modify_voxpopuli(item, normalizer=None):
    new_item = {}
    new_item["text"] = item["normalized_text"]
    #if ds_text_column == "normalized_text":
    new_item["normalized_text"] = normalize_text(new_item["text"], normalizer=normalizer)
    new_item["audio"] = item["audio"]
    audio = item['audio']['array']
    audio = torchaudio.transforms.Resample(orig_freq=item['audio']['sampling_rate'], new_freq=16000)(torch.tensor(audio, dtype=torch.float32))
    new_item["audio"]["sampling_rate"] = 16_000
    new_item["audio"]["array"] = audio.numpy()
    new_item["audio_len"] = len(new_item["audio"]["array"])
    return new_item


def modify_librispeech(item, normalizer=None):
    new_item = {}
    new_item["text"] = item["text"]
    new_item["audio"] = item["audio"]
    
    #if ds_text_column == "normalized_text":
    new_item["normalized_text"] = normalizer(new_item["text"].lower())
    #audio = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)(torch.tensor(audio, dtype=torch.float32))
    new_item["audio"]["sampling_rate"] = 16_000
    new_item["audio_len"] = len(item["audio"]["array"])
    return new_item

def modify_fisher(item, normalizer=None):
    new_item = {}
    new_item["text"] = item["labels_str"]
    new_item["audio"] = item["audio"]
    
    #if ds_text_column == "normalized_text":
    new_item["normalized_text"] = normalizer(new_item["text"].lower())
    #audio = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)(torch.tensor(audio, dtype=torch.float32))
    new_item["audio"]["sampling_rate"] = item["audio"]["sampling_rate"]
    fs = item["audio"]["sampling_rate"]
    audio = item["audio"]["array"]
    if fs != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(torch.tensor(audio, dtype=torch.float32))
        new_item["audio"]["array"] = audio
        new_item["audio"]["sampling_rate"] = 16000

    new_item["audio_len"] = item["total_len"]*item["audio"]["sampling_rate"]
    return new_item  

def modify_spokenwoz(item, normalizer=None):
    new_item = {}
    new_item["text"] = item["text"]
    new_item["audio"] = item["audio"]
    
    #if ds_text_column == "normalized_text":
    new_item["normalized_text"] = normalizer(new_item["text"].lower())
    #audio = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)(torch.tensor(audio, dtype=torch.float32))
    new_item["audio"]["sampling_rate"] = item["audio"]["sampling_rate"]
    fs = item["audio"]["sampling_rate"]
    audio = item["audio"]["array"]
    if fs != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(torch.tensor(audio, dtype=torch.float32))
        new_item["audio"]["array"] = audio
        new_item["audio"]["sampling_rate"] = 16000

    new_item["audio_len"] = len(item["audio"]["array"])
    return new_item  

def modify_dialog_studio(item, normalizer=None):
    new_item = {}
    if random.random() >0.5:
        new_item["text"] = item["text"]
    else:
        new_item["text"] = item["agent_text"]
    
    #if ds_text_column == "normalized_text":
    new_item["normalized_text"] = normalizer(new_item["text"].lower())
    return new_item    

def modify_open_subtitles(item, normalizer=None):
    new_item = {}
    new_item['text'] = item['english']
    new_item["normalized_text"] = normalizer(new_item["text"].lower())
    return new_item    

def normalize_text(text, normalizer=None):
    text = normalizer(text) 
    return re.sub(r"[^\w\s']|(?<!\w)'|'(?!\w)", "", text).lower()

def generate_dataset(shards, ds_type, normalizer=None):
    for shard_path in shards:
        print(shard_path)
        shard = datasets.load_from_disk(shard_path)
        for item in shard:
            if ds_type == "commonvoice":
                item = modify_commonvoice(item, normalizer=normalizer)
            if ds_type.startswith("librispeech"):
                item = modify_librispeech(item, normalizer = normalizer)
            if ds_type.startswith("fisher"):
                item = modify_fisher(item, normalizer=normalizer)
            if ds_type.startswith("spokenwoz"):
                item = modify_spokenwoz(item, normalizer=normalizer)
            if ds_type == "voxpopuli":
                item = modify_voxpopuli(item, normalizer=normalizer)
        
            yield {
                "text": item["text"],            
                "normalized_text": item["normalized_text"],
                "audio": item["audio"],
                "audio_len": item["audio_len"]
            }

def generate_text_ds(shards, ds_type, normalizer=None):
    for shard_path in shards:
        print(shard_path)
        shard = datasets.load_from_disk(shard_path)
        for item in shard:
            if ds_type=="open_subtitles":
                item = modify_open_subtitles(item, normalizer=normalizer)
            if ds_type=="dialog_studio":
                item = modify_dialog_studio(item, normalizer=normalizer)
            
            yield {
                "text": item["text"],            
                "normalized_text": item["normalized_text"],
            }

def main():
    path = "/mnt/matylda6/ivendrame/wavlm_connector_lm/notebooks/test_datasets/"
    dataset_path = "/mnt/scratch/tmp/ivendrame/huggingface/modules/datasets_modules/datasets/"
    splits = ["validation"]
    num_proc = 20
    normalizer = EnglishSpellingNormalizer()

    with open(os.path.join(path, 'datasets.yaml'), 'r') as f:
        config_datasets = yaml.load(f, Loader=yaml.FullLoader)
    for split in splits:
        for k,v in config_datasets[str(split)].items():
            ds_name = v["name"].split(":")[0] if ":" in v["name"] else v["name"]
            print(ds_name)
            ds_subset = v["name"].split(":")[1] if ":" in v["name"] else None
            ds_split = v["split"] #v["text_validation"]
            ds_text_column = v["text_column"]

            if ds_name == "dialog_studio":
                for ds in os.listdir(v["path"]):
                    split = ""
                    loaded_dataset = datasets.load_from_disk(os.path.join(v["path"],ds))
                    loaded_dataset= loaded_dataset[split]
                    shards = []
                    print(f"shards {k} {split}")
                    for i in range(8):
                        shard_path = os.path.join(dataset_path, f"{k}_{split}_shard_{i}.json")
                        shard = loaded_dataset.shard(num_shards=8, index=i)
                        shard.save_to_disk(shard_path)
                        shards.append(shard_path)
                    
                    print("Loaded dataset:", [i for i in shards])
                    new_dataset = datasets.Dataset.from_generator(generate_text_ds, gen_kwargs = {"shards": shards, "ds_type": k,  "normalizer": normalizer}, features=datasets.Features({
                        "text": datasets.Value("string"),
                        "normalized_text": datasets.Value("string"),
                        }), num_proc=num_proc)
                    new_dataset.save_to_disk(os.path.join(dataset_path, f"prep_{k}_text_train_{ds}"), )
            
            elif k == "open_subtitles":
                ds_split = "validation"
                print(ds_name, ds_subset, ds_split)
                loaded_dataset = datasets.load_dataset(ds_name, "all", split = ds_split, trust_remote_code=True, num_proc=4)
                loaded_dataset = loaded_dataset
                shards = []
                print(f"shards {k} {ds_split}")
                for i in range(1):
                    shard_path = os.path.join(dataset_path, f"{k}_{ds_split}_shard_{i}.json")
                    shard = loaded_dataset.shard(num_shards=1, index=i)
                    shard.save_to_disk(shard_path)
                    shards.append(shard_path)
                new_dataset = datasets.Dataset.from_generator(generate_text_ds, gen_kwargs = {"shards": shards,"ds_type": k, "normalizer": normalizer}, features=datasets.Features({
                "text": datasets.Value("string"),
                "normalized_text": datasets.Value("string"),
                }), num_proc=num_proc)
                new_dataset.save_to_disk(os.path.join(dataset_path, f"prep_{k}_{ds_split}"), )
            
            else:
                if "path" in v:
                    print("Loading from path", v['path'])
                    loaded_dataset = datasets.load_from_disk(os.path.join(v['path']))

                    #loaded_dataset = loaded_dataset[ds_split]
                else:
                    print("Loading from huggingface", ds_name, ds_subset, ds_split)
                    if ds_subset:
                        loaded_dataset = datasets.load_dataset(ds_name, ds_subset, split = ds_split, trust_remote_code=True, num_proc=num_proc)#, download_mode='force_redownload' )#download_mode='force_redownload'
                    else:
                        loaded_dataset = datasets.load_dataset(ds_name, split = ds_split, trust_remote_code=True, num_proc=num_proc)#,download_mode='force_redownload') #  download_mode='force_redownload'

                shards = []
                print(f"shards {k} {split}")
                for i in range(8):
                    shard_path = os.path.join(dataset_path, f"{k}_{split}_shard_{i}.json")
                    shard = loaded_dataset.shard(num_shards=10, index=i)
                    shard.save_to_disk(shard_path)
                    shards.append(shard_path)
                
                print("Loaded dataset:", [i for i in shards])
                new_dataset = datasets.Dataset.from_generator(generate_dataset, gen_kwargs = {"shards": shards, "ds_type": k, "normalizer": normalizer}, features=datasets.Features({
                    "text": datasets.Value("string"),
                    "normalized_text": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "audio_len": datasets.Value("int32")}), num_proc=num_proc)
                new_dataset.save_to_disk(os.path.join(dataset_path, f"prep_{k}_{split}"), )
                print("Saved dataset to", os.path.join(dataset_path, f"prep_{k}_{split}"))

if __name__ == '__main__':
    main()