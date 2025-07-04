import torch
import datasets
import os
import yaml
import torchaudio
import functools
import re
import data_asr
from whisper_normalizer.english import EnglishSpellingNormalizer

def modify_commonvoice(item, ds_text_column, normalizer=None):
    new_item = {}
    new_item["text"] = item["sentence"]
    if ds_text_column == "normalized_text":
        new_item["normalized_text"] = normalize_text(new_item["text"], normalizer=normalizer)
    new_item["audio"] = item["audio"]
    audio = item['audio']['array']
    audio = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)(torch.tensor(audio, dtype=torch.float32))
    new_item["audio"]["sampling_rate"] = 16_000
    new_item["audio"]["array"] = audio.numpy()
    new_item["audio_len"] = len(new_item["audio"]["array"])
    return new_item

def modify_librispeech(item, ds_text_column, normalizer=None):
    new_item = {}
    new_item["text"] = item["text"]
    new_item["audio"] = item["audio"]
    
    #if ds_text_column == "normalized_text":
    new_item["normalized_text"] = normalizer(new_item["text"].lower())
    #audio = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)(torch.tensor(audio, dtype=torch.float32))
    new_item["audio"]["sampling_rate"] = 16_000
    new_item["audio_len"] = len(item["audio"]["array"])
    return new_item


def normalize_text(text, normalizer=None):
    text = normalizer(text) 
    return re.sub(r"[^\w\s']|(?<!\w)'|'(?!\w)", "", text).lower()

def generate_dataset(shards, ds_type, ds_text_column, normalizer=None):
    for shard_path in shards:
        print(shard_path)
        shard = datasets.load_from_disk(shard_path)
        for item in shard:
            if ds_type == "commonvoice":
                item = modify_commonvoice(item, ds_text_column, normalizer=normalizer)
            if ds_type == "librispeech_dev_clean":
                item = modify_librispeech(item, ds_text_column, normalizer = normalizer)
            if ds_type == "librispeech_dev_other":
                item = modify_librispeech(item, ds_text_column, normalizer = normalizer)

            yield {
                "text": item["text"],            
                "normalized_text": item["normalized_text"] if "normalized_text" in item else item["text"],
                "audio": item["audio"],
                "audio_len": item["audio_len"]
            }

def main():
    path = "/mnt/matylda6/ivendrame/wavlm_connector_lm/scripts/joint_text_speech/"
    dataset_path = "/mnt/scratch/tmp/ivendrame/huggingface/modules/datasets_modules/datasets/"
    splits = ["train"]
    num_proc = 4
    normalizer = EnglishSpellingNormalizer()

    with open(os.path.join(path, 'datasets.yaml'), 'r') as f:
        config_datasets = yaml.load(f, Loader=yaml.FullLoader)
    for split in splits:
        for k,v in config_datasets[split].items():
            ds_name = v["name"].split(":")[0] if ":" in v["name"] else v["name"]
            ds_subset = v["name"].split(":")[1] if ":" in v["name"] else None
            ds_split = v["split"]
            ds_text_column = v["text_column"]
            if ds_subset:
                loaded_dataset = datasets.load_dataset(ds_name, ds_subset, split = ds_split, trust_remote_code=True, num_proc=num_proc)
            else:
                loaded_dataset = datasets.load_dataset(ds_name, split = ds_split, trust_remote_code=True, num_proc=num_proc)

            shards = []
            for i in range(8):
                shard_path = os.path.join(dataset_path, f"{k}_{split}_shard_{i}.json")
                shard = loaded_dataset.shard(num_shards=8, index=i)
                shard.save_to_disk(shard_path)
                shards.append(shard_path)
            
            print("Loaded dataset:", [i for i in shards])
            new_dataset = datasets.Dataset.from_generator(generate_dataset, gen_kwargs = {"shards": shards, "ds_type": k, "ds_text_column": ds_text_column, "normalizer": normalizer}, features=datasets.Features({
                "text": datasets.Value("string"),
                "normalized_text": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000),
                "audio_len": datasets.Value("int32")}), num_proc=num_proc)
            new_dataset.save_to_disk(os.path.join(dataset_path, f"prep_{k}_{split}"), )

if __name__ == '__main__':
    main()