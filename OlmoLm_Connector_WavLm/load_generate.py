import models_asr
import data_asr

import torch
import os
import argparse

import transformers
import functools
import yaml

from safe_gpu import safe_gpu

import tqdm
import jiwer
from torch.utils.tensorboard import SummaryWriter

safe_gpu.claim_gpus()
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type = str, default='',
                    help='name of the experiment')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda even if it is available')
parser.add_argument('--hidden_size', type=int, default=2048,
                    help='hidden size of transformer model')
parser.add_argument('--num_heads', type=int, default=12,
                    help='number of attention heads')
parser.add_argument('--ff_size', type=int, default=2048,
                    help='feed forward size in transformer model')
parser.add_argument('--num_layers', type=int, default=12,
                    help='number of layers in transformer model')
parser.add_argument('--max_len', type=int, default=10000,
                    help='maximum allowed length of input sequence')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout probability in transformer model')

parser.add_argument('--bpe_num_sentences', type=int, default=100_000,
                    help='number of sentences to train BPE on')
parser.add_argument('--bpe_vocab_size', type=int, default=4096,
                    help='vocab size of BPE')

# dataset arguments
# Will use shakespeare dataset, uncomment later to use wikitext
# parser.add_argument('--dataset', type=str, default='Salesforce/wikitext',
#                     help='dataset to use')
# parser.add_argument('--dataset_name', type=str, default='wikitext-103-v1',
#                     help='name of the dataset')
# parser.add_argument('--train_split', type=str, default='train',
#                     help='split to use for training')
# parser.add_argument('--valid_split', type=str, default='validation',
#                     help='split to use for validation')
# parser.add_argument('--test_split', type=str, default='test',
#                     help='split to use for testing')

parser.add_argument('--dataloader_num_workers', type=int, default=2,
                    help='number of workers for dataloader')

parser.add_argument('--validation_steps', type=int, default=1,
                    help='number of steps between validation runs')
parser.add_argument('--save_steps', type=int, default=1000,
                    help='number of steps between saving model checkpoints')
parser.add_argument('--warmup_steps', type=int, default=1000,
                    help='number of steps for linear warmup')
parser.add_argument('--early_stopping_patience', type=int, default=3,
                    help='number of validation runs to wait before early stopping')
parser.add_argument('--max_steps', type=int, default=50000,
                    help='maximum number of steps to train for')
parser.add_argument('--scheduler', type=str, default='linear', choices=['linear', 'cosine'],
                    help='learning rate scheduler to use')
parser.add_argument('--per_device_train_batch_size', type=int, default=8,
                    help='batch size per device for training')
parser.add_argument('--per_device_eval_batch_size', type=int, default=8,
                    help='batch size per device for evaluation')
parser.add_argument('--accumulation_steps', type=int, default=4,
                    help='number of steps to accumulate gradients for')

parser.add_argument('--peak_lr', type=float, default=1e-4,
                    help='peak learning rate')
parser.add_argument('--min_lr_ratio', type=float, default=0.4,
                    help='minimum learning rate as a ratio of peak learning rate')

parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--max_grad_value', type=float, default=5.0,
                    help='maximum absolute gradient value for gradient clipping')
parser.add_argument('--decode_checkpoint', type=str, default='best',
                    help='checkpoint to use for decoding')
parser.add_argument('--output_dir', type=str, default='/mnt/matylda6/ivendrame/wavlm_connector_lm/experiments/pretrained_lm/wavlmBase_EncConnTrain_eval/outputs',
                    help='output directory to save model checkpoints and generated outputs')
parser.add_argument('--resume', action='store_true',
                    help='resume training from latest')

parser.add_argument('--train_dir', type=str, default="/mnt/matylda6/ivendrame/data/librispeech_asr_data/trans_shuf.txt",
                    help='train dataset dir')
parser.add_argument('--val_dir', type=str, default="/mnt/matylda6/ivendrame/data/librispeech_asr_data_dev_test/all_dev.txt",
                    help='val dataset dir')
parser.add_argument('--log_dir', type=str, default="/mnt/matylda6/ivendrame/wavlm_connector_lm/tensorboard/connector_runs/generate_EncConnector_wavlmBase" ,
                    help='log runs dir')
parser.add_argument('--audio_dir', type=str, default="/mnt/matylda6/ivendrame/data/librispeech_asr_data_all_flacs",
                    help='audio dataset dir')
parser.add_argument('--tokenizer_dir', type=str, default="/mnt/matylda6/ivendrame/wavlm_connector_lm/tokenizer",
                    help='tokenizer dir')
parser.add_argument('--lm_dir', type=str, default=False,
                    help='lm dir')

parser.add_argument('--train_encoder', action='store_true',
                    help='train encoder')
parser.add_argument('--train_lm', action='store_true',
                    help='train lm')



args = parser.parse_args()
#model = models_asr.Transformer.load_from_dir(r"LLM_test6/latest", device)
torch.set_float32_matmul_precision('high')

tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf", trust_remote_code=True)

val_file = args.val_dir
audio_dir = args.audio_dir
dset_valid = data_asr.load_from_transcript(val_file, audio_dir)
validation_data = data_asr.AsrDataset(dset_valid, tokenizer, args.audio_dir)
collate_fn = functools.partial(data_asr.collate_fn_asr, tokenizer=tokenizer)    
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.per_device_eval_batch_size,
                                                collate_fn=collate_fn,
                                                num_workers=args.dataloader_num_workers)
encoder = models_asr.WavLMWrapper()
connector = models_asr.Connector(encoder.encoder.config.hidden_size, args.hidden_size, num_heads = 4, ff_size = 4*encoder.encoder.config.hidden_size)
connector = connector.to(torch.bfloat16)
lm = transformers.AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B-hf", trust_remote_code=True,
                                                        torch_dtype=torch.bfloat16,
                                                        attn_implementation='flash_attention_2',
                                                        )

#lm = models_asr.Transformer.load_from_dir(args.lm_dir, device)

model = models_asr.EncoderConnectorLmWithPretrainedLm(encoder, connector, lm, tokenizer)
model = model.to(device)
model = model.to(torch.bfloat16)

model.connector = model.connector.load_from_dir(os.path.join(args.output_dir, 'best'), device)
model.connector = model.connector.to(torch.bfloat16)
model.connector = model.connector.to(device)

model.encoder = model.encoder.load_from_dir(os.path.join(args.output_dir, 'best'), encoder.config, device)
model.encoder = model.encoder.to(torch.bfloat16)
model.encoder = model.encoder.to(device)

# Generate 5 paragraphs of up to 100 tokens each from the model and save into a file
model.eval()
with torch.no_grad():
    val_loss = 0
    val_acc = 0
    val_count = 0
    all_transcriptions = []
    all_references = []
    j = 1000
    for val_batch in tqdm.tqdm(validation_loader,):
        val_x = val_batch
        for key, value in val_x.items():
            if isinstance(value, torch.Tensor):
                val_x[key] = val_x[key].to(device)
        
        val_y = val_batch['labels'].to(device)
        #val_z = model(val_x)
        
        with torch.autocast(enabled = True, device_type = device.type, dtype= torch.bfloat16):
            text_x = model.generate(val_x, 100, padding='left')
            #print(text_x)
        text_y = val_batch['text_trans']
        all_transcriptions += text_x
        all_references += text_y

    writer = SummaryWriter(log_dir=args.log_dir)

    output_wer = jiwer.process_words(all_references, all_transcriptions)
    insertions = output_wer.insertions
    deletions = output_wer.deletions
    substitutions = output_wer.substitutions
    wer = output_wer.wer
    cer = jiwer.cer(all_references, all_transcriptions)

    full_trt = [[r, t_] for r, t_ in zip(all_references, all_transcriptions)]
    #full_trt = sorted(full_trt, key=lambda x: x[0])
    to_write = full_trt[:50]
    to_write = [f'| {x[0]} | {x[1]} |\n' for x in to_write]
    # Prepend header
    to_write = ['| Reference | Transcription |\n',
    '|------------|---------------|\n'] + to_write
    to_write = ''.join(to_write)
    writer.add_text('WER', to_write, j)
    writer.add_scalar("WER/wer", wer, j)
    writer.add_scalar("WER/insertions", insertions, j)
    writer.add_scalar("WER/deletions", deletions, j)
    writer.add_scalar("WER/substitutions", substitutions, j)
    writer.add_scalar("WER/cer", cer, j)

    writer.flush()
    print(to_write)