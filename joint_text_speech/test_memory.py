import os
import argparse
import functools
import time
import datetime

import torch
import tqdm
import yaml
import re

import data_asr
import models_asr
from safe_gpu import safe_gpu

import jiwer
from torch.utils.tensorboard import SummaryWriter
import transformers
import datasets
import accelerate

#print environment variables
print("HF datasets cache", os.environ.get('HF_DATASETS_CACHE', ''))
print("hf home", os.environ.get('HF_HOME', ''))

def lr_lambda_linear_with_min_lr(step, args):
    peak_lr = args.peak_lr
    min_lr = peak_lr * args.min_lr_ratio
    if step < args.warmup_steps:
        return step / args.warmup_steps
    slope = (peak_lr - min_lr) / (args.max_steps - args.warmup_steps)
    intercept = peak_lr - slope * args.warmup_steps
    current_lr = - slope * step + intercept
    return max(current_lr, min_lr) / peak_lr


def lr_lambda_cosine_with_min_lr(step, args):
    peak_lr = args.peak_lr
    min_lr = peak_lr * args.min_lr_ratio
    if step < args.warmup_steps:
        return step / args.warmup_steps
    slope = (peak_lr - min_lr) / 2
    intercept = min_lr
    current_lr = - slope * (1 + torch.cos(
        torch.tensor(step - args.warmup_steps) / (
                args.max_steps - args.warmup_steps) * torch.pi)).item() + intercept
    return current_lr / peak_lr


def lr_lambda(step, args):
    if args.scheduler == 'linear':
        return lr_lambda_linear_with_min_lr(step, args)
    return lr_lambda_cosine_with_min_lr(step, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type = str, default='',
                        help='name of the experiment')
    parser.add_argument('--no_cuda', action='store_true',
                        help='do not use cuda even if it is available')
    parser.add_argument('--num_heads', type=int, default=16,
                        help='number of attention heads')
    parser.add_argument('--ff_size', type=int, default=2048,
                        help='feed forward size in transformer model')
    parser.add_argument('--num_layers', type=int, default=10,
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

    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                        help='number of workers for dataloader')

    parser.add_argument('--validation_steps', type=int, default=1000,
                        help='number of steps between validation runs')
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='number of steps between saving model checkpoints')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='number of steps for linear warmup')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='number of validation runs to wait before early stopping')
    parser.add_argument('--max_steps', type=int, default=50000,
                        help='maximum number of steps to train for')
    parser.add_argument('--scheduler', type=str, default='linear', choices=['linear', 'cosine'],
                        help='learning rate scheduler to use')
    parser.add_argument('--per_device_train_asr_batch_size', type=int, default=8,
                        help='asr batch size per device for training')
    parser.add_argument('--per_device_train_text_batch_size', type=int, default=8,
                        help='text batch size per device for training')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8,
                        help='batch size per device for evaluation')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='number of steps to accumulate gradients for')

    parser.add_argument('--peak_lr', type=float, default=1e-4,
                        help='peak learning rate')
    parser.add_argument('--min_lr_ratio', type=float, default=0.4,
                        help='minimum learning rate as a ratio of peak learning rate')

    parser.add_argument('--bos_token', type=str, default=None, 
                        help="bos_token for lm, if None takes model.tokenizer.eos_token")
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--max_grad_value', type=float, default=5.0,
                        help='maximum absolute gradient value for gradient clipping')
    parser.add_argument('--decode_checkpoint', type=str, default='best',
                        help='checkpoint to use for decoding')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='output directory to save model checkpoints and generated outputs')
    parser.add_argument('--resume', action='store_true',
                        help='resume training from latest')

    parser.add_argument('--log_dir', type=str, default=None,
                        help='log runs dir')

    parser.add_argument('--lm_dir', type=str, default=None,
                        help='lm dir')
    parser.add_argument('--datasets_config', type=str, default=None,
                        help='datasets yaml file path')
    parser.add_argument('--encoder_model_name', type=str, default='microsoft/wavlm-large',
                        help='encoder model name')
    parser.add_argument('--lm_model_name', type=str, default="allenai/OLMo-1B-hf",
                        help='lm name as string')

    parser.add_argument('--pretrained_dir', type=str, default=None,
                        help='pretrained connector and encoder dir: takes models from /latest')
    parser.add_argument('--train_encoder', action='store_true',
                        help='train encoder')
    parser.add_argument('--train_lm', action='store_true',
                        help='train lm')
    parser.add_argument('--text_input', action='store_true',
                        help='use text and audio input for training, if False use audio input')
    parser.add_argument('--train_with_asr_text', action='store_true',
                        help='use text input from asr dataset for training, aligned with speech training, if False use text input from text dataset')
    parser.add_argument('--mse_loss', action='store_true',
                        help='only use in case asr text is true and text input is true, this will use mse loss between hidden states of text encoder and connector')
    parser.add_argument('--encoder_eval', action='store_true',
                        help='eval mode for encoder during training, if False train mode')
    parser.add_argument('--connector_eval', action='store_true',
                        help='eval mode for connector during training, if False train mode')
    parser.add_argument('--lm_eval', action='store_true',
                        help='eval mode for lm during training, if False train mode')
    parser.add_argument('--text_encoder_eval', action='store_true',
                        help='eval mode for text encoder during training, if False train mode')

    parser.add_argument('--text_weight', type=float, default=0.5,
                        help='weight for text encoder loss')
    parser.add_argument('--mask_rate', type=float, default=0.5,
                        help='mask rate for text encoder')
    parser.add_argument('--text_encoder_dim', type=int, default=512,
                        help='text encoder dimension')
    

    args = parser.parse_args()
    #torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision('high')

    def place_holder_fn(x):
        pass

    accelerator = accelerate.Accelerator()
    print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    _x = torch.tensor([1.0], device=accelerator.device, dtype=torch.bfloat16)

    # device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    device = accelerator.device
    print(f'Using device: {device} - index: {accelerator.process_index}')

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None
    for arg in vars(args):
        accelerator.print(f'{arg}: {getattr(args, arg)}')


    ## MODEL
    bos_token = args.bos_token
    lm_model_name = args.lm_model_name
    lm = transformers.AutoModelForCausalLM.from_pretrained(args.lm_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map=accelerator.device)
    lm_config = {"model_name": lm_model_name,
                 "bos_token": bos_token if bos_token is not None else lm.config.eos_token_id}
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.lm_model_name, trust_remote_code=True)

    if args.resume:
        accelerator.print('Resuming from checkpoint')
        connector = models_asr.Connector.load_from_dir(os.path.join(args.output_dir, 'latest'), device)
        #connector = connector.to(torch.bfloat16)
        connector = connector.to(device)
        if args.text_input:
            text_encoder = models_asr.TextEncoder.load_from_dir(os.path.join(args.output_dir, 'latest'), device)
            text_encoder = text_encoder.to(device)
        if args.train_encoder:
            encoder = models_asr.WavLMWrapper.load_from_dir(os.path.join(args.output_dir, 'latest'), device)
            #encoder= encoder.to(torch.bfloat16)
            encoder = encoder.to(device)

        if os.path.isfile(os.path.join(args.output_dir, 'lm_config.yaml')):
            with open(os.path.join(args.output_dir, 'lm_config.yaml'), 'r') as f:
                lm_config = yaml.load(f, Loader=yaml.FullLoader)
            lm_model_name = lm_config['model_name']
            bos_token = lm_config['bos_token']
        
        with open(os.path.join(args.output_dir, 'latest', 'metrics.yaml'), 'r') as f:
            all_metrics = yaml.safe_load(f)
        best_val_loss = min([m['val_loss'] for m in all_metrics])
    
    if args.pretrained_dir is not None:
        accelerator.print(f'Loading pretrained connector and encoder from {args.pretrained_dir}')
        connector = models_asr.Connector.load_from_dir(os.path.join(args.pretrained_dir, 'latest'), device)
        encoder = models_asr.WavLMWrapper.load_from_dir(os.path.join(args.pretrained_dir, 'latest'), device)
    else:
        encoder = models_asr.WavLMWrapper(args.encoder_model_name)
        connector = models_asr.Connector(encoder.encoder.config.hidden_size, lm.config.hidden_size, num_heads = 4, ff_size = 4*encoder.encoder.config.hidden_size)
    if args.text_input:
        text_encoder = models_asr.TextEncoder(args.mask_rate, tokenizer.vocab_size, args.text_encoder_dim, encoder.encoder.config.hidden_size, num_heads=4, ff_size = 4*(args.text_encoder_dim), pad_token=tokenizer.pad_token_id)
        
    #connector = connector.to(torch.bfloat16)
    #text_encoder = text_encoder.to(torch.bfloat16)
    if args.text_input:
        model = models_asr.EncoderConnectorLmWithPretrainedLm(encoder, connector, lm, tokenizer, text_encoder)
    else:
        model = models_asr.EncoderConnectorLmWithPretrainedLm(encoder, connector, lm, tokenizer)
    model = model.to(device)
    
    if not args.train_encoder:
        model.freeze_encoder()
    if not args.train_lm:
        model.freeze_lm()

    # parameters
    parameters_to_train = list(model.connector.parameters())
    if args.text_input:
        parameters_to_train += list(model.text_encoder.parameters())
    if args.train_encoder:
        parameters_to_train += list(model.encoder.parameters())
    if args.train_lm:
        parameters_to_train += list(model.lm.parameters())

    accelerator.print(model.config)
    accelerator.print(f'Number of total parameters: {sum(p.numel() for p in model.parameters())}')
    accelerator.print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')


    ## DATASETS
    # validation (dict of datasets)
    dset_valid = data_asr.load_from_config('validation', args.datasets_config) # dictonary ["commonvoice": ..., "librispeech": ... etc]
    val_subsets = {}
    for k, v in dset_valid.items():
        val_subsets[k] = data_asr.AudioDatasetHF({k:v}, tokenizer, bos_token)
    # train dataset
    dset_train_s = data_asr.load_from_config('train', args.datasets_config)
    train_asr_data = data_asr.AudioDatasetHF(dset_train_s, tokenizer, bos_token)
    if args.text_input:
        dset_train_t = data_asr.load_from_config('text_train', args.datasets_config)
        train_text_data = data_asr.TextDatasetHF(dset_train_t, tokenizer, bos_token)
    datasets_config = {"speech_train": dset_train_s,
                          "text_train": dset_train_t if args.text_input else None,
                          "validation": dset_valid}

    ## Optimizer, scheduler, data loaders
    if args.resume:
        optimizer = torch.optim.AdamW(parameters_to_train, lr=args.peak_lr, weight_decay=args.weight_decay)
        lr_lambda_partial = functools.partial(lr_lambda, args=args)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_partial)
        start_step = all_metrics[-1]['step']
        optimizer.load_state_dict(torch.load(os.path.join(args.output_dir, 'latest', 'optimizer.pt')))
        scheduler.load_state_dict(torch.load(os.path.join(args.output_dir, 'latest', 'scheduler.pt')))
        accelerator.print(f'Resuming from step {start_step}')
    else:
        optimizer = torch.optim.AdamW(parameters_to_train, lr=args.peak_lr, weight_decay=args.weight_decay)
        lr_lambda_partial = functools.partial(lr_lambda, args=args)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_partial)
        start_step = 0

    torch.manual_seed(24)
    # training dataloader
    collate_fn_asr = functools.partial(data_asr.collate_fn_asr, tokenizer=tokenizer)
    train_asr_loader = torch.utils.data.DataLoader(train_asr_data, batch_size=args.per_device_train_text_batch_size,
                                               collate_fn=collate_fn_asr,
                                               shuffle=True,
                                               num_workers=args.dataloader_num_workers)
                                          #     drop_last=True)
     
    if args.text_input:
        collate_fn_text = functools.partial(data_asr.collate_fn_text, tokenizer=tokenizer)
        train_text_loader = torch.utils.data.DataLoader(train_text_data, batch_size=args.per_device_train_asr_batch_size,
                                               collate_fn=collate_fn_text,
                                               shuffle=True,
                                               num_workers=args.dataloader_num_workers)                                    
    # validation dataloaders    
    validation_loaders = {}
    for name, ds in val_subsets.items():
        validation_loader = torch.utils.data.DataLoader(ds, batch_size=args.per_device_eval_batch_size,
                                                    collate_fn=collate_fn_asr,
                                                    num_workers=args.dataloader_num_workers)
        validation_loaders[name] = validation_loader

    # accelerator preparation
    model, optimizer, train_asr_loader, train_text_loader = accelerator.prepare(
        model, optimizer, train_asr_loader, train_text_loader
    )
    for k, v in validation_loaders.items():
        validation_loaders[k] = accelerator.prepare(v)
    train_asr_loader.dataset.is_fake = True
    train_text_loader_iter = iter(train_text_loader)
    train_asr_loader_iter = iter(train_asr_loader)
    
    
    if args.resume:
        train_asr_loader.dataset.is_fake = True
        train_asr_loader_iter = iter(train_asr_loader)
        found = False
        current_step = 0
        while True:
            for _ in train_asr_loader:
                current_step += 1
                if current_step == start_step:
                    train_asr_loader.dataset.is_fake = False
                    train_asr_loader_iter = iter(train_asr_loader)
                    found = True
                    break
            if found:
                break
    

    ## TRAINING LOOP
    accelerator.print(f'Starting step: {start_step}, max steps: {args.max_steps}')
    train_loss = 0
    training_acc = 0
    training_count = 0
    best_wer = float('inf')
    all_metrics = []

    for j in tqdm.tqdm(range(start_step+1, args.max_steps + 1), disable=not accelerator.is_main_process):
        optimizer.zero_grad()
        if args.connector_eval:
            accelerator.unwrap_model(model).connector.eval()
        else:
            accelerator.unwrap_model(model).connector.train()
        if args.encoder_eval:
            accelerator.unwrap_model(model).encoder.eval()
        else:
            accelerator.unwrap_model(model).encoder.train()
        if args.lm_eval:
            accelerator.unwrap_model(model).lm.eval()
        else:
            accelerator.unwrap_model(model).lm.train()
        if args.text_input:
            if args.text_encoder_eval:
                accelerator.unwrap_model(model).text_encoder.eval()
            else:
                accelerator.unwrap_model(model).text_encoder.train()
        
        ## ACCUMULATION STEPS
        for k in range(args.accumulation_steps):
            try:
                batch_s = next(train_asr_loader_iter)
            except StopIteration:
                train_asr_loader_iter = iter(train_asr_loader)
                batch_s = next(train_asr_loader_iter)
            for key, value in batch_s.items():
                if isinstance(value, torch.Tensor):
                    batch_s[key] = batch_s[key].to(device)
            x_s = batch_s
            x_s_clone = {}
            for key, value in x_s.items():
                if isinstance(value, torch.Tensor):
                    x_s_clone[key] = value.clone()
                else:
                    x_s_clone[key] = value
            y_s = batch_s['labels'].to(device)
            if args.text_input:
                try:
                    batch_t = next(train_text_loader_iter)
                except StopIteration:
                    train_text_loader_iter = iter(train_text_loader)
                    batch_t = next(train_text_loader_iter)
                for key, value in batch_t.items():
                    if isinstance(value, torch.Tensor):
                        batch_t[key] = batch_t[key].to(device)
                x_t = batch_t
                y_t = batch_t['labels'].to(device)

            if k < args.accumulation_steps - 1:
                with model.no_sync():
                    with torch.autocast(enabled = True, device_type = "cuda", dtype= torch.bfloat16):
                        z_s, loss_s, acc_s, hidden_states_s = model(x_s, is_text = False, output_hidden_states=True)
                        if args.text_input:
                            z_t, loss_t, acc_t = model(x_t, is_text = True, output_hidden_states=False)
                            loss = loss_s + args.text_weight*loss_t
                            if args.train_with_asr_text:
                                z_s_t, loss_s_t, acc_s_t, hidden_states_t = model(x_s_clone, is_text = True, output_hidden_states=True)
                                if args.mse_loss:
                                    z_s_t, loss_s_t, acc_s_t, hidden_states_t = model(x_s_clone, is_text = True, output_hidden_states=True)
                                    mse_loss = ((hidden_states_s - hidden_states_t)**2).to(device) #bxNxdims
                                    lengths = x_s_clone['input_len'].to(device)
                                    mask = models_asr.lengths_to_mask(lengths).to(device) #bxdims
                                    mse_loss = mse_loss*mask[:, :, None] #bxNxdims
                                    mse_loss = mse_loss.sum() / (mask.sum() * hidden_states_s.shape[-1])
                                    loss += mse_loss
                                else:
                                    loss += args.text_weight*loss_s_t
                        else:
                            loss = loss_s
                    (loss/args.accumulation_steps).backward()
            
            else:
                with torch.autocast(enabled = True, device_type = "cuda", dtype= torch.bfloat16):
                    z_s, loss_s, acc_s, hidden_states_s = model(x_s, is_text = False, output_hidden_states=True)
                    if args.text_input:
                        z_t, loss_t, acc_t = model(x_t, is_text = True, output_hidden_states=False)
                        loss = loss_s + args.text_weight*loss_t
                        if args.train_with_asr_text:
                            z_s_t, loss_s_t, acc_s_t, hidden_states_t = model(x_s_clone, is_text = True, output_hidden_states=True)
                            if args.mse_loss:
                                z_s_t, loss_s_t, acc_s_t, hidden_states_t = model(x_s_clone, is_text = True, output_hidden_states=True)
                                mse_loss = ((hidden_states_s - hidden_states_t)**2).to(device) #bxNxdims
                                lengths = x_s_clone['input_len'].to(device)
                                mask = models_asr.lengths_to_mask(lengths).to(device) #bxdims
                                mse_loss = mse_loss*mask[:, :, None] #bxNxdims
                                mse_loss = mse_loss.sum() / (mask.sum()*hidden_states_s.shape[-1])
                                loss += mse_loss
                            else:
                                loss += args.text_weight*loss_s_t
                    else:
                        loss = loss_s
                
                (loss/args.accumulation_steps).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_value)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        loss_s = accelerator.gather(loss_s).mean().item()
        acc_s = accelerator.gather(acc_s).mean().item()
        loss = accelerator.gather(loss).mean().item()
        if args.text_input:
            loss_t = accelerator.gather(loss_t).mean().item()
            acc_t = accelerator.gather(acc_t).mean().item()
            if args.train_with_asr_text:
                loss_s_t = accelerator.gather(loss_s_t).mean().item()
                acc_s_t = accelerator.gather(acc_s_t).mean().item()
                if args.mse_loss:
                    mse_loss = accelerator.gather(mse_loss).mean().item()
            else:
                loss_s_t = None
                acc_s_t = None
                mse_loss = None

        train_loss += loss
        training_acc += acc_s
        training_count += 1

        # Log training metrics
        if accelerator.is_main_process:
            writer.add_scalar("Loss/train", loss_s, j)
            if args.text_input:
                writer.add_scalar("Loss/text_rain", loss_t, j)
                writer.add_scalar("Accuracy/text_train", acc_t, j)
                if args.train_with_asr_text:
                    writer.add_scalar("Loss/text_asr_train", loss_s_t, j)
                    writer.add_scalar("Accuracy/text_asr_train", acc_s_t, j)
                    if args.mse_loss:
                        writer.add_scalar("Loss/mse_loss", mse_loss, j)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], j)
            writer.add_scalar("Accuracy/train", acc_s, j)
            
            writer.add_scalar("Data/Audio Length", batch_s['audio_len'].float().mean().item(), j)
            writer.add_scalar("Data/Text Length", batch_s['input_len'].float().mean().item(), j)
            writer.add_scalar("Data/Text (TextEncoder) Length", batch_t['input_len'].float().mean().item(), j)
            writer.add_scalar("Data/Batch Size", batch_s['audio'].shape[0], j)
            writer.add_scalar("Loss/Gradient Norm", grad_norm.item(), j)

        ## VALIDATION LOOP
        if j > 0 and j % args.validation_steps == 0:
            start_time = time.time()
            model.eval()
            for ds_name, val_dataset_loader in validation_loaders.items():
                with torch.no_grad():
                    val_loss = 0
                    val_acc = 0
                    val_count = 0
                    all_transcriptions = []
                    all_references = []

                    for val_batch in tqdm.tqdm(val_dataset_loader, disable=not accelerator.is_main_process):
                        val_x = val_batch
                        for key, value in val_x.items():
                            if isinstance(value, torch.Tensor):
                                val_x[key] = val_x[key].to(device)
                        val_y = val_batch['labels'].to(device)

                        with torch.autocast(enabled = True, device_type = "cuda", dtype= torch.bfloat16):
                            z, loss, acc = model(val_x, is_text = False, output_hidden_states=False)
                        val_loss += (loss).mean().item()
                        val_acc += acc.mean().item()
                        val_count += 1

                        with torch.autocast(enabled = True, device_type = "cuda", dtype= torch.bfloat16):
                            text_x = accelerator.unwrap_model(model).generate(val_x, 100, bos_token = bos_token)
                        text_y = val_batch['text_trans']

                        all_transcriptions += text_x
                        all_references += text_y

                output_wer = jiwer.process_words(all_references, all_transcriptions)
                insertions = output_wer.insertions
                deletions = output_wer.deletions
                substitutions = output_wer.substitutions
                wer = output_wer.wer
                cer = jiwer.cer(all_references, all_transcriptions)
                num_words = sum(len(t.split()) for t in all_transcriptions)
                num_chars = sum(len(t) for t in all_transcriptions)

                metrics = torch.tensor([insertions, deletions, substitutions, wer * num_words, cer * num_chars, num_words, num_chars], device=device)
                current_time = time.time()
                performance = (current_time - start_time) / 60  # in minutes
                print(f"Validation on {ds_name} ended at {datetime.datetime.now()} for process {accelerator.process_index}, took {performance:.2f} minutes")
                accelerator.wait_for_everyone()
                metrics = accelerator.reduce(metrics, reduction='sum')
                insertions, deletions, substitutions, wer, cer, num_words, num_chars = metrics.tolist()
                wer = wer / num_words
                cer = cer / num_chars

                # Log validation metrics
                if accelerator.is_main_process:
                    accelerator.print(ds_name)
                    accelerator.print(f'WER: {wer}, Insertions: {insertions}, Deletions: {deletions}, Substitutions: {substitutions}')
                
                    full_trt = [[r, t_] for r, t_ in zip(all_references, all_transcriptions)]
                    #full_trt = sorted(full_trt, key=lambda x: x[0])
                    to_write = full_trt[:50]
                    to_write = [f'| {x[0]} | {x[1]} |\n' for x in to_write]
                    # Prepend header
                    to_write = ['| Reference | Transcription |\n',
                    '|------------|---------------|\n'] + to_write
                    to_write = ''.join(to_write)
                    writer.add_text('WER', to_write, j)
                    #writer.add_text('WER', f'WER: {wer}, Insertions: {insertions}, Deletions: {deletions}, Substitutions: {substitutions}', j)
                    val_loss = val_loss / val_count
                    writer.add_scalar(f"Loss/validation_{ds_name}", val_loss, j)
                    val_acc = val_acc / val_count                         
                    writer.add_scalar(f"Accuracy/validation_{ds_name}", val_acc, j)
                    writer.add_scalar(f"WER/wer_{ds_name}", wer, j)
                    writer.add_scalar(f"WER/insertions_{ds_name}", insertions, j)
                    writer.add_scalar(f"WER/deletions_{ds_name}", deletions, j)
                    writer.add_scalar(f"WER/substitutions_{ds_name}", substitutions, j)
                    writer.add_scalar(f"WER/cer_{ds_name}", cer, j)

            train_loss = train_loss / training_count
            training_acc = training_acc / training_count
            if accelerator.is_main_process:
                for _k, _p in accelerator.unwrap_model(model).named_parameters():
                    if _p.requires_grad:
                        writer.add_histogram(f"Parameters/{_k}", _p.detach().cpu().float().numpy(), j)            
            logging_dict = {
                'step': j,
                'dataset' : ds_name,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': training_acc,
                'val_acc': val_acc,
                'wer' : wer,
                'learning_rate': optimizer.param_groups[0]['lr'],
                }
            all_metrics.append(logging_dict)
            accelerator.print(logging_dict)
            train_loss = 0
            training_acc = 0
            training_count = 0

            if accelerator.is_main_process:
                accelerator.unwrap_model(model).connector.save_to_directory(os.path.join(args.output_dir, 'latest'))
                if args.text_input:
                    accelerator.unwrap_model(model).text_encoder.save_to_directory(os.path.join(args.output_dir, 'latest'))
                if args.train_encoder:
                    accelerator.unwrap_model(model).encoder.save_to_directory(os.path.join(args.output_dir, 'latest'))
                yaml.dump(all_metrics, open(os.path.join(args.output_dir, 'latest', 'metrics.yaml'), 'w'))

                # save otpimizer state
                torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'latest', 'optimizer.pt'))
                # save scheduler state
                torch.save(scheduler.state_dict(), os.path.join(args.output_dir, 'latest', 'scheduler.pt'))

                if wer < best_wer:
                    best_wer = wer
                    accelerator.unwrap_model(model).connector.save_to_directory(os.path.join(args.output_dir, 'best'))
                    if args.text_input:
                        accelerator.unwrap_model(model).text_encoder.save_to_directory(os.path.join(args.output_dir, 'best'))
                    if args.train_encoder:
                        accelerator.unwrap_model(model).encoder.save_to_directory(os.path.join(args.output_dir, 'best'))
                    yaml.dump(all_metrics, open(os.path.join(args.output_dir, 'best', 'metrics.yaml'), 'w'))
            recent_wer = [m['wer'] for m in all_metrics[-args.early_stopping_patience:]]
            all_wers = [m['wer'] for m in all_metrics]
            if len(all_wers) > args.early_stopping_patience and all(v > min(all_wers) for v in recent_wer):
                accelerator.print('Early stopping')
                break

        if j % args.save_steps == 0 and accelerator.is_main_process:
            accelerator.unwrap_model(model).connector.save_to_directory(os.path.join(args.output_dir, f'checkpoint_{j}'))
            if args.text_input:
                accelerator.unwrap_model(model).text_encoder.save_to_directory(os.path.join(args.output_dir, f'checkpoint_{j}'))
            if args.train_encoder:
                accelerator.unwrap_model(model).encoder.save_to_directory(os.path.join(args.output_dir, f'checkpoint_{j}'))
            yaml.dump(all_metrics, open(os.path.join(args.output_dir, f'checkpoint_{j}', 'metrics.yaml'), 'w'))
            yaml.dump(lm_config, open(os.path.join(args.output_dir, 'lm_config.yaml'), 'w'))
            yaml.dump(datasets_config, open(os.path.join(args.output_dir, 'datasets_config.yaml'), 'w'))

        if accelerator.is_main_process:
            writer.flush()
            
    accelerator.print('Training complete!')


if __name__ == '__main__':
    main()
