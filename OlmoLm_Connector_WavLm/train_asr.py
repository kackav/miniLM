import os
import argparse
import functools

import torch
import tqdm
import yaml

import data_asr
import models_asr
from safe_gpu import safe_gpu

import jiwer
from torch.utils.tensorboard import SummaryWriter
import transformers
from whisper_normalizer.english import EnglishSpellingNormalizer

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
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='hidden size of transformer model')
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
    parser.add_argument('--per_device_train_batch_size', type=int, default=8,
                        help='batch size per device for training')
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
    parser.add_argument('--generate', action='store_true',
                        help='generate from latest')
    
    parser.add_argument('--train_dir', type=str, default=None,
                        help='train dataset dir')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='val dataset dir')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='log runs dir')
    parser.add_argument('--audio_dir', type=str, default=None,
                        help='audio dataset dir')
    parser.add_argument('--tokenizer_dir', type=str, default=None,
                        help='tokenizer dir')
    parser.add_argument('--lm_dir', type=str, default=None,
                        help='lm dir')
    parser.add_argument('--encoder_model_name', type=str, default='microsoft/wavlm-large',
                        help='encoder model name')
    parser.add_argument('--lm_model_name', type=str, default="allenai/OLMo-1B-hf",
                        help='lm name as string')

    parser.add_argument('--train_encoder', action='store_true',
                        help='train encoder')
    parser.add_argument('--train_lm', action='store_true',
                        help='train lm')
    
    parser.add_argument('--encoder_eval', action='store_true',
                        help='eval mode for encoder during training, if False train mode')
    parser.add_argument('--connector_eval', action='store_true',
                        help='eval mode for connector during training, if False train mode')
    parser.add_argument('--lm_eval', action='store_true',
                        help='eval mode for lm during training, if False train mode')

    args = parser.parse_args()
    if args.tokenizer_dir is None:
        args.tokenizer_dir = args.output_dir
    
    #torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision('high')

    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    safe_gpu.claim_gpus()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    writer = SummaryWriter(log_dir=args.log_dir)

    val_file = args.val_dir
    train_file = args.train_dir
    dset_train = data_asr.load_from_transcript(train_file, args.audio_dir)
    dset_valid = data_asr.load_from_transcript(val_file, args.audio_dir)
    trans_train = [{"text": _["text"][0]} for _ in dset_train]
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.lm_model_name, trust_remote_code=True)
    train_data = data_asr.AsrDataset(dset_train, tokenizer, args.audio_dir, args.bos_token)

    encoder = models_asr.WavLMWrapper(args.encoder_model_name)
    connector = models_asr.Connector(encoder.encoder.config.hidden_size, args.hidden_size, num_heads = 4, ff_size = 4*encoder.encoder.config.hidden_size)
    connector = connector.to(torch.bfloat16)
    lm = transformers.AutoModelForCausalLM.from_pretrained(args.lm_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map='auto')
   #lm = models_asr.Transformer.load_from_dir(args.lm_dir, device)
    
    model = models_asr.EncoderConnectorLmWithPretrainedLm(encoder, connector, lm, tokenizer)
    model = model.to(device)
    
    if not args.train_encoder:
        model.freeze_encoder()
    if not args.train_lm:
        model.freeze_lm()

    parameters_to_train = list(model.connector.parameters())
    if args.train_encoder:
        parameters_to_train += list(model.encoder.parameters())
    if args.train_lm:
        parameters_to_train += list(model.lm.parameters())
    optimizer = torch.optim.AdamW(parameters_to_train, lr=args.peak_lr, weight_decay=args.weight_decay)
    
    print(model.config)
    
    print(f'Number of total parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    lr_lambda_partial = functools.partial(lr_lambda, args=args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_partial)

    # Running variables for keeping track of training metrics
    train_loss = 0
    training_acc = 0
    training_count = 0
    best_val_loss = float('inf')
    all_metrics = []

    bos_token = args.bos_token
    lm_model_name = args.lm_model_name
    start_step = 0
    if args.resume:
        print('Resuming from checkpoint')
        model.connector = model.connector.load_from_dir(os.path.join(args.output_dir, 'latest'), device)
        model.connector = model.connector.to(torch.bfloat16)
        model.connector = model.connector.to(device)
        if args.train_encoder:
            model.encoder = model.encoder.load_from_dir(os.path.join(args.output_dir, 'latest'), device)
            model.encoder = model.encoder.to(torch.bfloat16)
            model.encoder = model.encoder.to(device)
        
        parameters_to_train = list(model.connector.parameters())
        if args.train_encoder:
            parameters_to_train += list(model.encoder.parameters())
        if args.train_lm:
            parameters_to_train += list(model.lm.parameters())
        optimizer = torch.optim.AdamW(parameters_to_train, lr=args.peak_lr, weight_decay=args.weight_decay)
        lr_lambda_partial = functools.partial(lr_lambda, args=args)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_partial)
        optimizer.load_state_dict(torch.load(os.path.join(args.output_dir, 'latest', 'optimizer.pt')))
        scheduler.load_state_dict(torch.load(os.path.join(args.output_dir, 'latest', 'scheduler.pt')))
        if os.path.isfile(os.path.join(args.output_dir, 'lm_config.yaml')):
            with open(os.path.join(args.output_dir, 'lm_config.yaml'), 'r') as f:
                lm_config = yaml.load(f, Loader=yaml.FullLoader)
            lm_model_name = lm_config['model_name']
            bos_token = lm_config['bos_token']
        else:
            lm_model_name = args.lm_model_name
            bos_token = args.bos_token
        
        with open(os.path.join(args.output_dir, 'latest', 'metrics.yaml'), 'r') as f:
            all_metrics = yaml.safe_load(f)
        best_val_loss = min([m['val_loss'] for m in all_metrics])
        start_step = all_metrics[-1]['step']

    model.lm = transformers.AutoModelForCausalLM.from_pretrained(lm_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map='auto')
    tokenizer = transformers.AutoTokenizer.from_pretrained(lm_model_name, trust_remote_code=True)
    train_data = data_asr.AsrDataset(dset_train, tokenizer, args.audio_dir, bos_token)
    validation_data = data_asr.AsrDataset(dset_valid, tokenizer, args.audio_dir, bos_token)

    collate_fn = functools.partial(data_asr.collate_fn_asr, tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.per_device_train_batch_size,
                                               collate_fn=collate_fn,
                                               shuffle=True,
                                               num_workers=args.dataloader_num_workers)
                                          #     drop_last=True)
                                        
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.per_device_eval_batch_size,
                                                    collate_fn=collate_fn,
                                                    num_workers=args.dataloader_num_workers)
    train_loader_iter = iter(train_loader)  
    normalizer = EnglishSpellingNormalizer()
    #with torch.autocast(enabled=True, device_type='cuda', dtype=torch.bfloat16):
    for j in tqdm.tqdm(range(start_step+1, args.max_steps + 1)):
        optimizer.zero_grad()
        if args.connector_eval:
            model.connector.eval()
        else:
            model.connector.train()
        if args.encoder_eval:
            model.encoder.eval()
        else:
            model.encoder.train()
        if args.lm_eval:
            model.lm.eval()
        else:
            model.lm.train()
        for k in range(args.accumulation_steps):
            if args.generate:
                break

            try:
                batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch = next(train_loader_iter)
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = batch[key].to(device)
            x = batch
            y = batch['labels'].to(device)

            with torch.autocast(enabled = True, device_type = "cuda", dtype= torch.bfloat16): 
                z, loss, acc = model(x)

            (loss/args.accumulation_steps).backward()

            # optimizer.param_groups[0]['lr'] = scheduler.get_last_lr()[0]
            train_loss += loss.mean().item()
            # acc = ((z.argmax(dim=-1) == y) * (y >= 0) ).sum() / (y >= 0).sum()
            training_acc += acc.mean().item()
            training_count += 1

        if not(args.generate):
            writer.add_scalar("Loss/train", loss, j)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], j)
            writer.add_scalar("Accuracy/train", acc, j)
            writer.add_scalar("Data/Audio Length", batch['audio_len'].float().mean().item(), j)
            writer.add_scalar("Data/Text Length", batch['input_len'].float().mean().item(), j)
            writer.add_scalar("Data/Batch Size", batch['audio'].shape[0], j)

            # clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_value)
            # torch.nn.utils.clip_grad_value_(model.parameters(), args.max_grad_value)
            writer.add_scalar("Loss/Gradient Norm", grad_norm.item(), j)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if j > 0 and j % args.validation_steps == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_acc = 0
                val_count = 0
                all_transcriptions = []
                all_references = []
                for val_batch in tqdm.tqdm(validation_loader,):
                    val_x = val_batch
                    for key, value in val_x.items():
                        if isinstance(value, torch.Tensor):
                            val_x[key] = val_x[key].to(device)
                    
                    val_y = val_batch['labels'].to(device)
                    #val_z = model(val_x)
                    with torch.autocast(enabled = True, device_type = "cuda", dtype= torch.bfloat16): 
                        z, loss, acc = model(val_x)
                    #loss = criterion(val_z.permute(0, 2, 1), val_y)
                    val_loss += (loss).mean().item()
                    #acc = ((val_z.argmax(dim=-1) == val_y) * (val_y >= 0)).sum() / (val_y >= 0).sum()
                    val_acc += acc.mean().item()
                    val_count += 1

                    with torch.autocast(enabled = True, device_type = "cuda", dtype= torch.bfloat16):
                        text_x = [normalizer(text_str) for text_str in model.generate(val_x, 100, bos_token = bos_token)]
                    text_y = [normalizer(text_str) for text_str in val_batch['text_trans']]
                    all_transcriptions += text_x
                    all_references += text_y

                output_wer = jiwer.process_words(all_references, all_transcriptions)
                insertions = output_wer.insertions
                deletions = output_wer.deletions
                substitutions = output_wer.substitutions
                wer = output_wer.wer
                cer = jiwer.cer(all_references, all_transcriptions)
                print(f'WER: {wer}, Insertions: {insertions}, Deletions: {deletions}, Substitutions: {substitutions}')
            
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
            if args.generate:

                val_loss = val_loss / val_count
                writer.add_scalar("Loss/validation", val_loss, j)
                val_acc = val_acc / val_count
                writer.add_scalar("Accuracy/validation", val_acc, j)
                writer.add_scalar("WER/wer", wer, j)
                writer.add_scalar("WER/insertions", insertions, j)
                writer.add_scalar("WER/deletions", deletions, j)
                writer.add_scalar("WER/substitutions", substitutions, j)
                writer.add_scalar("WER/cer", cer, j)
                logging_dict = {
                'step': j,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'wer' : wer,
                }
                print(logging_dict)
                yaml.dump(logging_dict, open(os.path.join(args.output_dir, 'latest', 'val_metrics.yaml'), 'w'))
                writer.flush()
                os._exit(0)

            train_loss = train_loss / training_count
            training_acc = training_acc / training_count
            val_loss = val_loss / val_count
            writer.add_scalar("Loss/validation", val_loss, j)
            val_acc = val_acc / val_count
            writer.add_scalar("Accuracy/validation", val_acc, j)
            writer.add_scalar("WER/wer", wer, j)
            writer.add_scalar("WER/insertions", insertions, j)
            writer.add_scalar("WER/deletions", deletions, j)
            writer.add_scalar("WER/substitutions", substitutions, j)
            writer.add_scalar("WER/cer", cer, j)
            logging_dict = {
                'step': j,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': training_acc,
                'val_acc': val_acc,
                'wer' : wer,
                'learning_rate': optimizer.param_groups[0]['lr'],
                }
            all_metrics.append(logging_dict)
            print(logging_dict)
            train_loss = 0
            training_acc = 0
            training_count = 0

            model.connector.save_to_directory(os.path.join(args.output_dir, 'latest'))
            if args.train_encoder:
                model.encoder.save_to_directory(os.path.join(args.output_dir, 'latest'))
            lm_config = {'model_name' : lm_model_name,
                 'bos_token' : bos_token,
            }
            with open(os.path.join(args.output_dir, 'lm_config.yaml'), 'w') as f:
                yaml.dump(lm_config, f)
    
            yaml.dump(all_metrics, open(os.path.join(args.output_dir, 'latest', 'metrics.yaml'), 'w'))


            # save otpimizer state
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'latest', 'optimizer.pt'))
            # save scheduler state
            torch.save(scheduler.state_dict(), os.path.join(args.output_dir, 'latest', 'scheduler.pt'))

           # yaml.dump(model.config, open(os.path.join(args.output_dir, 'latest', 'config.yaml'), 'w'))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.connector.save_to_directory(os.path.join(args.output_dir, 'best'))
                if args.train_encoder:
                    model.encoder.save_to_directory(os.path.join(args.output_dir, 'best'))
                yaml.dump(all_metrics, open(os.path.join(args.output_dir, 'best', 'metrics.yaml'), 'w'))
            recent_val_loss = [m['val_loss'] for m in all_metrics[-args.early_stopping_patience:]]
            all_val_losses = [m['val_loss'] for m in all_metrics]
            if len(all_val_losses) > args.early_stopping_patience and all(v > min(all_val_losses) for v in recent_val_loss):
                print('Early stopping')
                break

        if j % args.save_steps == 0:
            model.connector.save_to_directory(os.path.join(args.output_dir, f'checkpoint_{j}'))
            if args.train_encoder:
              model.encoder.save_to_directory(os.path.join(args.output_dir, f'checkpoint_{j}'))
            yaml.dump(all_metrics, open(os.path.join(args.output_dir, f'checkpoint_{j}', 'metrics.yaml'), 'w'))
    
        writer.flush()
            
    print('Training complete!')


if __name__ == '__main__':
    main()
