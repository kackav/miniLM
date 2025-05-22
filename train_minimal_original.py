import os
import argparse
import functools

import torch
import tqdm
import yaml

import data_minimal
import models_minimal


def lr_lambda_linear_with_min_lr(step, args):
    peak_lr = args.peak_lr
    min_lr = peak_lr * args.min_lr_ratio
    if step < args.warmup_steps:
        return step / args.warmup_steps
    slope = (peak_lr - min_lr) / (args.max_steps - args.warmup_steps)
    intercept = peak_lr - slope * args.warmup_steps
    current_lr = slope * step + intercept
    return max(current_lr, min_lr) / peak_lr


def lr_lambda_cosine_with_min_lr(step, args):
    peak_lr = args.peak_lr
    min_lr = peak_lr * args.min_lr_ratio
    if step < args.warmup_steps:
        return step / args.warmup_steps
    slope = (peak_lr - min_lr) / 2
    intercept = min_lr
    current_lr = slope * (1 + torch.cos(
        torch.tensor(step - args.warmup_steps) / (
                args.max_steps - args.warmup_steps) * torch.pi)).item() + intercept
    return current_lr / peak_lr


def lr_lambda(step, args):
    if args.scheduler == 'linear':
        return lr_lambda_linear_with_min_lr(step, args)
    return lr_lambda_cosine_with_min_lr(step, args)


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout probability in transformer model')

    parser.add_argument('--bpe_num_sentences', type=int, default=100000,
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
    parser.add_argument('--early_stopping_patience', type=int, default=30000,
                        help='number of validation runs to wait before early stopping')
    parser.add_argument('--max_steps', type=int, default=50000,
                        help='maximum number of steps to train for')
    parser.add_argument('--scheduler', type=str, default='linear', choices=['linear', 'cosine'],
                        help='learning rate scheduler to use')
    parser.add_argument('--per_device_train_batch_size', type=int, default=128,
                        help='batch size per device for training')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=128,
                        help='batch size per device for evaluation')

    parser.add_argument('--peak_lr', type=float, default=1e-4,
                        help='peak learning rate')
    parser.add_argument('--min_lr_ratio', type=float, default=0.4,
                        help='minimum learning rate as a ratio of peak learning rate')

    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay')
    parser.add_argument('--max_grad_value', type=float, default=5.0,
                        help='maximum absolute gradient value for gradient clipping')
    parser.add_argument('--decode_checkpoint', type=str, default='best',
                        help='checkpoint to use for decoding')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='output directory to save model checkpoints and generated outputs')

    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    # dset = datasets.load_dataset(args.dataset, args.dataset_name)
    # dset_train = dset[args.train_split]
    # dset[args.valid_split]
    dset_train, dset_valid = data_minimal.load_shakespeare_data()

    tokenizer = data_minimal.train_bpe(dset_train, vocab_size=args.bpe_vocab_size, output_dir=args.output_dir,
                                       num_sentences=args.bpe_num_sentences)

    train_data = data_minimal.Dataset(dset_train, tokenizer)
    validation_data = data_minimal.Dataset(dset_valid, tokenizer)

    collate_fn = functools.partial(data_minimal.collate_fn, tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.per_device_train_batch_size,
                                               collate_fn=collate_fn,
                                               num_workers=args.dataloader_num_workers)

    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.per_device_eval_batch_size,
                                                    collate_fn=collate_fn,
                                                    num_workers=args.dataloader_num_workers)

    model = models_minimal.Transformer(args.bpe_vocab_size,
                                       args.hidden_size,
                                       args.num_heads,
                                       args.ff_size,
                                       args.num_layers,
                                       args.max_len,
                                       args.dropout)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    print(model.config)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)  # -100 is the label padding token in collate_fn

    lr_lambda_partial = functools.partial(lr_lambda, args=args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_partial)

    train_loader_iter = iter(train_loader)

    # Running variables for keeping track of training metrics
    train_loss = 0
    training_acc = 0
    training_count = 0
    best_val_loss = float('inf')
    all_metrics = []

    for j in tqdm.tqdm(range(1, args.max_steps + 1)):
        try:
            batch = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            batch = next(train_loader_iter)
        x = batch['input_ids'].to(device)
        y = batch['labels'].to(device)
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z.permute(0, 2, 1), y)
        loss.backward()
        # clip gradients
        pytorch_total_params = sum(p.numel() for p in model.parameters())
       
        torch.nn.utils.clip_grad_value_(model.parameters(), args.max_grad_value)
        optimizer.step()
        scheduler.step()
        train_loss += loss.mean().item()
        acc = ((z.argmax(dim=-1) == y) * (y >= 0) ).sum() / (y >= 0).sum()
        training_acc += acc.mean().item()
        training_count += 1
        if j > 0 and j % args.validation_steps == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_acc = 0
                val_count = 0
                for val_batch in tqdm.tqdm(validation_loader,):
                    val_x = val_batch['input_ids'].to(device)
                    val_y = val_batch['labels'].to(device)
                    val_z = model(val_x)
                    loss = criterion(val_z.permute(0, 2, 1), val_y)
                    val_loss += (loss).mean().item()
                    acc = ((val_z.argmax(dim=-1) == val_y) * (val_y >= 0)).sum() / (val_y >= 0).sum()
                    val_acc += acc.mean().item()
                    val_count += 1
            train_loss = train_loss / training_count
            training_acc = training_acc / training_count
            val_loss = val_loss / val_count
            val_acc = val_acc / val_count
            logging_dict = {
                'step': j,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': training_acc,
                'val_acc': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr'],
                }
            all_metrics.append(logging_dict)
            print(logging_dict)
            model.train()
            train_loss = 0
            training_acc = 0
            training_count = 0

            model.save_to_directory(os.path.join(args.output_dir, 'latest'))
            yaml.dump(all_metrics, open(os.path.join(args.output_dir, 'latest', 'metrics.yaml'), 'w'))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_to_directory(os.path.join(args.output_dir, 'best'))
                yaml.dump(all_metrics, open(os.path.join(args.output_dir, 'best', 'metrics.yaml'), 'w'))
            recent_val_loss = [m['val_loss'] for m in all_metrics[-args.early_stopping_patience:]]
            all_val_losses = [m['val_loss'] for m in all_metrics]
            if len(all_val_losses) > args.early_stopping_patience and all(v > min(all_val_losses) for v in recent_val_loss):
                print('Early stopping')
                break

        if j % args.save_steps == 0:
            model.save_to_directory(os.path.join(args.output_dir, f'checkpoint_{j}'))
            yaml.dump(all_metrics, open(os.path.join(args.output_dir, f'checkpoint_{j}', 'metrics.yaml'), 'w'))
    
    # After training is done, the following could be run standalone e.g. in a jupyter notebook to play around with the model
    # Load the best model
    model = models_minimal.Transformer.load_from_dir(os.path.join(args.output_dir, args.decode_checkpoint), device)
    tokenizer = data_minimal.load_tokenizer(os.path.join(args.output_dir, f'bpe_tokenizer_{args.bpe_vocab_size}.json'))

    # Generate 5 paragraphs of up to 100 tokens each from the model and save into a file
    x = tokenizer.token_to_id('[BOS]') * torch.ones(5, 1, dtype=torch.long, )
    z = model.generate(x, 100, temperature=1.0)
    # decode
    sentences = tokenizer.decode_batch(z.tolist())
    with open(os.path.join(args.output_dir, f'generated.txt'), 'w') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
            print(sentence)


if __name__ == '__main__':
    main()
