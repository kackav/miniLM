import json
import torch
import tqdm
import accelerate
import compute_metrics
import yaml
import os
import data_asr
import transformers
import argparse
import models_asr
from peft import get_peft_model, LoraConfig, PeftModel

hf_dataset_path = os.environ.get('HF_HOME', '')+ '/modules/datasets_modules/datasets'

print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', ''))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                        help='number of workers for dataloader')

    parser.add_argument('--per_device_eval_batch_size', type=int, default=8,
                        help='batch size per device for evaluation')
   
    parser.add_argument('--use_llm_emb', action='store_true',
                        help = "set to True if you don't want to initialize new embeddings in the Text Encoder")
    parser.add_argument('--use_bos_from_lm', action='store_true',
                        help = "set to True if you want to use lm.config.bos_token. If Lm doesn't have it, specify bos_token argument (default is eos token)")
    parser.add_argument('--bos_token', type=str, default=None, 
                        help="bos_token for lm, if None takes eos token from tokenizer")
   
    parser.add_argument('--output_dir', type=str, default='models',
                        help='output directory to save model checkpoints and generated outputs')
    
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
    
    parser.add_argument('--encoder_dir', type=str, default=None,
                        help='encoder dir, if you want to load encoder from different dir than connector')

    accelerator = accelerate.Accelerator()
    print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    _x = torch.tensor([1.0], device=accelerator.device, dtype=torch.bfloat16)

    # device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    device = accelerator.device
    print(f'Using device: {device} - index: {accelerator.process_index}')
    # n_gpus = int(len(os.environ.get('CUDA_VISIBLE_DEVICES'))/2)+1
    n_gpus = accelerator.num_processes
    args = parser.parse_args()
    input_args = {}
    for arg in vars(args):
        accelerator.print(f'{arg}: {getattr(args, arg)}')
        input_args[f'{arg}'] = f'{getattr(args, arg)}'
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.lm_model_name, trust_remote_code=True)

    datasets_config = yaml.load(open(args.datasets_config), Loader=yaml.FullLoader)
    dset_valid = data_asr.load_dial_from_config('validation', datasets_config, hf_dataset_path, tokenizer,
                                                accelerator=accelerator) # dictonary ["commonvoice": ..., "librispeech": ... etc]
   
    lm_model_name = args.lm_model_name
    lm = transformers.AutoModelForCausalLM.from_pretrained(args.lm_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map=accelerator.device, )
    # if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
    #     bos_token = tokenizer.bos_token_id
    # elif args.bos_token is not None:
    #     bos_token = tokenizer.encode(args.bos_token)[0]
    # else:
    #     bos_token = tokenizer.eos_token_id
    if args.use_bos_from_lm or (hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None):
        bos_token = lm.config.bos_token_id
    else:
        if args.bos_token is None:
            bos_token = lm.config.eos_token_id
        else:
            bos_token = tokenizer.encode(args.bos_token)[0]
    val_subsets = {}
    for k, v in dset_valid.items():
        inner_dataset = data_asr.DialogDatasetHF({k:v}, tokenizer, bos_token)
        val_subsets[k] = data_asr.InferenceDialogDatasetHF(inner_dataset)

    collate_fn_dialog_inference = data_asr.CollateFnDialogInference(tokenizer=tokenizer, non_inference_collate_fn=data_asr.CollateFnDialog(tokenizer))
    
    validation_loaders = {}
    for name, ds in val_subsets.items():
        validation_loader = torch.utils.data.DataLoader(ds, batch_size=args.per_device_eval_batch_size,
                                                    collate_fn=collate_fn_dialog_inference,
                                                    num_workers=args.dataloader_num_workers_val, prefetch_factor=4, pin_memory= False)
        validation_loaders[name] = validation_loader
    connector = models_asr.Connector.load_from_dir(args.pretrained_dir, device)
    connector = connector.to(device)
    if os.path.isfile(os.path.join(args.pretrained_dir,'encoder_config.yaml')):
        accelerator.print(f'Loading Encoder config from {args.pretrained_dir}')
        encoder = models_asr.WavLMWrapper.load_from_dir(args.pretrained_dir, device, deactivate_masked_spec_embed = True)
    else:
        encoder = models_asr.WavLMWrapper.load_from_dir(args.encoder_dir, device, deactivate_masked_spec_embed = True)
        
    if os.path.isfile(os.path.join(args.pretrained_dir, 'lm', 'adapter_config.json')):
        accelerator.print(f'Loading LoRA config from {args.pretrained_dir}')
        lm = PeftModel.from_pretrained(lm, os.path.join(args.pretrained_dir,'lm'),
                                       torch_device="cpu") 
    model = models_asr.EncoderConnectorLmWithPretrainedLm(encoder, connector, lm, tokenizer)
    # if os.path.isfile(os.path.join(args.pretrained_dir, 'TextEncoder_model.pt')):
    #         accelerator.print(f'Loading pretrained text encoder from {args.pretrained_dir}')
    #         text_encoder = models_asr.TextEncoder.load_from_dir(args.pretrained_dir, device)

    # try:
    #     model = models_asr.EncoderConnectorLmWithPretrainedLm(encoder, connector, lm, tokenizer, text_encoder)
    # except:
    #     if args.use_llm_emb:
    #         text_encoder = models_asr.TextEncoder(args.mask_rate, vocab_size=None, dim_input = args.text_encoder_dim, dim_output = encoder.encoder.config.hidden_size, num_heads=4, ff_size = 4*(args.text_encoder_dim), pad_token=tokenizer.pad_token_id, dim_lm_embeddings=lm.config.hidden_size, num_layers = args.text_encoder_layers)
    #     else:
    #         text_encoder = models_asr.TextEncoder(args.mask_rate, tokenizer.vocab_size, args.text_encoder_dim, encoder.encoder.config.hidden_size, num_heads=4, ff_size = 4*(args.text_encoder_dim), pad_token=tokenizer.pad_token_id, num_layers = args.text_encoder_layers)
    #     model = models_asr.EncoderConnectorLmWithPretrainedLm(encoder, connector, lm, tokenizer, text_encoder)
    model = model.to(device)
    for k, v in validation_loaders.items():
        validation_loaders[k] = accelerator.prepare(v)
    model = accelerator.prepare(model)

    os.makedirs(args.output_dir, exist_ok=True)
    for ds_name, val_dataset_loader in validation_loaders.items():
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            val_count = 0
            all_transcriptions = []
            all_references = []
            all_dst_metrics = {
                "domain_tp": 0,
                "domain_fp": 0,
                "domain_fn": 0,
                "slot_k_tp": 0,
                "slot_k_fp": 0,
                "slot_k_fn": 0,
                "slot_v_tp": 0,
                "slot_v_fp": 0,
                "slot_v_fn": 0,
                "num_erroneous_turns": 0,
                "num_turns": 0,
                # "ref_labels": [],
                # "hyp_labels": [],
                }
            jga_full_dict = {}
            for f, (val_batch, val_pointers) in enumerate(tqdm.tqdm(val_dataset_loader, disable=not accelerator.is_main_process)):
                # if f >= 10:
                #      break
                val_x = val_batch
                for vx in val_x:
                    for key, value in vx.items():
                        if isinstance(value, torch.Tensor):
                            vx[key] = vx[key].to(device)

                # for _vx, vx in enumerate(val_x):
                #     with torch.autocast(enabled = True, device_type = "cuda", dtype= torch.bfloat16):
                #         z, loss, acc = model(vx, is_text = False, output_hidden_states=False)
                #     val_loss += accelerator.gather(loss).mean().item()
                #     val_acc += accelerator.gather(acc).mean().item()
                #     val_count += 1
                # ref = val_batch["text_trans"]
                # hyp = z.argmax(dim=-1)
                # hyp = [h[:l] for l, h in zip(val_x["input_len"], hyp)]
                # hyp = [tokenizer.decode(h, skip_special_tokens=True) for h in hyp]
                with torch.autocast(enabled = True, device_type = "cuda", dtype= torch.bfloat16):
                    hyp, jga_dict = accelerator.unwrap_model(model).generate_dialog_multi_turn(val_batch, val_pointers, 500, bos_token)
                jga_full_dict.update(jga_dict)
                ref = [v["text_trans"] for v in val_batch]
                ref = sum(ref, [])
                hyp = sum(hyp, [])
                batch_metrics = compute_metrics.compute_dst_training_metrics(ref, hyp)
                all_transcriptions += batch_metrics.pop("hyp_labels")
                all_references += batch_metrics.pop("ref_labels")
                all_dst_metrics = {k: all_dst_metrics[k] + batch_metrics[k] for k in all_dst_metrics.keys()}
            keys = all_dst_metrics.keys()
            values = torch.tensor(list(all_dst_metrics.values()), device=device)
            values = accelerator.reduce(values).tolist()
            all_dst_metrics = {k: v for k, v in zip(keys, values)}
            dst_summary_metrics = compute_metrics.compute_dst_precision_recall_f1(all_dst_metrics)
            jga = (all_dst_metrics["num_turns"] - all_dst_metrics["num_erroneous_turns"]) / all_dst_metrics["num_turns"] if all_dst_metrics["num_turns"] > 0 else 0.0
            dst_summary_metrics["jga"] = jga
            with open(os.path.join(args.output_dir, f"predictions_{ds_name}_proc_{accelerator.process_index}.json"), "w") as f:
                    json.dump(jga_full_dict, f, indent=4)
            if accelerator.is_main_process:
                with open(os.path.join(args.output_dir, f"predictions_{ds_name}_all.json"), "w") as f:
                    jga_full_dict = {}
                    for proc_id in range(accelerator.num_processes):
                        with open(os.path.join(args.output_dir, f"predictions_{ds_name}_proc_{proc_id}.json"), "r") as f_:
                            jga_full_dict.update(json.load(f_))
                    json.dump(jga_full_dict, f, indent=4)
                print(f'Validation dst metrics for {ds_name}: {dst_summary_metrics}')
                with open(os.path.join(args.output_dir, "metrics.yaml"), "a") as f:
                    yaml.dump({f'val_{ds_name}_dst_summary_metrics': dst_summary_metrics}, f)

if __name__ == '__main__':
    main()