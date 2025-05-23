import os
import torch
import torch.nn as nn
import yaml

from typing import Dict
import transformers

try:
    import flash_attn
    print(f'Using flash_attn version {flash_attn.__version__}')
except ImportError:
    flash_attn = None


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, causal = True
                 ):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.torch_flashable = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.dropout = nn.Dropout(dropout) if (flash_attn is None and not self.torch_flashable) else dropout
        self.causal = causal

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x)
        # The first two (if, elif) are just efficient implementations of what is in the else block
        if flash_attn is not None:
            dtype = x.dtype
            qkv = qkv.to(torch.bfloat16)
            qkv = qkv.view(b, t, 3, self.num_heads, self.head_dim)
            x = flash_attn.flash_attn_qkvpacked_func(
                qkv, softmax_scale=self.scale, dropout_p=self.dropout, causal=self.causal
            )
            x = x.view(b, t, c)
            x = x.to(dtype)
        elif self.torch_flashable:
            qkv = qkv.chunk(3, dim=-1)
            q, k, v = (qkv[i].view(*x.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
                       for i in range(3))
            if self.torch_flashable:
                x = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, dropout_p=self.dropout, is_causal=self.causal,
                )
                x = x.transpose(1, 2).reshape(b, t, c)
        else:
            qkv = qkv.chunk(3, dim=-1)
            q, k, v = (qkv[i].view(*x.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
                       for i in range(3))
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if self.causal:
                mask = torch.triu(torch.ones(*attn.shape[-2:], device=attn.device), diagonal=1)
                attn.masked_fill_(mask == 1, float('-inf'))
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            x = (attn @ v).transpose(1, 2).reshape(*x.shape)
        x = self.fc(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size, dropout, causal = True):
        super(TransformerEncoderBlock, self).__init__()
        self.attn = SelfAttention(hidden_size, num_heads, dropout=dropout, causal = causal)
        self.linear1 = nn.Linear(hidden_size, ff_size)
        self.linear2 = nn.Linear(ff_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def postnorm_forward(self, x):
        attn_output = self.attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_output))
        return x

    def forward(self, x):
        norm = self.norm1(x)
        attn_output = self.attn(norm)
        x = x + self.dropout(attn_output)
        norm = self.norm2(x)
        ff_output = self.linear2(self.activation(self.linear1(norm)))
        x = x + self.dropout(ff_output)
        return x


class PositionalEmbedding(nn.Module):
    # Simplified positional embedding with single embedding per position
    def __init__(self, hidden_size, max_len):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Parameter(torch.randn(max_len, hidden_size))

    def forward(self, x):
        return x + self.embedding[:x.shape[1]].unsqueeze(0)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, hidden_size, max_len):
        super(SinusoidalPositionalEmbedding, self).__init__()
        embedding = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(torch.log(torch.tensor(10000.0)) / hidden_size))
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('embedding', embedding)

    def forward(self, x):
        return x + self.embedding[:x.shape[1]].unsqueeze(0)
    
class TextEncoder(nn.Module):
    def __init__(self,
                 mask_rate,
                 vocab_size,
                 dim_input,
                 dim_output,
                 num_heads,
                 ff_size,                 
                 tokenizer,
                 num_layers = 2,
                 dropout = 0,
                 causal=False
                 ):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, dim_input, padding_idx=tokenizer.pad_token_id)
        self.positional_embedding = SinusoidalPositionalEmbedding(dim_input, max_len = 512)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(dim_input, num_heads, ff_size, dropout, causal=causal)
                                     for _ in range(num_layers)])
        self.mask_rate = mask_rate
        self.linear = nn.Linear(dim_input, dim_output)

        self.config = {
            'mask_rate': mask_rate,
            'vocab_size': vocab_size,
            'dim_input': dim_input,
            'dim_output': dim_output,
            'num_heads': num_heads,
            'ff_size': ff_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'tokenizer': tokenizer.__dict__,
            'causal': causal,
        }

    def forward(self, x):
        # x looks like this: 
            # "text_trans" : text_trans,
            # "input_ids": input_ids,
            # "input_len": input_len,
            # "labels": labels,
            # "labels_len": labels_len
            # }      
        features = self.embedding(x['input_ids'])
        text_length = x['input_len']
        mask = torch.rand((features.shape[0], features.shape[1], 1), device=features.device)
        features = torch.where(mask>self.mask_rate, features, torch.zeros_like(features))
        features = self.positional_embedding(features)
        for block in self.blocks:
            features = block(features)

        x['TextEncoder_output'] = self.linear(features)
        x['TextEncoder_output_length'] = text_length
        # x should look like this: 
            # "text_trans" : text_trans,
            # "input_ids": input_ids,
            # "input_len": input_len,
            # "labels": labels,
            # "labels_len": labels_len,
            # "TextEncoder_output" : TextEncoder_output,
            # "TextEncoder_output_length" : TextEncoder_output_length
            # }

        return x

    def save_to_directory(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'TextEncoder_config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        torch.save(self.state_dict(), os.path.join(output_dir, 'TextEncoder_model.pt'))

    @classmethod
    def load_from_dir(cls, output_dir, device=None):
        with open(os.path.join(output_dir, 'TextEncoder_config.yaml'), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'TextEncoder_model.pt'), weights_only=True, map_location=device))
        return model


def lengths_to_mask(lengths: torch.Tensor):
        if lengths is None:
            return None
        max_length = lengths.max()
        mask = torch.arange(max_length)[None, :].to(lengths.device) < lengths[:, None]
        return mask


def lengths_to_left_padded_mask(lengths):
    if lengths is None:
        return None
    max_length = lengths.max()
    mask = torch.arange(max_length - 1, -1, -1)[None, :].to(lengths.device) < lengths[:, None]
    return mask


def shift_batch_right(input_tensor, lengths):
    z = torch.zeros_like(input_tensor)
    for i, (au, le) in enumerate(zip(input_tensor, lengths)):
        z[i, -le:] = au[:le]
    return z


class WavLMWrapper(nn.Module): 
    def __init__(self, model_name="microsoft/wavlm-large"):
        super().__init__()
        self.encoder = transformers.WavLMModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.encoder.config.mask_feature_prob = 0.0
        self.strides = [_.conv.stride[0] for _ in self.encoder.feature_extractor.conv_layers]
        self.kernel_sizes = [_.conv.kernel_size[0] for _ in self.encoder.feature_extractor.conv_layers]
        self.config = {
            'model_name': model_name,
            # 'strides': self.strides,
            # 'kernel_sizes': self.kernel_sizes,
            }
        
    def forward(self,  x: Dict[str, torch.Tensor]):
        features = x['audio']
        speech_length = x['audio_len']
        attention_mask = lengths_to_mask(speech_length).long()
        
        for kernel_size, stride in zip(self.kernel_sizes, self.strides):
            speech_length = (1 + (speech_length - kernel_size) / stride).int()

        features = self.encoder(features, attention_mask=attention_mask)
        features = features.last_hidden_state

        x['encoder_output'] = features
        x['encoder_output_length'] = speech_length
        return x

    def save_to_directory(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'encoder_config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        torch.save(self.state_dict(), os.path.join(output_dir, 'encoder_model.pt'))
    
    @classmethod
    def load_from_dir(cls, output_dir, device=None):
        with open(os.path.join(output_dir, 'encoder_config.yaml'), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'encoder_model.pt'), weights_only=True, map_location=device))
        return model


class Connector(nn.Module):
    def __init__(self, dim_input, dim_output, num_heads, ff_size, num_connector_layers = 2, kernel_sizes = (7,7), strides = (3,2)):
        super(Connector, self).__init__()
        assert len(kernel_sizes) == len(strides), "kernel_sizes and strides must have the same length"
        self.cnn = nn.ModuleList([nn.Conv1d(dim_input, dim_input, kernel_size=kernel, stride=stride)
                                  for kernel, stride in zip(kernel_sizes, strides)])
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.positional_embedding = SinusoidalPositionalEmbedding(dim_input, max_len = 512)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(dim_input, num_heads, ff_size, dropout = 0, causal=False)
                                     for _ in range(num_connector_layers)])
        
        self.linear = nn.Linear(dim_input, dim_output)
        self.config = {'dim_input': dim_input,
                       'dim_output': dim_output,
                       'num_heads': num_heads,
                       'ff_size': ff_size,
                       'num_connector_layers': num_connector_layers,
                       'kernel_sizes': list(kernel_sizes),
                       'strides': list(strides)
                       }
   
    def forward(self, x, is_text = False):
        if not(is_text): #encoder output
            features = x['encoder_output']
            speech_length = x['encoder_output_length']
            features = nn.functional.gelu(features)
            for i, (kernel_size, stride) in enumerate(zip(self.kernel_sizes, self.strides)):
                features = features.transpose(1, 2)
                features = self.cnn[i](features)
                features = features.transpose(1, 2)
                features = nn.functional.gelu(features)
                speech_length = (1 + (speech_length - kernel_size) / stride).int()
            features = self.positional_embedding(features)
            input_length = speech_length
            features = features[:, :input_length.max().item(), :]
        else: #TextEncoder output
            #masked text embeddings with positional embeddings.
            features = x['TextEncoder_output']
            input_length = x['TextEncoder_output_length']
        for block in self.blocks:
            features = block(features)

        x['connector_output'] = self.linear(features)
        x['connector_output_length'] = input_length
        return x
    
    def save_to_directory(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'connector_config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        torch.save(self.state_dict(), os.path.join(output_dir, 'connector_model.pt'))
    
    @classmethod
    def load_from_dir(cls, output_dir, device=None):
        with open(os.path.join(output_dir, 'connector_config.yaml'), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # if 'num_encoder_layers' in config:
        #     config['num_connector_layers'] = config.pop('num_encoder_layers')
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'connector_model.pt'), weights_only=True, map_location=device))
        return model

class EncoderConnectorLmWithPretrainedLm(nn.Module):
    def __init__(self, encoder, text_encoder, connector, lm, tokenizer):
        super(EncoderConnectorLmWithPretrainedLm, self).__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.connector = connector
        self.lm = lm
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = tokenizer
        self.printing = False
        self.config = {'encoder': [encoder.config],
                'connector' : [connector.config],
                'lm' : [lm.config.__dict__],}

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def freeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = False

    def forward(self, x, is_text = False):
        text, text_lengths = x['input_ids'], x['input_len']
        labels, labels_lengths = x['labels'], x['labels_len']
        if self.printing:
            print(x['audio'].shape)
            print(x['audio_len'])
            print(x['input_ids'].shape)
            print(x['input_len'])
        if not is_text:
            x = self.encoder(x)
            if self.printing:
                print('encoder')
                print(x['encoder_output'].shape)
                print(x['encoder_output_length'])

        else:
            x = self.text_encoder(x)
            if self.printing:
                print('TextEncoder_output')
                print(x['TextEncoder_output'].shape)
                print(x['TextEncoder_output_length'])

        x = self.connector(x, is_text = is_text)
        if self.printing:
            print('connector')
            print(x['connector_output'].shape)
            print(x['connector_output_length'])

        connector_output, connector_lengths = x['connector_output'], x['connector_output_length']

        text_embedding = self.lm.get_input_embeddings()(text)
        if self.printing:
            print(f'text embedding shape:{text_embedding.shape}')

        connector_output_r = shift_batch_right(connector_output, connector_lengths)
        connector_mask = lengths_to_left_padded_mask(connector_lengths)
        full_embedding = torch.cat((connector_output_r, text_embedding), dim=1)
        text_mask = torch.ones(text_embedding.shape[0], text_embedding.shape[1], device=connector_output.device, dtype=torch.bool)
        full_mask = torch.cat((connector_mask, text_mask), dim=1)

        lm_outputs = self.lm(inputs_embeds = full_embedding, attention_mask=full_mask)
        logits = lm_outputs['logits']
        if self.printing:
            print(f'logits shape:{logits.shape}')

        y_logits = logits[:, -labels.shape[1]:]
        loss = self.criterion(y_logits.transpose(1, -1), labels)
        accuracy = ((y_logits.argmax(dim=-1) == labels).float()*lengths_to_mask(text_lengths)).sum() / text_lengths.sum()
        self.printing = False
        return y_logits, loss, accuracy

    def generate(self, x, max_len, bos_token):  # 100 maxlen
        batch_size = x['audio'].shape[0]
        if bos_token is None:
            bos_token = self.tokenizer.eos_token_id
        else:
            bos_token = self.tokenizer.encode(bos_token)[0]
        x = self.encoder(x)
        x = self.connector(x)
        connector_output, connector_lengths = x['connector_output'], x['connector_output_length']
        text = torch.ones(batch_size, 1, device=connector_output.device, dtype=torch.long) * bos_token

        gen_config = transformers.GenerationConfig(
            bos_token_id=bos_token, #self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=bos_token, #self.tokenizer.eos_token_id,
            decoder_end_token_id=self.tokenizer.eos_token_id,
            length_penalty=1,  # gen_args.length_penalty,
            early_stopping=False,  # gen_args.early_stopping,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=2,  # gen_args.num_beams,
            max_new_tokens=max_len,  # gen_args.max_new_tokens,
        )

        self.lm.generation_config = gen_config
        text_embedding = self.lm.get_input_embeddings()(text)
        connector_output_r = shift_batch_right(connector_output, connector_lengths)
        full_embedding = torch.cat((connector_output_r, text_embedding), dim=1)
        text = self.lm.generate(inputs_embeds=full_embedding, attention_mask=lengths_to_left_padded_mask(connector_lengths + 1),
                                 do_sample=False)

        text_str = self.tokenizer.batch_decode(text.tolist(), skip_special_tokens=True)
        return text_str

def concat_two_sequences(x, y, x_lengths, y_lengths):
    lengths = x_lengths + y_lengths
    z_shape = list(x.shape)
    z_shape[1] = lengths.max().item()
    z = torch.zeros(*z_shape, device=x.device)
    for i in range(x.shape[0]):
        z[i, :x_lengths[i]] = x[i, :x_lengths[i]]
        z[i, x_lengths[i]:lengths[i]] = y[i, :y_lengths[i]]
    return z, lengths

def separate_two_sequences(x, x_lengths, z_lengths):
    y_lengths = x_lengths - z_lengths
    z_shape = list(x.shape)
    z_shape[1] = z_lengths.max().item()
    z = torch.zeros(*z_shape, device=x.device)

    y_shape = list(x.shape)
    y_shape[1] = y_lengths.max().item()
    y = torch.zeros(*y_shape, device=x.device)

    for i in range(x.shape[0]):
        z[i, :z_lengths[i]] = x[i, :z_lengths[i]]
        y[i, :y_lengths[i]] = x[i, z_lengths[i]:z_lengths[i] + y_lengths[i]]
    return z, y, z_lengths, x_lengths

