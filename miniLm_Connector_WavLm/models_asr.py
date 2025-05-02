import os
import torch
import torch.nn as nn
import yaml

from typing import Dict
import transformers
import data_asr

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


class Transformer(nn.Module):
    def __init__(self,
                 num_embeddings,
                 hidden_size,
                 num_heads,
                 ff_size,
                 num_layers,
                 max_len,
                 dropout,
                 tokenizer=None,
                 causal=True
                 ):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings, hidden_size, padding_idx=None if tokenizer is None else tokenizer.token_to_id('[PAD]')
        )
        self.positional_embedding = PositionalEmbedding(hidden_size, max_len)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(hidden_size, num_heads, ff_size, dropout, causal=causal)
                                     for _ in range(num_layers)])
        self.output = nn.Linear(hidden_size, num_embeddings)

        self.config = {'num_embeddings': num_embeddings,
                       'hidden_size': hidden_size,
                       'num_heads': num_heads,
                       'ff_size': ff_size,
                       'num_layers': num_layers,
                       'max_len': max_len,
                       'dropout': dropout,
                       'causal': causal,
                       }

    def forward(self, x, input_is_embeddings=False):
        if not input_is_embeddings:
            x = self.embedding(x)
            x = self.positional_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.output(x)
        return x

    def save_to_directory(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'lm_config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        torch.save(self.state_dict(), os.path.join(output_dir, 'lm_model.pt'))

    @classmethod
    def load_from_dir(cls, output_dir, device=None):
        with open(os.path.join(output_dir, 'config.yaml'), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'model.pt'), weights_only=True, map_location=device))
        return model


def lengths_to_mask(lengths: torch.Tensor):
    if lengths is None:
        return None
    max_length = lengths.max()
    mask = torch.arange(max_length)[None, :].to(lengths.device) < lengths[:, None]
    return mask


class WavLMWrapper(nn.Module): 
    def __init__(self, model_name="microsoft/wavlm-base-plus"):
        super().__init__()
        self.encoder = transformers.WavLMModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.encoder.config.mask_feature_prob = 0.0
        self.strides = [_.conv.stride[0] for _ in self.encoder.feature_extractor.conv_layers]
        self.kernel_sizes = [_.conv.kernel_size[0] for _ in self.encoder.feature_extractor.conv_layers]
        self.config = {'strides': self.strides,
                'kernel_sizes': self.kernel_sizes
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


class Connector(nn.Module):
    def __init__(self, dim_input, dim_output, num_heads, ff_size, num_connector_layers = 2, kernel_sizes = (7,7), strides = (3,2)):
        super(Connector, self).__init__()
        assert len(kernel_sizes) == len(strides), "kernel_sizes and strides must have the same length"
        self.cnn = nn.ModuleList([nn.Conv1d(dim_input, dim_input, kernel_size=kernel, stride=stride)
                                  for kernel, stride in zip(kernel_sizes, strides)])
        self.strides = strides
        self.kernel_sizes = kernel_sizes
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
   
    def forward(self, x):
        features = x['encoder_output']
        speech_length = x['encoder_output_length']
        for i, (kernel_size, stride) in enumerate(zip(self.kernel_sizes, self.strides)):
            features = features.transpose(1, 2)
            features = self.cnn[i](features)
            features = features.transpose(1, 2)
            speech_length = (1 + (speech_length - kernel_size) / stride).int()
    
        features = features[:, :speech_length.max().item(), :]
        for block in self.blocks:
            features = block(features)

        x['connector_output'] = self.linear(features)
        x['connector_output_length'] = speech_length
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
        if 'num_encoder_layers' in config:
            config['num_connector_layers'] = config.pop('num_encoder_layers')
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'connector_model.pt'), weights_only=True, map_location=device))
        return model


class EncoderConnectorLm(nn.Module):
    def __init__(self, encoder, connector, lm, tokenizer):
        super(EncoderConnectorLm, self).__init__()
        self.encoder = encoder
        self.connector = connector
        self.lm = lm
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction = "sum")
        self.tokenizer = tokenizer
        self.printing = True
        self.config = {'encoder': encoder.config,
                       'connector' : connector.config,
                       'lm' : lm.config}
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def freeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = False

    def forward(self, x):
        text, text_lengths = x['input_ids'], x['input_len']
        labels, labels_lengths = x['labels'], x['labels_len']
        if self.printing:
            print(x['audio'].shape)
            print(x['audio_len'])
            print(x['input_ids'].shape)
            print(x['input_len'])

        x = self.encoder(x)

        if self.printing:
            print(x['encoder_output'].shape)
            print(x['encoder_output_length'])

        x = self.connector(x)
        if self.printing:
            print(x['connector_output'].shape)
            print(x['connector_output_length'])

        connector_output, connector_lengths = x['connector_output'], x['connector_output_length']
        text_embedding = self.lm.embedding(text)
        text_embedding = self.lm.positional_embedding(text_embedding)
        full_embedding, full_lengths = concat_two_sequences(connector_output, text_embedding, connector_lengths, text_lengths)
        if self.printing:
            print(full_embedding.shape)
            print(full_lengths)
        logits = self.lm(
            full_embedding,
            input_is_embeddings=True,
            )
        x_logits, y_logits, x_ln, y_ln = separate_two_sequences(logits, full_lengths, connector_lengths)
        loss = self.criterion(y_logits.transpose(1, -1), labels)
        accuracy = ((y_logits.argmax(dim=-1) == labels).float()*lengths_to_mask(text_lengths)).sum() / text_lengths.sum()
        self.printing = False
        return y_logits, loss, accuracy

    def generate(self, x, max_len): #100 maxlen
        batch_size = x['audio'].shape[0]
        x = self.encoder(x)
        x = self.connector(x)
        connector_output, connector_lengths = x['connector_output'], x['connector_output_length']
        text = torch.ones(batch_size, 1, device = connector_output.device, dtype=torch.long) * self.tokenizer.token_to_id("[BOS]")
        eos_token = self.tokenizer.token_to_id("[EOS]")
        x_still_active = torch.ones(batch_size, device=x['audio'].device, dtype=torch.bool)
        for i in range(max_len):
            text_lengths = text.shape[1] * torch.ones(batch_size, device = connector_output.device, dtype=torch.long)
            text_embedding = self.lm.embedding(text)
            text_embedding = self.lm.positional_embedding(text_embedding)
            full_embedding, full_lengths = concat_two_sequences(connector_output, text_embedding, connector_lengths, text_lengths)
            logits = self.lm(
                full_embedding,
                input_is_embeddings=True,
                )
            x_logits, y_logits, x_ln, y_ln = separate_two_sequences(logits, full_lengths, connector_lengths)
            z = y_logits[:, -1].argmax(dim=-1)
            z_to_concat = torch.where(x_still_active, z, torch.full_like(z, eos_token))
            text = torch.cat([text, z_to_concat.unsqueeze(1)], dim=1)
            x_still_active = x_still_active & (z != eos_token)
            if not x_still_active.any():
                break
        text_str = self.tokenizer.decode_batch(text.tolist())
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

