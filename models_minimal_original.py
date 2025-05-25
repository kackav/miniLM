import os
import torch
import torch.nn as nn
import yaml

try:
    import flash_attn
    print(f'Using flash_attn version {flash_attn.__version__}')
except ImportError:
    flash_attn = None


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout,
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

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x)
        # The first two (if, elif) are just efficient implementations of what is in the else block
        if flash_attn is not None:
            qkv = qkv.view(b, t, 3, self.num_heads, self.head_dim)
            x = flash_attn.flash_attn_qkvpacked_func(
                qkv, softmax_scale=self.scale, dropout_p=self.dropout, causal=True
            )
            x = x.view(b, t, c)
        elif self.torch_flashable:
            qkv = qkv.chunk(3, dim=-1)
            q, k, v = (qkv[i].view(*x.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
                       for i in range(3))
            if self.torch_flashable:
                x = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, dropout_p=self.dropout, is_causal=True,
                )
                x = x.transpose(1, 2).reshape(b, t, c)
        else:
            qkv = qkv.chunk(3, dim=-1)
            q, k, v = (qkv[i].view(*x.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
                       for i in range(3))
            attn = (q @ k.transpose(-2, -1)) * self.scale
            mask = torch.triu(torch.ones(*attn.shape[-2:], device=attn.device), diagonal=1)
            attn.masked_fill_(mask == 1, float('-inf'))
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            x = (attn @ v).transpose(1, 2).reshape(*x.shape)
        x = self.fc(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size, dropout,):
        super(TransformerEncoderBlock, self).__init__()
        self.attn = SelfAttention(hidden_size, num_heads, dropout=dropout,)
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
                 ):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings, hidden_size, padding_idx=None if tokenizer is None else tokenizer.token_to_id('[PAD]')
        )
        self.blocks = nn.ModuleList([TransformerEncoderBlock(hidden_size, num_heads, ff_size, dropout,)
                                     for _ in range(num_layers)])
        self.positional_embedding = PositionalEmbedding(hidden_size, max_len)
        self.output = nn.Linear(hidden_size, num_embeddings)

        self.config = {'num_embeddings': num_embeddings,
                       'hidden_size': hidden_size,
                       'num_heads': num_heads,
                       'ff_size': ff_size,
                       'num_layers': num_layers,
                       'max_len': max_len,
                       'dropout': dropout,
                       }

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.output(x)
        return x

    def save_to_directory(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        torch.save(self.state_dict(), os.path.join(output_dir, 'model.pt'))

    @classmethod
    def load_from_dir(cls, output_dir, device=None):
        with open(os.path.join(output_dir, 'config.yaml'), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'model.pt'), weights_only=True, map_location=device))
        return model

    def generate_with_eos(self, x, max_len, eos_token, temperature=1.0):
        self.eval()
        x_still_active = torch.ones(x.shape[0], 1, device=x.device, dtype=torch.bool)
        with torch.no_grad():
            for _ in range(max_len):
                z = self(x)
                z = z[:, -1] / temperature
                z = torch.multinomial(torch.softmax(z, dim=-1), 1)
                z_to_concat = torch.where(x_still_active, z, torch.full_like(z, eos_token))
                x = torch.cat([x, z_to_concat], dim=1)
                x_still_active = x_still_active & (z != eos_token)
                if not x_still_active.any():
                    break
        return x
