import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embed, head_size, bias=False)
        self.query = nn.Linear(config.n_embed, head_size, bias=False)
        self.value = nn.Linear(config.n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #(B, T, head_size)
        q = self.query(x) #(B, T, head_size)
        wei = q @ k.transpose(-2, -1) * C**-0.5 #
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embed // config.n_head
        self.sa = MultiHeadAttention(config, head_size)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

@dataclass
class ModelConfig:
    block_size: int = 256
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384
    dropout: float = 0.2

class BigramLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # nn.Sequential(
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     nn.LayerNorm(n_embed),
        # )
        self.ln_f = nn.LayerNorm(config.n_embed) # final layer norm
        # self.sa_heads = MultiHeadAttention(4, n_embed//4) # 4 of 8 dimensional self attention
        # self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

    def forward(self, idx, targets= None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) #(B, T, C = channels)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T, C)
        x = tok_emb + pos_emb #(B, T, C)
        # x = self.sa_heads(x) #apply one self attention head
        # x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #(B, T, Cw)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to last block_size token
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] / temperature # becomes (B, C)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
