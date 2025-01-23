import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import textwrap

assert torch.cuda.is_available()


@dataclass
class Config:
    """A pythonic way to hold data (i.e. as an object)."""
    # temperature: float = 1.0  # temperature of softmax TODO:
    batch_size: int = 32  # number of batches processed per step
    context_size: int = 256  # max context length of vocabs fed to predict the next
    vocab_size: int = 65
    n_layer: int = 6  # number of layers of repeated (ATT-MPL) blocks
    dim_head: int = 128  # dimension of each head
    n_heads = 6  # number of heads
    dim_embd: int = n_heads * dim_head  # dimension of Embedding space

    dropout: float = 0.2
    device: str = 'cuda'
    eval_iters: int = 500  # number of evaluation interations to be averaged per estimation


class SLM(nn.Module):
    """"Re-implementation of Andrej Karpathy's lectures from his YouTube channel and his `nn-zero-to-hero` companion repository.

    I baptised it Small Language Model (SLM) since it is has only the chars of a text as vocabulary."""

    def __init__(self, vocab, name='model'):
        super().__init__()  # since subclass of nn.Module
        # defining configs from vocab
        self.config = Config()
        self.config.vocab_size = len(vocab)
        self.char_to_int = {ch: i for i, ch in enumerate(vocab)}
        self.int_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.encode = lambda s: [self.char_to_int[c] for c in s]
        self.decode = lambda l: ''.join([self.int_to_char[n] for n in l])

        # defining model layers
        self.token_embedding_table = nn.Embedding(
            self.config.vocab_size, self.config.dim_embd)
        self.position_embedding_table = nn.Embedding(
            self.config.context_size, self.config.dim_embd)
        self.blocks = nn.Sequential(*[Block(self.config)],
                                    nn.LayerNorm(self.config.dim_embd)
                                    )
        self.lm_head = nn.Linear(self.config.dim_embd, self.config.vocab_size)
        self.config.name = name
        self.config.MODEL_PATH = f'models/{name}{self.config.context_size}.pt'
        self.to(self.config.device)

    def forward(self, idx, targets=None):  # to be trained as a pytorch nn class
        B, P = idx.shape
        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(
            torch.arange(P).to(self.config.device))
        x = token_embedding + position_embedding
        x = self.blocks(x)
        # dims (B, P, C) where B is "batch", P is "position" and C is "channel" is vocab_size
        # / self.config.temperature # if I choose to introduce temperature
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, P, C = logits.shape
            logits = logits.view(B*P, C)  # reshape
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """to generate a snippet of tokens"""
        for _ in range(max_new_tokens):
            idx_context = idx[:, -self.config.context_size:]
            logits, loss = self(idx_context)  # cropping the context
            logits = logits[:, -1, :]  # get the last element of logits
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def snippet_from_keyboard(self, wrap=False, max_new_tokens=1000):
        """to generate a snippet of text from user keyboard and well formated to print"""
        i = input()
        if not i:
            context = torch.zeros((1, 1), dtype=torch.long).to(
                self.config.device)
        else:
            context = torch.tensor(
                [self.encode(i)], dtype=torch.long).to(self.config.device)
        if wrap:
            output_text = textwrap.fill(self.decode(self.generate(
                context, max_new_tokens=max_new_tokens)[0].tolist()), width=100)
        else:
            output_text = self.decode(self.generate(
                context, max_new_tokens=max_new_tokens)[0].tolist())
        print(output_text)

    def snippet(self, s='', wrap=False, max_new_tokens=1000):
        """to generate a snippet of text well formated to print"""
        if not s:
            context = torch.zeros((1, 1), dtype=torch.long).to(
                self.config.device)
        else:
            context = torch.tensor(
                [self.encode(s)], dtype=torch.long).to(self.config.device)
        if wrap:
            output_text = textwrap.fill(self.decode(self.generate(
                context, max_new_tokens=max_new_tokens)[0].tolist()), width=100)
        else:
            output_text = self.decode(self.generate(
                context, max_new_tokens=max_new_tokens)[0].tolist())
        print(output_text)


class HeadAttention(nn.Module):
    """One head of self-attention."""

    def __init__(self, config):
        super().__init__()  # since subclass of nn.Module
        self.key = nn.Linear(config.dim_embd, config.dim_head, bias=False)
        self.query = nn.Linear(config.dim_embd, config.dim_head, bias=False)
        self.value = nn.Linear(config.dim_embd, config.dim_head, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(
            config.context_size, config.context_size)))  # buffers are not parameters
        # TODO: this dropout below might be a bad idea, check it later
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):  # to be trained as a pytorch nn class
        B, P, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # attention scores queries dot keys
        scores = q @ k.transpose(-2, -1) * C**-0.5
        scores = scores.masked_fill(self.tril[:P, :P] == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)  # attention pattern
        scores = self.dropout(scores)
        v = self.value(x)
        return scores @ v


class MultiHeadAttention(nn.Module):
    """A multiple head self-attention."""

    def __init__(self, config):
        super().__init__()  # since subclass of nn.Module
        self.heads = nn.ModuleList([HeadAttention(config)
                                   for _ in range(config.n_heads)])
        # projection layer "back into the pathway"
        self.proj = nn.Linear(config.dim_embd, config.dim_embd)
        # dropout before back into pathway
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):  # to be trained as a pytorch nn class
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MPL(nn.Module):
    """A multi-perceptron layer."""

    def __init__(self, config):
        super().__init__()  # since subclass of nn.Module
        self.net = nn.Sequential(
            nn.Linear(config.dim_embd, 4*config.dim_embd),
            nn.GELU(),  # or ReLU
            nn.Dropout(config.dropout),
        )
        # projection layer "back into the pathway"
        self.proj = nn.Linear(4*config.dim_embd, config.dim_embd)

    def forward(self, x):  # to be trained as a pytorch nn class
        out = self.net(x)
        out = self.proj(out)
        return out


class Block(nn.Module):
    """A block of multi-headed self-attention followed by a pre- and post-normalized MPL."""  # TODO: learn more about normalization of perceptrons

    def __init__(self, config):
        super().__init__()  # since subclass of nn.Module
        self.att = MultiHeadAttention(config)
        self.mpl = MPL(config)
        self.pre_ln = nn.LayerNorm(config.dim_embd)  # normalized the features
        self.post_ln = nn.LayerNorm(config.dim_embd)

    def forward(self, x):  # to be trained as a pytorch nn class
        x = x+self.att(self.pre_ln(x))
        x = x+self.mpl(self.post_ln(x))
        return x


def get_batch(config, data, mode, split=0.9):
    """get train or val batches"""
    n = int(split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    work_data = train_data if mode == 'train' else val_data
    random_locations = torch.randint(
        len(work_data) - config.context_size, (config.batch_size,))
    x = torch.stack([work_data[i:i+config.context_size]
                    for i in random_locations])
    y = torch.stack([work_data[1+i:1+i+config.context_size]
                    for i in random_locations])
    return x.to(config.device), y.to(config.device)


@torch.no_grad()
def estimate_loss(model, data, split=0.9):
    """get a fast estimate of train and val loss"""
    out = {}
    model.eval()
    for mode in ['train', 'val']:
        losses = torch.zeros(model.config.eval_iters)
        # TODO: make this for-loop parallelizable to profit of cuda speed-up
        for k in range(model.config.eval_iters):
            X, Y = get_batch(model.config, data, mode, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[mode] = losses.mean()
    model.train()
    return out
