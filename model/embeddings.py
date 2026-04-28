import torch
import torch.nn as nn


class GPTEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, input_ids):
        B, T = input_ids.shape

        tok_emb = self.token_embed(input_ids)

        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos_ids = pos_ids.expand(B, T)

        pos_emb = self.pos_embed(pos_ids)

        return tok_emb + pos_emb