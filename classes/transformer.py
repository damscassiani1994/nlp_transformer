import torch
import torch.nn as nn

from classes.decoder import DecoderLayer
from classes.encoder import EncoderLayer



class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_encoder_layers, 
                 num_decoder_layers, dropout, max_seq_len, device):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, max_seq_len) 
                                             for _ in range(num_encoder_layers)])
        
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout, max_seq_len) 
                                             for _ in range(num_decoder_layers)])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        src_mask = src_mask.to(self.device)
        B, tgt_len = tgt.shape
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask.to(self.device)

        nopeak_mask = torch.triu(torch.ones(1, tgt_len, tgt_len), diagonal=1).bool()
        nopeak_mask = nopeak_mask.to(self.device)

        nopeak_mask = nopeak_mask.unsqueeze(0).unsqueeze(1)
        tgt_mask = tgt_mask & ~nopeak_mask
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.encoder_embedding(src)
        tgt_embedded = self.decoder_embedding(tgt)

        src_embedded = self.dropout(src_embedded)
        tgt_embedded = self.dropout(tgt_embedded)

        enc_output = src_embedded
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, tgt_mask, src_mask)

        ouput = self.fc_out(dec_output)
        return ouput
    

       