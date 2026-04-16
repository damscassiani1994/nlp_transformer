import torch
import torch.nn as nn


class GreedySearchTransformerDecoder(nn.Module):
    def __init__(self, model, sos_token, eos_token, max_length, device):
        super(GreedySearchTransformerDecoder, self).__init__()
        self.model = model
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_length = max_length
        self.device = device

    def generate_square_subsequent_mask(self, mask_size):
        mask = torch.triu(torch.ones(mask_size, mask_size), diagonal=1).bool()
        return mask.to(self.device)
    
    def forward(self, src):
        batch_size = src.size(0)

        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(self.device)

        enc_output = self.model.encoder_embedding(src)
        #enc_output = self.model.positional_encoding(enc_output)
        enc_output = self.model.dropout(enc_output)

        for layer in self.model.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        decoder_ouput = torch.ones(batch_size, 1, dtype=torch.long, device=self.device) * self.sos_token

        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(self.max_length):

            tgt_mask = (decoder_ouput != 0).unsqueeze(1).unsqueeze(3).to(self.device)

            seq_len = decoder_ouput.size(1)
            nopeak_mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).bool().to(self.device)
            nopeak_mask = nopeak_mask.unsqueeze(1)
            tgt_mask = tgt_mask & nopeak_mask

            dec_output = self.model.decoder_embedding(decoder_ouput)
            #dec_output = self.model.positional_encoding(dec_output)
            dec_output = self.model.dropout(dec_output)

            for layer in self.model.decoder_layers:
                dec_output = layer(dec_output, enc_output, tgt_mask, src_mask)

            output = self.model.fc_out(dec_output)

            logits = output[:, -1, :]

            #Penalizando repeticiones
            for i in range(batch_size):
                for token in decoder_ouput[i]:
                    logits[i, token] /= 1.2

            temperature = 1.0
            logits = logits / temperature

            top_k = 5
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

            probs = torch.softmax(top_k_logits, dim=-1)

            next_word = top_k_indices.gather(-1, torch.multinomial(probs, num_samples=1))
            ##next_word = torch.argmax(logits, dim=-1, keepdim=True)

            decoder_ouput = torch.cat((decoder_ouput, next_word), dim=1)

            finished |= (next_word.squeeze(1) == self.eos_token)
            if finished.all():
                break
        return decoder_ouput


