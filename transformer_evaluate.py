import torch
import os
from classes.greedy_search_transformer_decoder import GreedySearchTransformerDecoder
from classes.vocabulary import Voc, SOS_token, EOS_token
from train.transformer_train import evaluate_tr_input
from util.transformer_util import transfor_max_length, get_data_batches
from classes.transformer import Transformer

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available else "cpu"
print("Device to evaluate on: ", device)

checkpoit = torch.load(os.path.join('datasets', 'save', 'transformer_model2', '50000_checkpoint.tar'), map_location=device)

transformer_sd = checkpoit['model_state_dict']
voc_dic = checkpoit['voc_dict']
max_seq_length = checkpoit['max_seq_length']
src_vocab_size = checkpoit['src_vocab_size']
tgt_vocab_size = checkpoit['tgt_vocab_size']
d_model = checkpoit['d_model']
num_heads = checkpoit['num_heads']
d_ff = checkpoit['d_ff']
num_encoder_layers = checkpoit['num_encoder_layers']
num_decoder_layers = checkpoit['num_decoder_layers']
dropout = checkpoit['dropout']

voc = Voc(voc_dic['name'])
voc.__dict__ = voc_dic

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff,
                          num_encoder_layers, num_decoder_layers, dropout, max_seq_length, device)
transformer.load_state_dict(transformer_sd)
transformer = transformer.to(device)
transformer.eval()

greedy_decoder = GreedySearchTransformerDecoder(transformer, SOS_token, EOS_token, max_seq_length, device)

input_sentence = "How are you doing today?"
evaluate_tr_input(transformer, max_seq_length, device, voc, greedy_decoder)




