import csv

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import os
import codecs
from io import open
import math
from classes.vocabulary import Voc
from train.transformer_train import itertrain_transformer
from util.transformer_util import loadPrepareData

corpus_data_path= "./datasets/movie-corpus"
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available else "cpu"

print("Divece", device)

datafile = os.path.join(corpus_data_path, "formatted_dailydialog.txt")
delimiter = '%'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

save_dir = os.path.join('datasets', 'save')
voc: Voc = None
voc, pairs = loadPrepareData(corpus_data_path, datafile)
print("\npairs:")
for pair in pairs[:10]:
    print(pair)

src_vocab_size = voc.num_words
tgt_vocab_size = voc.num_words
d_model = 512
num_heads = 8
num_encoder_layers = 4
num_decoder_layers = 4
d_ff = 4 * d_model
max_seq_length = 60
dropout = 0.1
n_iteration = 40000
batch_size = 8
print_every = 1

print("\nMax sequence length:", max_seq_length)

print("\nTraining Transformer...")
itertrain_transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, max_seq_length,
                      dropout, n_iteration, pairs, voc, device, batch_size, 
                      num_encoder_layers, num_decoder_layers, print_every)
print("\nTraining Transformer finished!")