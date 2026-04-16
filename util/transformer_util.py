
import re
import unicodedata

from classes.vocabulary import Voc, SOS_token, EOS_token
import torch

from classes.vocabulary import PAD_token

MAX_LENGTH = 50

def transfor_max_length(pairs):
    return max(max(len(pair[0].split()), len(pair[1].split())) for pair in pairs)

def indexFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def padding_sequences(sequence, max_length, pad_value=PAD_token):
    num_pad = max_length - len(sequence)
    return sequence + ([pad_value] * num_pad)


def get_data_batches(pairs, max_length, vocab: Voc):
    input_bacthes = []
    target_batches = []

    for pair in pairs:
        input_seq = indexFromSentence(vocab, pair[0])
        target_seq = [SOS_token] + indexFromSentence(vocab, pair[1])

        input_bacthes.append(input_seq)
        target_batches.append(target_seq)

    inputs = [padding_sequences(seq, max_length) for seq in input_bacthes]
    targets = [padding_sequences(seq, max_length) for seq in target_batches]

    return torch.tensor(inputs), torch.tensor(targets)

def unicode_to_accii(text: str):
    return ''.join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )

def normalizeString(text: str):
    text = unicode_to_accii(text.lower().strip())
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    text = re.sub(r"\s+", r" ", text).strip()
    return text

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name: str):
    print("Reading lines...")
    lines = open(datafile, encoding="utf-8").read().strip().split('\n')
    pairs = [[normalizeString(text) for text in l.split('%')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# Returns True if both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH

# Filter pairs using the ``filterPair`` condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def loadPrepareData(corpus_name, datafile):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs