

import random
import torch
import os

from classes.transformer import Transformer
from classes.vocabulary import PAD_token, SOS_token
from util.transformer_util import get_data_batches, indexFromSentence, normalizeString


def itertrain_transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, 
                      max_seq_length, dropout, n_iteration, pairs, voc, device, batch_size, 
                      num_ecoder_layers, num_decoder_layers,print_every=100):
    
    print("Preparing training batches...")
    train_bacthes = [get_data_batches(random.sample(pairs, batch_size), max_seq_length, voc) 
                     for _ in range(n_iteration)]
    print("Training batches prepared.")

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff,
                              num_ecoder_layers, num_decoder_layers, dropout, max_seq_length, device)
    transformer.to(device)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_token)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for iteration in range(1, n_iteration + 1):
        
        training_batch = train_bacthes[iteration - 1]
        input_variable, target_variable = training_batch

        input_variable = input_variable.to(device)
        target_variable = target_variable.to(device)
    
        optimizer.zero_grad()
        output = transformer(input_variable, target_variable[:, :-1])
        output[:, :, SOS_token] = -1e9
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), target_variable[:, 1:].contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        optimizer.step()

        if iteration % print_every == 0:
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, loss.item()))

        
        if iteration == n_iteration:
            directory = os.path.join('datasets', 'save', 'transformer_model2')
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'voc_dict': voc.__dict__,
                'max_seq_length': max_seq_length,
                'src_vocab_size': src_vocab_size,
                'tgt_vocab_size': tgt_vocab_size,
                'd_model': d_model,
                'num_heads': num_heads,
                'd_ff': d_ff,
                'num_encoder_layers': num_ecoder_layers,
                'num_decoder_layers': num_decoder_layers,
                'dropout': dropout
            }, os.path.join(directory, '{}_checkpoint.tar'.format(iteration)))
            print("Saved model at iteration {} to {}".format(iteration, directory))
    
    return transformer


def evaluate_tr(transformer, max_seq_length, device, voc, sentence, greedy_decoder):

    index_batch = [indexFromSentence(voc, sentence)]
    index_batch = torch.tensor(index_batch, dtype=torch.long, device=device)

    output = greedy_decoder(index_batch)

    return output

def evaluate_tr_input(transformer, max_seq_length, device, voc, greedy_decoder):
    input_sentence = ''
    while(True):
        try:
            input_sentence = input("> ")
            if input_sentence.lower() in ['exit', 'quit', 'q']:
                print("Exiting...")
                break
            
            input_sentence = normalizeString(input_sentence)
            output = evaluate_tr(transformer, max_seq_length, device, voc, input_sentence, greedy_decoder)
            output = output.squeeze().tolist()
            decoded_words = [voc.index2word[token] for token in output]
            print('Bot:', ' '.join(decoded_words[1:-1]))
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print("Boot: I didn't understand that. Please try again.")
            continue