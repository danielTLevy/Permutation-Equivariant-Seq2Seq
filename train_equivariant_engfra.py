# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb


import perm_equivariant_seq2seq.utils as utils
from perm_equivariant_seq2seq.equivariant_models import EquiSeq2Seq
from perm_equivariant_seq2seq.engfra_data_utils import get_engfra_split, get_equivariant_engfra_languages, get_equivariances
from perm_equivariant_seq2seq.engfra_data_utils import equivariances, splits
from perm_equivariant_seq2seq.utils import tensors_from_pair
from perm_equivariant_seq2seq.symmetry_groups import get_permutation_equivariance
from test_utils import test_accuracy

"""
[1]: Lake and Baroni 2019: Generalization without systematicity: On the 
compositional skills of seq2seq networks
[2]: Bahdanau et al. 2014: Neural machine translation by jointly learning to 
align and translate
[3]: Russin et ak. 2019: Compositional generalization in a deep seq2seq model 
by saparating syntax and semantics
"""
os.environ["WANDB_MODE"] = "dryrun"
wandb.init(project="equi_seq2seq", entity="teamname")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1

# Parse command-line arguments
parser = argparse.ArgumentParser()
# Model options
parser.add_argument('--hidden_size', 
                    type=int, 
                    default=64, 
                    help='Number of hidden units in encoder / decoder')
parser.add_argument('--layer_type', 
                    choices=['GGRU', 'GRNN', 'GLSTM'], 
                    default='GLSTM',
                    help='Type of rnn layers to be used for recurrent components')
parser.add_argument('--use_attention', 
                    dest='use_attention', 
                    default=True, 
                    action='store_true',
                    help="Boolean to use attention in the decoder")
parser.add_argument('--bidirectional', 
                    dest='bidirectional', 
                    default=True, 
                    action='store_true',
                    help="Boolean to use bidirectional encoder.")
# Equivariance options:
parser.add_argument('--equivariance', 
                    choices=equivariances)
# Optimization and training hyper-parameters
parser.add_argument('--split', 
                    choices=splits,
                    help='Each possible split defines a different experiment as proposed by [1]')
parser.add_argument('--weight_decay', 
                    type=float, 
                    default=0., 
                    help='Weight decay for optimizer')
parser.add_argument('--batch_size', 
                    type=int, 
                    default=1, 
                    help='Number of pairs between each gradient step')
parser.add_argument('--validation_size', 
                    type=float, 
                    default=0.1, 
                    help='Validation proportion to use for early-stopping')
parser.add_argument('--n_iters', 
                    type=int, 
                    default=200000,
                    help='number of training iterations')
parser.add_argument('--learning_rate', 
                    type=float, 
                    default=1e-4, 
                    help='init learning rate')
parser.add_argument('--teacher_forcing_ratio', 
                    type=float, 
                    default=0.5)
parser.add_argument('--save_dir', 
                    type=str, 
                    default='./models/', 
                    help='Top-level directory for saving experiment')
parser.add_argument('--print_freq', 
                    type=int, 
                    default=50, 
                    help='Frequency with which to print training loss')
parser.add_argument('--save_freq', 
                    type=int, 
                    default=2000,
                    help='Frequency with which to save models during training')
args = parser.parse_args()

args.save_path = os.path.join(args.save_dir,
                              '%s' % args.split,
                              'rnn_%s_hidden_%s_directions_%s' % (
                                  args.layer_type, 
                                  args.hidden_size, 
                                  2 if args.bidirectional else 1
                              ))
# Create model directory
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)


def pair_generator(pairs, batch_size):
    """Create a generator for batches of pairs

    Args:
        pairs (list::pairs): List of input / output sentences to generate from
        batch_size (int): Size of each batch (number of translation pairs in a 
        batch
    Returns:
        (generator) Object that yields batches of pairs to translate
    """
    for i in range(0, len(pairs), batch_size):
        yield pairs[i:i + batch_size]


def train(batch,
          model_to_train,
          enc_optimizer,
          dec_optimizer,
          loss_fn,
          teacher_forcing_ratio):
    """Perform one training iteration for the model

    Args:
        batch (torch.tensor): Tensor representation (1-hot) of sentence in 
        input language
        model_to_train (nn.Module: Seq2SeqModel): seq2seq model being trained
        enc_optimizer (torch.optimizer): Optimizer object for model encoder
        dec_optimizer (torch.optimizer): Optimizer object for model decoder
        loss_fn (torch.nn.Loss): Loss object used for training
        teacher_forcing_ratio (float): Ratio with which true word is used as 
        input to decoder
    Returns:
        (torch.scalar) Value of loss achieved by model at current iteration
    """
    model_to_train.train()
    # Forget gradients via optimizers
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()
    # Instantiate some iteration variables
    loss = 0
    batch_size = len(batch)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    for training_triplet in batch:
        sentence_loss = 0
        input_tensor, target_tensor = training_triplet
        model_output = model_to_train(input_tensor=input_tensor,
                                      target_tensor=target_tensor,
                                      use_teacher_forcing=use_teacher_forcing)

        target_length = target_tensor.size(0)
        for di in range(target_length):
            decoder_output = model_output[di]
            sentence_loss += loss_fn(decoder_output[None, :], target_tensor[di])
            _, decoder_output_symbol = decoder_output.topk(1)
            if decoder_output_symbol.item() == EOS_token:
                break
        loss += sentence_loss / (di + 1)

    loss /= batch_size
    if output_symmetry_group.learnable:
        loss += 1e-4 * output_symmetry_group.doubly_stochastic_regularization()

    loss.backward()

    # Clip gradients by norm (5.) and take optimization step
    torch.nn.utils.clip_grad_norm_(model_to_train.encoder.parameters(), 5.)
    torch.nn.utils.clip_grad_norm_(model_to_train.decoder.parameters(), 5.)
    enc_optimizer.step()
    dec_optimizer.step()

    return loss.item()



if __name__ == '__main__':
    # Load data
    train_pairs, test_pairs = get_engfra_split(split=args.split)
    print("Got training and testing pairs")

    in_equivariances, out_equivariances = get_equivariances(args.equivariance)
    eng_lang, fra_lang = \
        get_equivariant_engfra_languages(pairs=train_pairs+test_pairs,
                                       input_equivariances=in_equivariances,
                                       output_equivariances=out_equivariances)
    print("making symmetry group")
    input_symmetry_group = get_permutation_equivariance(eng_lang)
    output_symmetry_group = get_permutation_equivariance(fra_lang)

    # Initialize model
    model = EquiSeq2Seq(input_symmetry_group=input_symmetry_group,
                        output_symmetry_group=output_symmetry_group,
                        input_language=eng_lang,
                        encoder_hidden_size=args.hidden_size,
                        decoder_hidden_size=args.hidden_size,
                        output_language=fra_lang,
                        layer_type=args.layer_type,
                        use_attention=args.use_attention,
                        bidirectional=args.bidirectional)
    model.to(device)
    wandb.watch(model)

    # Initialize optimizers
    encoder_optimizer = torch.optim.Adam(model.encoder.parameters(),
                                         lr=args.learning_rate,
                                         weight_decay=args.weight_decay)
    decoder_optimizer = torch.optim.Adam(model.decoder.parameters(),
                                         lr=args.learning_rate,
                                         weight_decay=args.weight_decay)

    # Split off validation set
    val_size = int(len(train_pairs) * args.validation_size)
    random.shuffle(train_pairs)
    train_pairs, val_pairs = train_pairs[val_size:], train_pairs[:val_size]

    # Convert data to torch tensors
    training_pairs = [tensors_from_pair(random.choice(train_pairs), 
                                        eng_lang,
                                        fra_lang)
                      for i in range(args.n_iters)]
    training_eval = [tensors_from_pair(pair, eng_lang, fra_lang)
                     for pair in train_pairs]
    validation_pairs = [tensors_from_pair(pair, eng_lang, fra_lang)
                        for pair in val_pairs]
    testing_pairs = [tensors_from_pair(pair, eng_lang, fra_lang)
                     for pair in test_pairs]

    # Initialize criterion
    criterion = nn.NLLLoss().to(device)

    # Initialize printing / plotting variables
    print_loss_total = 0
    plot_loss_total = 0

    # Enter training loop
    best_bleu = 0.
    model_path = utils.create_exp_dir(args)
    progress = tqdm(enumerate(pair_generator(training_pairs, args.batch_size)),
                    desc="Loss: ", total=int(len(training_pairs)/args.batch_size), position=0, leave=True)
    for iteration, pair_batch in progress:
        # Compute loss (and take one gradient step)
        loss = train(batch=pair_batch,
                     model_to_train=model,
                     enc_optimizer=encoder_optimizer,
                     dec_optimizer=decoder_optimizer,
                     loss_fn=criterion,
                     teacher_forcing_ratio=args.teacher_forcing_ratio)

        print_loss_total += loss
        plot_loss_total += loss

        # Print, plot, etc'
        if (iteration + 1) % args.print_freq == 0:
            print_loss_avg = print_loss_total / args.print_freq
            print_loss_total = 0
            progress.set_description("Loss: {:.4f}".format(print_loss_avg))
            wandb.log({"Train Loss": print_loss_avg})

        if (iteration + 1) % args.save_freq == 0:
            # save model if is better
            if args.validation_size > 0.:
                val_acc, val_bleu = test_accuracy(model, validation_pairs, True)
                val_acc = val_acc
                wandb.log({"Validation Accuracy": val_acc, "Validation BLEU": val_bleu})
                if val_bleu > best_bleu:
                    best_bleu = val_bleu
                    save_path = os.path.join(model_path, 'best_validation.pt')
                    print('\nBest validation accuracy at iteration %s: %s' % (iteration + 1, val_acc))
                    print('\nBest validation BLEU score at iteration %s: %s' % (iteration + 1, val_bleu))
                    torch.save(model.state_dict(), save_path)
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
                    
                else:
                    print('\nNew validation BLEU %s worse than previous best %s' % (val_bleu, best_bleu))
    # Save fully trained model
    save_path = os.path.join(model_path, 'model_fully_trained.pt')
    torch.save(model.state_dict(), save_path)
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
