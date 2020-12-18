# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
import argparse
import os
from tqdm import tqdm
from test_utils import test_accuracy, evaluate
import torch
import torch.nn as nn

import perm_equivariant_seq2seq.utils as utils
from perm_equivariant_seq2seq.models import BasicSeq2Seq
from perm_equivariant_seq2seq.engfra_data_utils import get_engfra_split, get_invariant_engfra_languages, splits
from perm_equivariant_seq2seq.utils import tensors_from_pair

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1


parser = argparse.ArgumentParser()
# Experiment options
parser.add_argument('--experiment_dir', 
                    type=str, 
                    help='Path to experiment directory, should contain args and model')
parser.add_argument('--fully_trained', 
                    dest='fully_trained', 
                    default=False, 
                    action='store_true',
                    help="Boolean indicating whether to use fully trained model")
parser.add_argument('--iterations', 
                    type=int, 
                    default=0, 
                    help='If not fully trained, how many training iterations.')
parser.add_argument('--compute_train_accuracy', 
                    dest='compute_train_accuracy', 
                    default=False, 
                    action='store_true',
                    help="Boolean to evaluate train accuracy")
parser.add_argument('--compute_test_accuracy', 
                    dest='compute_test_accuracy', 
                    default=False, 
                    action='store_true',
                    help="Boolean to evaluate test accuracy")
parser.add_argument('--print_translations', 
                    type=int, 
                    default=0,
                    help="Print a small number of translations from the test set")
parser.add_argument('--print_param_nums', 
                    dest='print_param_nums', 
                    default=False, 
                    action='store_true',
                    help="Print the number of model parameters")
parser.add_argument('--p_new_prim',
                    dest='p_new_prim',
                    default=0.1,
                    help="Proportion of training samples to make new primitive pair",
                    type=float
                    )
args = parser.parse_args()
# Model options
parser.add_argument('--layer_type',
                    choices=['LSTM', 'GRU', 'RNN'], 
                    default='LSTM',
                    help='Type of rnn layers to be used for recurrent components')
parser.add_argument('--hidden_size', 
                    type=int, 
                    default=64, 
                    help='Number of hidden units in encoder / decoder')
parser.add_argument('--semantic_size', 
                    type=int,
                    default=64, 
                    help='Dimensionality of semantic embedding')
parser.add_argument('--num_layers', 
                    type=int, 
                    default=1, 
                    help='Number of hidden layers in encoder')
parser.add_argument('--use_attention', 
                    dest='use_attention', 
                    default=False, 
                    action='store_true',
                    help="Boolean to use attention in the decoder")
parser.add_argument('--bidirectional', 
                    dest='bidirectional', 
                    default=False, 
                    action='store_true',
                    help="Boolean to use bidirectional encoder")
parser.add_argument('--drop_rate', 
                    type=float, 
                    default=0.1, 
                    help="Dropout drop rate (not keep rate)")
# Optimization and training hyper-parameters
parser.add_argument('--split', 
                    choices=splits,
                    help='Each possible split defines a different experiment as proposed by [1]')
parser.add_argument('--validation_size', 
                    type=float, 
                    default=0.,
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
                    default=1000, 
                    help='Frequency with which to print training loss')
parser.add_argument('--plot_freq', 
                    type=int, 
                    default=20, 
                    help='Frequency with which to plot training loss')
parser.add_argument('--save_freq', 
                    type=int,
                    default=200000, 
                    help='Frequency with which to save models during training')



def train(input_tensor,
          target_tensor,
          model_to_train,
          enc_optimizer,
          dec_optimizer,
          loss_fn,
          teacher_forcing_ratio):
    """Perform one training iteration for the model

    Args:
        input_tensor: (torch.tensor) Tensor representation (1-hot) of sentence 
        in input language
        target_tensor: (torch.tensor) Tensor representation (1-hot) of target 
        sentence in output language
        model_to_train: (nn.Module: Seq2SeqModel) seq2seq model being trained
        enc_optimizer: (torch.optimizer) Optimizer object for model encoder
        dec_optimizer: (torch.optimizer) Optimizer object for model decoder
        loss_fn: (torch.nn.Loss) Loss object used for training
        teacher_forcing_ratio: (float) Ratio with which true word is used as 
        input to decoder
    Returns:
        (torch.scalar) Value of loss achieved by model at current iteration
    """
    model.train()
    # Forget gradients via optimizers
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    model_output = model_to_train(input_tensor=input_tensor,
                                  target_tensor=target_tensor,
                                  use_teacher_forcing=use_teacher_forcing)
    train_loss = 0

    target_length = target_tensor.size(0)
    for di in range(target_length):
        decoder_output = model_output[di]
        train_loss += loss_fn(decoder_output[None, :], target_tensor[di])
        _, decoder_output_symbol = decoder_output.topk(1)
        if decoder_output_symbol.item() == EOS_token:
            break
    train_loss.backward()

    # Clip gradients by norm (5.) and take optimization step
    torch.nn.utils.clip_grad_norm_(model_to_train.encoder.parameters(), 5.)
    torch.nn.utils.clip_grad_norm_(model_to_train.decoder.parameters(), 5.)
    enc_optimizer.step()
    dec_optimizer.step()

    return train_loss.item() / target_length

def train_new_prim(model_to_train, new_pairs, args):
        encoder_optimizer = torch.optim.Adam(model_to_train.encoder.parameters(), 
                                         lr=args.learning_rate)
        decoder_optimizer = torch.optim.Adam(model_to_train.decoder.parameters(), 
                                         lr=args.learning_rate)
        criterion = nn.NLLLoss().to(device)
        progress = tqdm(range(len(new_pairs)), position=0, leave=True)
        for i in progress:
            training_pair = new_pairs[i - 1]
            iteration_input, iteration_output = training_pair
            loss = train(input_tensor=iteration_input,
                         target_tensor=iteration_output,
                         model_to_train=model,
                         enc_optimizer=encoder_optimizer,
                         dec_optimizer=decoder_optimizer,
                         loss_fn=criterion,
                         teacher_forcing_ratio=args.teacher_forcing_ratio)
        return model_to_train

if __name__ == '__main__':
    # Make sure all data is contained in the directory and load arguments
    args_path = os.path.join(args.experiment_dir, "commandline_args.txt")
    if args.fully_trained:
        model_path = os.path.join(args.experiment_dir, "model_fully_trained.pt")
    else:
        model_path = os.path.join(args.experiment_dir, "model_trained_%s.pt" % args.iterations)
    assert os.path.exists(args.experiment_dir), "Experiment directory not found"
    assert os.path.exists(model_path), "Model number not found in directory"
    assert os.path.exists(args_path), "Argparser details directory not found in directory"
    experiment_arguments = utils.load_args_from_txt(parser, args_path)

    # Load data
    train_pairs, test_pairs = get_engfra_split(split=experiment_arguments.split)
    eng_lang, fra_lang = get_invariant_engfra_languages(train_pairs+test_pairs)

    if experiment_arguments.split == "add_book":
        new_prim_pair = ['book', 'livre']
    elif experiment_arguments.split == "add_house":
        new_prim_pair = ['house', 'maison']

    # Initialize model
    model = BasicSeq2Seq(input_language=eng_lang,
                         encoder_hidden_size=experiment_arguments.hidden_size,
                         decoder_hidden_size=experiment_arguments.semantic_size,
                         output_language=fra_lang,
                         layer_type=experiment_arguments.layer_type,
                         use_attention=experiment_arguments.use_attention,
                         drop_rate=experiment_arguments.drop_rate,
                         bidirectional=experiment_arguments.bidirectional,
                         num_layers=experiment_arguments.num_layers)

    # Move model to device and load weights
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    

    
    # Convert data to torch tensors
    training_eval = [tensors_from_pair(pair, eng_lang, fra_lang) 
                     for pair in train_pairs]
    testing_pairs = [tensors_from_pair(pair, eng_lang, fra_lang) 
                     for pair in test_pairs]
    
    if experiment_arguments.split in ["add_book", "add_house"]:
        num_new_prim_pairs = int(args.p_new_prim*len(training_eval))
        new_prim_training_pairs = [tensors_from_pair(new_prim_pair, eng_lang, fra_lang)]*num_new_prim_pairs
        model = train_new_prim(model, new_prim_training_pairs, experiment_arguments)

    # Compute accuracy and print some translations
    if args.compute_train_accuracy:
        train_acc, train_bleu = test_accuracy(model, training_eval, True)
        print("Model train accuracy: %s" % train_acc)
        print("Model train BLEU: %s" % train_bleu)
    if args.compute_test_accuracy:
        test_acc, test_bleu = test_accuracy(model, testing_pairs, True)
        print("Model test accuracy: %s" % test_acc)
        print("Model test BLEU: %s" % test_bleu)
    if args.print_param_nums:
        print("Model contains %s params" % model.num_params)
    for i in range(args.print_translations):
        pair = random.choice(test_pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, bleu_score = evaluate(model, eng_lang, fra_lang, pair[0], True, pair[1])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('BLEU Score: ', bleu_score)
        print('')

