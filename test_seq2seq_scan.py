# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
import argparse
import os

import torch
from test_utils import test_accuracy, evaluate
import perm_equivariant_seq2seq.utils as utils
from perm_equivariant_seq2seq.models import BasicSeq2Seq
from perm_equivariant_seq2seq.scan_data_utils import get_scan_split, get_invariant_scan_languages
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
                    choices=[None, 'simple', 'add_jump', 'length_generalization'],
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





if __name__ == '__main__':
    # Make sure all data is contained in the directory and load arguments
    args_path = os.path.join(args.experiment_dir, "commandline_args.txt")
    if args.fully_trained:
        model_path = os.path.join(args.experiment_dir, "model_fully_trained.pt")
    else:
        model_path = os.path.join(args.experiment_dir, "model_iteration%s.pt" % args.iterations)
    assert os.path.exists(args.experiment_dir), "Experiment directory not found"
    assert os.path.exists(model_path), "Model number not found in directory"
    assert os.path.exists(args_path), "Argparser details directory not found in directory"
    experiment_arguments = utils.load_args_from_txt(parser, args_path)

    # Load data
    train_pairs, test_pairs = get_scan_split(split=experiment_arguments.split)
    commands, actions = get_invariant_scan_languages(train_pairs)

    # Initialize model
    model = BasicSeq2Seq(input_language=commands,
                         encoder_hidden_size=experiment_arguments.hidden_size,
                         decoder_hidden_size=experiment_arguments.semantic_size,
                         output_language=actions,
                         layer_type=experiment_arguments.layer_type,
                         use_attention=experiment_arguments.use_attention,
                         drop_rate=experiment_arguments.drop_rate,
                         bidirectional=experiment_arguments.bidirectional,
                         num_layers=experiment_arguments.num_layers)

    # Move model to device and load weights
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    # Convert data to torch tensors
    training_eval = [tensors_from_pair(pair, commands, actions) 
                     for pair in train_pairs]
    testing_pairs = [tensors_from_pair(pair, commands, actions) 
                     for pair in test_pairs]

    # Compute accuracy and print some translations
    if args.compute_train_accuracy:
        train_acc = test_accuracy(model, training_eval)
        print("Model train accuracy: %s" % train_acc.item())
    if args.compute_test_accuracy:
        test_acc = test_accuracy(model, testing_pairs)
        print("Model test accuracy: %s" % test_acc.item())
    if args.print_param_nums:
        print("Model contains %s params" % model.num_params)
    for i in range(args.print_translations):
        pair = random.choice(test_pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(model, commands, actions, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

