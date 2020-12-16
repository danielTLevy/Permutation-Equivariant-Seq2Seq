# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
from perm_equivariant_seq2seq.utils import tensor_from_sentence
from nltk.translate.bleu_score import sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1

def test_accuracy(model_to_test, pairs, use_bleu=False):
    """Test a model (metric: accuracy) on all pairs in _pairs_

    Args:
        model_to_test: (seq2seq) Model object to be tested
        pairs: (list::pairs) List of list of input/output language pairs
    Returns:
        (float) accuracy on test pairs
    """
    def get_model_sentence(model_output):
        # First, extract sentence up to EOS
        _, sentence_ints = model_output.data.topk(1)
        # If there is no EOS token, take the complete list
        try:
            eos_location = torch.nonzero(sentence_ints == EOS_token)[0][0]
        except:
            eos_location = len(sentence_ints) - 2
        return sentence_ints[:eos_location+1]    
        
    def bleu_score(target, model_sentence):
        reference = [[i.item() for i in target]]
        candidate =  [i.item() for i in model_sentence]
        return sentence_bleu(reference, candidate)

    
    def sentence_correct(target, model_sentence):
        if len(model_sentence) != len(target):
            print(len(model_sentence) - len(target))
            return torch.tensor(0, device=device)
        else:
            correct = model_sentence == target
            return torch.prod(correct).to(device)

    accuracies = []
    bleu_scores = []
    model_to_test.eval()
    with torch.no_grad():
        progress = tqdm(pairs,  desc="Loss: ", position=0, leave=True)
        for pair in progress:
            input_tensor, output_tensor = pair
            model_output = model_to_test(input_tensor=input_tensor)
            model_sentence = get_model_sentence(model_output)
            accuracies.append(sentence_correct(output_tensor, model_sentence))
            if use_bleu:
                bleu_scores.append(bleu_score(output_tensor, model_sentence))
    mean_accuracy = torch.stack(accuracies).type(torch.float).mean()
    if use_bleu:
        mean_bleu = torch.stack(bleu_scores).type(torch.float).mean()
        return mean_accuracy, mean_bleu
    else:
        return mean_accuracy

def evaluate(model_to_eval,
             inp_lang,
             out_lang,
             sentence):
    """Decode one sentence from input -> output language with the model

    Args:
        model_to_eval: (nn.Module: Seq2SeqModel) seq2seq model being evaluated
        inp_lang: (Lang) Language object for input language
        out_lang: (Lang) Language object for output language
        sentence: (torch.tensor) Tensor representation (1-hot) of sentence in 
        input language
    Returns:
        (list) Words in output language as decoded by model
    """
    model_to_eval.eval()
    with torch.no_grad():
        input_sentence = tensor_from_sentence(inp_lang, sentence)
        model_output = model_to_eval(input_tensor=input_sentence)

        decoded_words = []
        for di in range(model_to_eval.max_length):
            topv, topi = model_output[di].data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(out_lang.index2word[topi.item()])
        return decoded_words

