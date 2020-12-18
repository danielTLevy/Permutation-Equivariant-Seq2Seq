# Permutation-Equivariant-Seq2Seq
Reproducing the experiments in "Permutation Equivariant Models for Compositional Generalization in Language"

## Licence
Permutation-Equivariant-Seq2Seq is licensed under the MIT license. The text of the license can be found [here](https://github.com/facebookresearch/Permutation-Equivariant-Seq2Seq/blob/master/LICENSE).


## Dependencies
This code requires the following:
* Python 3.6 or greater
* PyTorch 1.0 or greater 

## Installation
1. Clone or download this repository.
2. Enter top level directory ```cd Permutation-Equivariant-Seq2Seq```
3. Install SCAN dataset: clone the [SCAN](https://github.com/brendenlake/SCAN) repository into top level directory.

## Usage (Levy, Humane, Belkebir)
To run the code for the COMP 550 Final Project:

Run ```python train_equivariant_engfra.py --equivariance bookcarhouse``` for an equivariant seq2seq network, ```python train_seq2seq_engfra.py``` for a standard seq2seq.

If using the "add house" task, add the argument ```--split add_house```. Otherwise, for a normal translation task, add the flag ```--split bookcarhouse```

To test the code,run:
```python test_equivariant_engfra.py`` for an equivariant seq2seq network,
```python test_seq2seq_engfra.py``` for a standard seq2seq.
It will list instructions on how to retrieve the model you wish to test, whether to print training accuracy, test accuracy, or example sentences, and more

## Usage (Gordon et al.)
To reproduce the paper experiments with a permutation equivariant sequence-2-sequence model on SCAN, run the following commands:

1. "Simple Split" with verb equivariance:

```python train_equivariant_scan.py --split simple --equivariance verb --use_attention --bidirectional```

2. "Add jump" split with verb equivariance:

```python train_equivariant_scan.py --split add_jump --equivariance verb --use_attention --bidirectional```

3. "Around right" split with direction equivariance:
  
```python train_equivariant_scan.py --split around_right --equivariance direction --use_attention --bidirectional```

4. "Length generalization" split with direction and verb equivariance:
  
```python train_equivariant_scan.py --split length_generalization --equivariance direction+verb --use_attention --bidirectional```

Models will automatically be saved in the `models` directory. After training, models can be tested using the following command:

```python test_equivariant_scan.py --experiment_dir MODEL_DIRECTORY --best_validation```

Use the `--compute_train_accuracy` and `--compute_test_accuracy` flags to compute the desired quantities.

Note: the repository also supports training non-equivariant models on the SCAN tasks. This can be achieved with the following commands:

1. "Simple Split":

```python train_seq2seq_scan.py --split simple --use_attention --bidirectional```

2. "Add jump":

```python train_seq2seq_scan.py --split add_jump --use_attention --bidirectional```

3. "Around right":
  
```python train_seq2seq_scan.py --split around_right --use_attention --bidirectional```

4. "Length generalization":
  
```python train_seq2seq_scan.py --split length_generalization --use_attention --bidirectional```

## Contact
To ask questions or report issues, please open an issue on the issues tracker.

## Citation
If you use this code, please cite our [paper](https://openreview.net/forum?id=SylVNerFvr):
```
@inproceedings{
Gordon2020Permutation,
title={Permutation Equivariant Models for Compositional Generalization in Language},
author={Jonathan Gordon and David Lopez-Paz and Marco Baroni and Diane Bouchacourt},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=SylVNerFvr}
}
```
