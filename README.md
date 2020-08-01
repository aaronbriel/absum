# absum - Abstractive Summarization for Data Augmentation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/absum.svg)](https://badge.fury.io/py/absum)
![Python 3.6, 3.7](https://img.shields.io/badge/python-3.6%20%7C%203.7-green.svg)

## Introduction
Imbalanced class distribution is a common problem in ML. Undersampling combined with oversampling are two methods of addressing this issue. 
A technique such as SMOTE can be effective for oversampling, although the problem becomes a bit more difficult with multilabel datasets. 
[MLSMOTE](https://www.sciencedirect.com/science/article/abs/pii/S0950705115002737) has been proposed, but the high dimensional nature of numerical vectors created from text can sometimes make other forms of data augmentation more appealing.

absum is an NLP library that uses abstractive summarization to perform data augmentation in order to oversample under-represented classes in datasets. Recent developments in abstractive summarization make this approach optimal in achieving realistic data for the augmentation process.

It uses the latest [Huggingface T5](https://huggingface.co/transformers/model_doc/t5.html) model by default but is designed in a modular way to allow you to use any pre-trained or out-of-the-box Transformers models capable of abstractive summarization. 
absum is format agnostic, expecting only a dataframe containing text and all features. It also uses multiprocessing to achieve optimal performance.

## Algorithm
1. Append counts or the number of rows to add for each feature are first calculated with a ceiling threshold. Namely, if a given feature has 1000 rows and the ceiling is 100, its append count will be 0.

2. For each feature it then completes a loop from an append index range to the append count specified for that given feature. The append index is stored
to allow for multi processing.

3. An abstractive summarization is calculated for a specified size subset of all rows that uniquely have the given feature. 
If multiprocessing is set, the call to abstractive summarization is stored in a task array later passed to a sub-routine that runs the calls in parallel using the [multiprocessing](https://docs.python.org/2/library/multiprocessing.html) library, vastly reducing runtime.

4. Each summarization is appended to a new dataframe with the respective features one-hot encoded. 

## Installation
### Via pip

```bash
pip install absum
```

### From source

```bash
git clone https://github.com/aaronbriel/absum.git
pip install [--editable] .
```

or

```bash
pip install git+https://github.com/aaronbriel/absum.git
```

## Usage

absum expects a DataFrame containing a text column which defaults to 'text', and the remaining columns representing one-hot encoded features.
If additional columns are present that you do not wish to be considered, you have the option to pass in specific one-hot encoded features as a comma-separated string to the 'features' parameter. All available parameters are detailed in the Parameters section below.

```bash
import pandas as pd
from absum import Augmentor

csv = 'path_to_csv'
df = pd.read_csv(csv)
augmentor = Augmentor(df, text_column='review_text')
df_augmented = augmentor.abs_sum_augment()
# Store resulting dataframe as a csv
df_augmented.to_csv(csv.replace('.csv', '-augmented.csv'), encoding='utf-8', index=False)
```

When running you may see the following warning message which can be ignored: 
"Token indices sequence length is longer than the specified maximum sequence length for this model (2987 > 512). 
Running this sequence through the model will result in indexing errors". For more information refer to [this issue](https://github.com/huggingface/transformers/issues/1791).

## Parameters

| Name | Type | Description |
| ---- | ---- | ----------- |
| df | (:class:`pandas.Dataframe`) | Dataframe containing text and one-hot encoded features.
| text_column | (:obj:`string`, `optional`, defaults to "text") | Column in df containing text.
| features | (:obj:`string`, `optional`, defaults to None) | Comma-separated string of features to possibly augment data for.
| device | (:class:`torch.device`, `optional`, 'cuda' or 'cpu') | Torch device to run on cuda if available otherwise cpu.
| model | (:class:`~transformers.T5ForConditionalGeneration`, `optional`, defaults to T5ForConditionalGeneration.from_pretrained('t5-small')) | Model used for abstractive summarization.
| tokenizer | (:class:`~transformers.T5Tokenizer`, `optional`, defaults to T5Tokenizer.from_pretrained('t5-small')) | Tokenizer used for abstractive summarization.
| return_tensors | (:obj:str, `optional`, defaults to "pt") | Can be set to ‘tf’, ‘pt’ or ‘np’ to return respectively TensorFlow tf.constant, PyTorch torch.Tensor or Numpy :oj: np.ndarray instead of a list of python integers.
| num_beams | (:obj:`int`, `optional`, defaults to 4) | Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.
| no_repeat_ngram_size | (:obj:`int`, `optional`, defaults to 4 | If set to int > 0, all ngrams of size no_repeat_ngram_size can only occur once.
| min_length | (:obj:`int`, `optional`, defaults to 10) | The min length of the sequence to be generated. Between 0 and infinity. Default to 10.
| max_length | (:obj:`int`, `optional`, defaults to 50) | The max length of the sequence to be generated. Between min_length and infinity. Default to 50.
| early_stopping | (:obj:`bool`, `optional`, defaults to True) | bool if set to True beam search is stopped when at least num_beams sentences finished per batch. Defaults to False as defined in configuration_utils.PretrainedConfig.
| skip_special_tokens | (:obj:`bool`, `optional`, defaults to True) | Don't decode special tokens (self.all_special_tokens). Default: False.
| num_samples | (:obj:`int`, `optional`, defaults to 100) | Number of samples to pull from dataframe with specific feature to use in generating new sample with Abstractive Summarization.
| threshold | (:obj:`int`, `optional`, defaults to 3500) | Maximum ceiling for each feature, normally the under-sample max.
| multiproc | (:obj:`bool`, `optional`, defaults to True) | If set, stores calls to abstractive summarization in array which is then passed to run_cpu_tasks_in_parallel to allow for increasing performance through multiprocessing.
| debug | (:obj:`bool`, `optional`, defaults to True) | If set, prints generated summarizations.

## Citation

Please reference [this library](https://github.com/aaronbriel/absum) and the HuggingFace [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) library if you use this work in a published or open-source project.
