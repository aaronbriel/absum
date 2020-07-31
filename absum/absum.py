# coding=utf-8
# Copyright 2020 Aaron Briel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from multiprocessing import Process
import numpy as np
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


class Augmentor(object):
    """
    Uses Abstractive Summarization for Data Augmentation to address multi-label class imbalance.
    Parameters:
        df (:class:`pandas.Dataframe`): Dataframe containing text and one-hot encoded features.
        text_column (:obj:`string`, `optional`, defaults to "text"): Column in df containing text.
        features (:obj:`list`, `optional`, defaults to None): Features to possibly augment data for.
        device (:class:`torch.device`, `optional`, defaults to 'cuda' if available otherwise 'cpu'):
            Torch device to run on cuda if available otherwise cpu.
        model (:class:`~transformers.T5ForConditionalGeneration`, `optional`, defaults to 't5-small'):
            T5ForConditionalGeneration model from pretrained 't5-small'.
        tokenizer (:class:`~transformers.T5Tokenizer`, `optional`, defaults to 't5-small'):
            T5Tokenizer model from pretrained 't5-small'.
        return_tensors (:obj:str, `optional`, defaults to "pt") – Can be set to ‘tf’, ‘pt’ or ‘np’ to return
            respectively TensorFlow tf.constant, PyTorch torch.Tensor or Numpy :oj: np.ndarray instead of a
            list of python integers.
        num_beams (:obj:`int`, `optional`, defaults to 4): Number of beams for beam search. Must be between 1
            and infinity. 1 means no beam search. Default to 1.
        no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 4): If set to int > 0, all ngrams of size
            no_repeat_ngram_size can only occur once.
        min_length (:obj:`int`, `optional`, defaults to 10): The min length of the sequence to be generated.
            Between 0 and infinity. Default to 10.
        max_length (:obj:`int`, `optional`, defaults to 50): The max length of the sequence to be generated.
            Between min_length and infinity. Default to 50.
        early_stopping (:obj:`bool`, `optional`, defaults to True): bool if set to True beam search is stopped
            when at least num_beams sentences finished per batch. Defaults to False as defined in
            configuration_utils.PretrainedConfig.
        skip_special_tokens (:obj:`bool`, `optional`, defaults to True): Don't decode special tokens
            (self.all_special_tokens). Default: False.
        num_samples (:obj:`int`, `optional`, defaults to 100): Number of samples to pull from dataframe with
            specific feature to use in generating new sample with Abstract Summarization.
        threshold (:obj:`int`, `optional`, defaults to 3500): Maximum ceiling for each feature, normally the
            under-sample max.
        multiproc (:obj:`bool`, `optional`, defaults to True): If set, stores calls for Abstract Summarization in
            array which is then passed to run_cpu_tasks_in_parallel to allow for increasing performance through
            multiprocessing.
        debug (:obj:`bool`, `optional`, defaults to True): If set, prints generated summarizations.
    """
    def __init__(
            self,
            df,
            text_column='text',
            features=None,
            device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
            model=T5ForConditionalGeneration.from_pretrained('t5-small'),
            tokenizer=T5Tokenizer.from_pretrained('t5-small'),
            return_tensors='pt',
            num_beams=4,
            no_repeat_ngram_size=4,
            min_length=10,
            max_length=50,
            early_stopping=True,
            skip_special_tokens=True,
            num_samples=100,
            threshold=3500,
            multiproc=True,
            debug=True
    ):
        self.df = df
        self.text_column = text_column
        self.features = features
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.min_length = min_length
        self.max_length = max_length
        self.early_stopping = early_stopping
        self.skip_special_tokens = skip_special_tokens
        self.num_samples = num_samples
        self.threshold = threshold
        self.multiproc = multiproc
        self.debug = debug
        self.append_index = 0
        self.df_append = None

        # If features passed in, convert to list of strings. Otherwise assume all features
        # aside from text are in play
        if self.features:
            self.features = self.features.split(",")
        else:
            self.features = self.df.columns.tolist()
            self.features.remove(self.text_column)

    def get_abstractive_summarization(self, text):
        """
        Computes abstractive summarization of specified text
        :param text: Text to summarize
        :param debug: Whether to print
        :return: Abstractive summarization text
        """
        t5_prepared_text = "summarize: " + text

        if self.device.type == 'cpu':
            tokenized_text = self.tokenizer.encode(
                t5_prepared_text,
                return_tensors=self.return_tensors).to(self.device)
        else:
            tokenized_text = self.tokenizer.encode(
                t5_prepared_text,
                return_tensors=self.return_tensors)

        summary_ids = self.model.generate(tokenized_text,
                                          num_beams=self.num_beams,
                                          no_repeat_ngram_size=self.no_repeat_ngram_size,
                                          min_length=self.min_length,
                                          max_length=self.max_length,
                                          early_stopping=self.early_stopping)

        output = self.tokenizer.decode(summary_ids[0],
                                       skip_special_tokens=self.skip_special_tokens)

        if self.debug:
            print("\nSummarized text: \n", output)

        return output

    def abs_sum_augment(self):
        """
        Gets append counts (number of rows to append) for each feature and initializes main
        classes' dataframe to be appended to that number of rows. Initializes all feature
        values of said array to 0 to accommodate future one-hot encoding of features. Loops
        over each feature then executes loop to number of rows needed to be appended for
        oversampling to reach needed amount for given feature. If multiproc is set, calls
        to process_abstractive_summarization are stored in a tasks array, which is then passed to a
        function that allows multiprocessing of said summarizations to vastly reduce runtime.
        :return: Dataframe appended with augmented samples to make underrepresented
        features match the count of the majority features.
        """
        counts = self.get_append_counts(self.df)
        # Create append dataframe with length of all rows to be appended
        self.df_append = pd.DataFrame(index=np.arange(sum(counts.values())), columns=self.df.columns)

        # Creating array of tasks for multiprocessing
        tasks = []

        # set all feature values to 0
        for feature in self.features:
            self.df_append[feature] = 0

        for feature in self.features:
            num_to_append = counts[feature]
            for num in range(self.append_index, self.append_index + num_to_append):
                if self.multiproc:
                    tasks.append(self.process_abstractive_summarization(feature, num))
                else:
                    self.process_abstractive_summarization(feature, num)

            # Updating index for insertion into shared appended dataframe to preserve indexing
            # in multiprocessing situation
            self.append_index += num_to_append

        if self.multiproc:
            run_cpu_tasks_in_parallel(tasks)

        return self.df_append

    def process_abstractive_summarization(self, feature, num):
        """
        Samples a subset of rows from main dataframe where given feature is exclusive. The
        subset is then concatenated to form a single string and passed to an abstractive summarizer
        to generate a new data entry for the append count, augmenting said dataframe with rows
        to essentially oversample underrepresented data. df_append is set as a class variable to
        accommodate that said dataframe may need to be shared among multiple processes.
        :param feature: Feature to filter on
        :param num: Count of place in abs_sum_augment loop
        """
        # Pulling rows where only specified feature is set to 1
        df_feature = self.df[(self.df[feature] == 1) & (self.df[self.features].sum(axis=1) == 1)]
        df_sample = df_feature.sample(self.num_samples, replace=True)
        text_to_summarize = ' '.join(df_sample[:self.num_samples]['review_text'])
        new_review = self.get_abstractive_summarization(text_to_summarize)
        self.df_append.at[num, 'review_text'] = new_review
        self.df_append.at[num, feature] = 1

    def get_feature_counts(self, df):
        """
        Gets dictionary of features and their respective counts
        :param df: Dataframe with one hot encoded features to pull categories/features from
        :return: Dictionary containing count of each feature
        """
        shape_array = {}
        for feature in self.features:
            shape_array[feature] = df[feature].sum()
        return shape_array

    def get_append_counts(self, df):
        """
        Gets number of rows that need to be augmented for each feature up to threshold
        :param df: Dataframe with one hot encoded features to pull categories/features from
        :return: Dictionary containing number to append for each category
        """
        append_counts = {}
        feature_counts = self.get_feature_counts(df)

        for feature in self.features:
            if feature_counts[feature] >= self.threshold:
                count = 0
            else:
                count = self.threshold - feature_counts[feature]

            append_counts[feature] = count

        return append_counts


def run_cpu_tasks_in_parallel(tasks):
    """
    Takes array of tasks, loops over them to start each process, then loops over each
    to join them
    :param tasks: Array of tasks or function calls to start and join
    """
    running_tasks = [Process(target=task) for task in tasks]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()


def main():
    # Sample usage
    csv = 'path_to_csv'
    df = pd.read_csv(csv)
    augmentor = Augmentor(df, text_column='text')
    df_augmented = augmentor.abs_sum_augment()
    df_augmented.to_csv(csv.replace('.csv', '-augmented.csv'), encoding='utf-8', index=False)


if __name__ == "__main__":
    main()
