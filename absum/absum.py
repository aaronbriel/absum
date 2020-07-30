import argparse

from box import Box
import numpy as np
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


class AbSumAugmentor(object):
    def __init__(self, args):
        self.args = args
        self.features = args.features
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = T5ForConditionalGeneration.from_pretrained(args.model)
        self.tokenizer = T5Tokenizer.from_pretrained(args.model)

    def get_abstract_summary(self, text, debug=False):
        """
        Computes abstract summarization of specified text
        :param text: Text to summarize
        :param debug: Whether to print
        :return: Abstracted summary text
        """
        t5_prepared_text = "summarize: " + text
        if debug:
            print("original text preprocessed: \n", text)

        if self.device.type == 'cpu':
            tokenized_text = self.tokenizer.encode(
                t5_prepared_text,
                return_tensors=self.args.return_tensors).to(self.device)
        else:
            tokenized_text = self.tokenizer.encode(
                t5_prepared_text,
                return_tensors=self.args.return_tensors)

        summary_ids = self.model.generate(tokenized_text,
                                          num_beams=self.args.num_beams,
                                          no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                          min_length=self.args.min_length,
                                          max_length=self.args.max_length,
                                          early_stopping=self.args.early_stopping)

        output = self.tokenizer.decode(summary_ids[0],
                                       skip_special_tokens=self.args.skip_special_tokens)

        if debug:
            print("\n\nSummarized text: \n", output)

        return output

    def abs_sum_augment(self, df, num_samples=100):
        """
        Gets append counts (number of rows to append) for each feature then samples a
        subset of rows from main dataframe where that feature is exclusive. The subset is
        then concatenated to form a single string and passed to an abstract summarizer to
        generate a new data entry for the append count, augmenting said dataframe with rows
        to essentially oversample underrepresented data.
        :param df: Main dataframe
        :param num_samples: Number of samples to pull
        :return: Dataframe appended with augmented samples to make underrepresented
        features match the count of the majority features.
        """
        import time
        t1 = time.process_time()
        counts = self.get_append_counts(df)
        # Create append dataframe with length of all rows to be appended
        df_append = pd.DataFrame(index=np.arange(sum(counts.values())), columns=df.columns)

        # set all feature values to 0
        for feature in self.features:
            df_append[feature] = 0

        for feature in self.features:
            num_to_append = counts[feature]
            # Pulling rows where only specified feature is set to 1
            df_feature = df[(df[feature] == 1) & (df[self.args.features].sum(axis=1) == 1)]
            for num in range(0, num_to_append):
                df_sample = df_feature.sample(num_samples, replace=True)
                text_to_summarize = ' '.join(df_sample[:num_samples]['review_text'])
                new_review = self.get_abstract_summary(text_to_summarize)
                df_append.at[num, 'review_text'] = new_review
                df_append.at[num, feature] = 1

                print("run_time: ", (time.process_time() - t1))

        return df.append(df_append)

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
            if feature_counts[feature] >= self.args.threshold:
                count = 0
            else:
                count = self.args.threshold - feature_counts[feature]

            append_counts[feature] = count

        return append_counts


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default='./data/google_pp_category.csv',
        type=str,
        required=True,
        help="CSV containing data to augment",
    )

    parser.add_argument(
        "--aug_csv",
        default=None,
        type=str,
        required=False,
        help="Name of CSV to be saved, containing augmented data",
    )

    parser.add_argument(
        "--text",
        default='text',
        type=str,
        required=False,
        help="Column containing text",
    )

    parser.add_argument(
        "--features",
        default=None,
        type=str,
        required=False,
        help="Comma separated string of features",
    )

    parser.add_argument(
        "--threshold",
        default=None,
        type=int,
        required=False,
        help="Maximum ceiling for each feature, normally the undersample max.",
    )

    parser.add_argument(
        "--model",
        default='t5-small',
        type=str,
        required=False,
        help="Base model for T5 model and tokenizer",
    )

    parser.add_argument(
        "--return_tensors",
        default='pt',
        type=str,
        required=False,
        help="Can be set to ‘tf’, ‘pt’ or ‘np’ to return respectively TensorFlow tf.constant, PyTorch torch.Tensor "
             "or Numpy :oj: np.ndarray instead of a list of python integers.",
    )

    parser.add_argument(
        "--num_beams",
        default=4,
        type=int,
        required=False,
        help="Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.",
    )

    parser.add_argument(
        "--no_repeat_ngram_size",
        default=4,
        type=int,
        required=False,
        help="If set to int > 0, all ngrams of size no_repeat_ngram_size can only occur once.",
    )

    parser.add_argument(
        "--min_length",
        default=10,
        type=int,
        required=False,
        help="The min length of the sequence to be generated. Between 0 and infinity. Default to 10.",
    )

    parser.add_argument(
        "--max_length",
        default=30,
        type=int,
        required=False,
        help="The max length of the sequence to be generated. Between min_length and infinity. Default to 30.",
    )

    parser.add_argument(
        "--early_stopping",
        default=4,
        type=bool,
        required=False,
        help="bool if set to True beam search is stopped when at least num_beams sentences finished per batch. "
             "Defaults to False as defined in configuration_utils.PretrainedConfig.",
    )

    parser.add_argument(
        "--skip_special_tokens",
        default=True,
        type=bool,
        required=False,
        help="Don’t decode special tokens (self.all_special_tokens). Default: False.",
    )

    cl_args = parser.parse_args()
    args = Box(vars(cl_args))

    csv = cl_args.csv

    aug_csv = cl_args.aug_csv
    if not aug_csv:
        aug_csv = csv.replace('.csv', '-augmented.csv')
        args.aug_csv = aug_csv

    df_ = pd.read_csv(csv)

    # If features passed in, convert to list of strings. Otherwise assume all features aside from text are in play
    if cl_args.features:
        args.features = cl_args.features.split(",")
    else:
        args.features = df_.columns.tolist()
        args.features.remove(args.text)

    augmentor = AbSumAugmentor(args)
    df_augmented = augmentor.abs_sum_augment(df_)
    df_augmented.to_csv(aug_csv)


if __name__ == "__main__":
    main()
