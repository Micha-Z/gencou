"""
Transforms csv-files to tsv-files and includes some helper classes and the function to convert argument texts to feature vectors.

:author: Jan Stanicki
"""

from __future__ import absolute_import, division, print_function
import os
import sys
import logging
import csv

import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification

TRAIN_CORP_CSV = '../data/frames/data/Webis-argument-framing_train.csv'
TRAIN_CORP_TSV = '../data/frames/data/Webis-argument-framing_train.tsv'
TEST_CORP_CSV = '../data/frames/data/Webis-argument-framing_test.csv'
TEST_CORP_TSV = '../data/frames/data/Webis-argument-framing_test.tsv'

train_df = pd.read_csv(TRAIN_CORP_CSV, sep='|', skiprows=1, header=None)

train_df_bert = pd.DataFrame({
    #'id':range(len(train_df)),
    'label':train_df[2],
    'alpha':['a']*train_df.shape[0],
    'text': train_df[5].replace(r'\n', ' ', regex=True)
})

train_df_bert.to_csv(TRAIN_CORP_TSV, sep='\t', index=1, header=False)

test_df = pd.read_csv(TEST_CORP_CSV, sep='|', skiprows=1, header=None)

test_df_bert = pd.DataFrame({
    #'id':range(len(test_df)),
    'label':test_df[2],
    'alpha':['a']*test_df.shape[0],
    'text': test_df[5].replace(r'\n', ' ', regex=True)
})

test_df_bert.to_csv(TEST_CORP_TSV, sep='\t', index=1, header=False)

logger = logging.getLogger()
csv.field_size_limit(2147483647) # Increase CSV reader's field limit incase we have long text.


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs an InputExample.
        Args:
            :param guid: Unique id for the example.
            :param text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            :param text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            :param label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MultiClassificationProcessor(DataProcessor):
    """Processor for multi classification dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "Webis-argument-framing_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "Webis-argument-framing_test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [str(x) for x in range(10)] #["0", "1", "2", "3", "4", "5", "6"] Try:

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_example_to_feature(example_row):
    """
    Converts an argument example to a feature vector.

    :param example_row: tuple of example object, label_map, max_seq_length, tokenizer and output_mode needed for preprocessing
    :type example_row: tuple

    """
    example, label_map, max_seq_length, tokenizer, output_mode = example_row
    output_mode = "classification"
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length


    if hasattr(example_row, 'label_id') and output_mode == "classification":
        label_id = label_map[example.label]
        return InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             label_id=label_id)
    else:
        return InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             label_id=None)
