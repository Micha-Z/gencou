"""
This interface classifies a single argument and assigns a frame. It's the interface to be integrated in other modules.

:author: Jan Stanicki
"""

import os
from multiprocessing import Pool, cpu_count


import numpy as np
from tqdm import tqdm, trange
from tools import *
import torch
from torch.nn import Softmax
from torch.utils.data import TensorDataset
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from frame_bert_prepare_data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MAX_SEQ_LENGTH = 512
BERT_MODEL = 'frames40ep.tar.gz'
TASK_NAME = 'frames40ep'
OUTPUT_MODE = 'classification'
OUTPUT_DIR = f'../data/frames/outputs/{TASK_NAME}/'
CACHE_DIR = '../data/frames/cache/'

class FrameBert(BertForSequenceClassification):
    """
    BertForSequenceClassification extended with a softmax layer
    """
    def __init__(self, config, num_labels=10):
        super().__init__(config, num_labels=10)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(config.hidden_size, num_labels), torch.nn.Softmax(dim=1))

def classify_single_argument(input_argument):
    """
    takes an input_argument, processes that to a feature vector and classifies this.
    returns prediction.

    :param input_argument: is the argument to be classified.
    :type input_argument: string
    """



    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR + 'vocab.txt', do_lower_case=False)

    processor = MultiClassificationProcessor()
    eval_example = InputExample(guid=101, text_a=input_argument)
    label_list = processor.get_labels() # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] for Multi classification
    num_labels = len(label_list)
    label_map = {label: i for i, label in enumerate(label_list)}
    eval_example_for_processing = (eval_example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE)
    process_count = cpu_count() - 1

    eval_features = convert_example_to_feature(eval_example_for_processing)

    input_ids_ = torch.tensor(eval_features.input_ids, dtype=torch.long)
    input_ids_ = input_ids_.unsqueeze(0)
    input_mask_ = torch.tensor(eval_features.input_mask, dtype=torch.long)
    input_mask_ = input_mask_.unsqueeze(0)
    segment_ids_ = torch.tensor(eval_features.segment_ids, dtype=torch.long)
    segment_ids_ = segment_ids_.unsqueeze(0)
    eval_data = TensorDataset(input_ids_, input_mask_, segment_ids_)

    # Load pre-trained model (weights)
    model = FrameBert.from_pretrained(CACHE_DIR + BERT_MODEL, cache_dir=CACHE_DIR, num_labels=len(label_list))
    model.to(device)

    preds = []

    input_ids_ = input_ids_.to(device)
    input_mask_ = input_mask_.to(device)
    segment_ids_ = segment_ids_.to(device)

    with torch.no_grad():
        logits = model(input_ids_, segment_ids_, input_mask_, labels=None)

    if len(preds) == 0:
        preds.append(logits.detach().cpu().numpy())
    else:
        preds[0] = np.append(
            preds[0], logits.detach().cpu().numpy(), axis=0)
    preds = preds[0]
    preds = np.argmax(preds, axis=1)

    return preds
