"""
Transforms test data into feature vectors, then evaluates the model. the result is saved in eval_results.txt.

:author: Jan Stanicki
"""

import os
import pickle
import logging
from multiprocessing import Pool, cpu_count

import sklearn
from sklearn.metrics import matthews_corrcoef, multilabel_confusion_matrix
import numpy as np
from tools import *
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from frame_bert_prepare_data import *
from tqdm import tqdm, trange
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule


logging.basicConfig(level=logging.CRITICAL)

class FrameBert(BertForSequenceClassification):
	def __init__(self, config, num_labels=10):
		super().__init__(config, num_labels=10)
		self.classifier = torch.nn.Sequential(torch.nn.Linear(config.hidden_size, num_labels), torch.nn.Softmax())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""""DATA_DIR: The input data dir. Contains the .tsv files (or other data files) for the task."""
DATA_DIR = "../data/frames/data/"

"""BERT_MODEL: choses bert model, in this case the trained fine-tuning model"""
BERT_MODEL = 'frames40ep.tar.gz'

TASK_NAME = 'frames40ep'

"""OUTPUD_DIR: The output directory where the fine-tuned model and checkpoints will be written."""
OUTPUT_DIR = f'../data/frames/outputs/{TASK_NAME}/'

"""REPORTS_DIR: The directory where the evaluation reports will be written to."""
REPORTS_DIR = f'../data/frames/reports/{TASK_NAME}_evaluation_reports/'

"""CHACHE_DIR: This is where BERT will look for pre-trained models to load parameters from."""
CACHE_DIR = '../data/frames/cache/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 512

TRAIN_BATCH_SIZE = 12
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 0
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

if os.path.exists(REPORTS_DIR) and os.listdir(REPORTS_DIR):
        REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
        os.makedirs(REPORTS_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
    REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
    os.makedirs(REPORTS_DIR)

def get_eval_report(task_name, labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    f1_score = sklearn.metrics.f1_score(labels, preds, average='micro')
    print(f1_score)
    precision = sklearn.metrics.precision_score(labels, preds, average='micro')
    print(precision)
    recall = sklearn.metrics.recall_score(labels, preds, average='micro')
    print(recall)
    accuracy = sklearn.metrics.accuracy_score(labels, preds)
    mcm = multilabel_confusion_matrix(labels, preds)
    print(mcm)
    return {
        "task": task_name,
        "mcc": mcc,
        "multilabel_confusion_matrix": mcm,
        "f1_score_micro": f1_score,
        "precision_micro": precision,
        "recall_micro": recall
   }

def compute_metrics(task_name, labels, preds):
    assert len(preds) == len(labels)
    return get_eval_report(task_name, labels, preds)

processor = MultiClassificationProcessor()
eval_examples = processor.get_dev_examples(DATA_DIR)
label_list = processor.get_labels() # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] for Multi classification
num_labels = len(label_list)
eval_examples_len = len(eval_examples)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR + 'vocab.txt', do_lower_case=False)
label_map = {label: i for i, label in enumerate(label_list)}
eval_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in eval_examples]

model = FrameBert.from_pretrained(CACHE_DIR + BERT_MODEL, num_labels=len(label_list))


process_count = cpu_count() - 1
if __name__ ==  '__main__':
    print(f'Preparing to convert {eval_examples_len} examples..')
    print(f'Spawning {process_count} processes..')
    with Pool(process_count) as p:
        eval_features = list(tqdm(p.imap(convert_example_to_feature, eval_examples_for_processing), total=eval_examples_len))
    #eval_features = convert_example_to_feature(f.eval_example_for_processing for f in eval_examples_for_processing )

all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

if OUTPUT_MODE == "classification":
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
elif OUTPUT_MODE == "regression":
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

# Load pre-trained model (weights)
model = FrameBert.from_pretrained(CACHE_DIR + BERT_MODEL, cache_dir=CACHE_DIR, num_labels=len(label_list))

model.to(device)

model.eval()
eval_loss = 0
nb_eval_steps = 0
preds = []

for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask, labels=None)

    # create eval loss and other metric required by the task
    if OUTPUT_MODE == "classification":
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
    elif OUTPUT_MODE == "regression":
        loss_fct = MSELoss()
        tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

    eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    if len(preds) == 0:
        preds.append(logits.detach().cpu().numpy())
    else:
        preds[0] = np.append(
            preds[0], logits.detach().cpu().numpy(), axis=0)

eval_loss = eval_loss / nb_eval_steps
preds = preds[0]
if OUTPUT_MODE == "classification":
    preds = np.argmax(preds, axis=1)
elif OUTPUT_MODE == "regression":
    preds = np.squeeze(preds)
result = compute_metrics(TASK_NAME, all_label_ids.numpy(), preds)

result['eval_loss'] = eval_loss

output_eval_file = os.path.join(REPORTS_DIR, "eval_results.txt")
with open(output_eval_file, "w") as writer:
    logger.info("***** Eval results *****")
    for key in (result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
