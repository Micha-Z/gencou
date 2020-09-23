#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Short snippet to batch create the argument knowledge graph from akglib.py.
 Adjust to your liking and make sure to refer to the documentation.

The following order should be kept:

1. Create the class (specify the path to your args-me corpus as parameter)
2. Load the corpus AKG.load_corpus
3. Process the corpus AKG.process_arguments
4. Construct the graph AKG.construct_graph
5. write the training data for the seq2seq model AKG.write_seq2seq_train

:author: Julius Unverfehrt
:version: 0.9
"""

import logging
from akglib import AKG

if __name__ == "__main__":

    # Insert the path to the args-me corpus (or sample) here
    CORPUSPATH = "../data/args-me.json"

    logging.basicConfig(
        filename="../data/akg.log", level=logging.DEBUG, filemode="w"
    )
    # disable clutter logging from allennlp
    for allog in logging.root.manager.loggerDict:
        if "allennlp" in allog:
            logging.getLogger(allog).disabled = True

    AKG = AKG()
    AKG.load_corpus(CORPUSPATH)
    AKG.process_arguments()
    # save the processed corpus, disable if you don't need this save
    AKG.save(data_proc=True)

    AKG.construct_graph()
    AKG.save(data_proc=True, graph=True)

    AKG.write_seq2seq_train()

    # short test
    # TEXT = ("Superman is stronger than batman. He has natural superpowers.", "PRO")
    # RESULT = AKG.get_counter_triples(*TEXT)
    # print(RESULT)
