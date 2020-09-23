#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Small interface for the pretrained args-me_15000_graph.

If you want to use your own sentences for the demo,
feel free to adjust /data/arguments.txt, don't change the formating though:

- first line : argument
- second line: stance
- repeat

For own trainings refer to the documentation.

:author: Julius Unverfehrt
:version: 0.9
"""

import logging
from pprint import pprint
# local
from akglib import AKG

if __name__ == "__main__":

    logging.basicConfig(
        filename="../data/interface.log", level=logging.CRITICAL, filemode="w"
    )
    # disable clutter logging from allennlp
    for allog in logging.root.manager.loggerDict:
        if "allennlp" in allog:
            logging.getLogger(allog).disabled = True

    print("\ngencou - pretrained 15k")
    print("\n")
    print("Loading data...")
    AKG = AKG()
    AKG.load_corpus("../data/akg/args-me_15000.json")
    AKG.load(graph=True)

    while True:
        print("\n")
        print("[1] Batch")
        print("[2] Interactive")
        print("[3] Exit")
        print("\n")
        COMMAND = input(">>> ")
        if COMMAND == "1":
            ARGUMENT, STANCE = [], []
            print("Reading ../data/arguments.txt")
            with open("../data/arguments.txt", "r") as openfile:
                for idx, line in enumerate(openfile.readlines()):
                    if not idx % 2 or idx == 0:
                        ARGUMENT.append(line.strip())
                    else:
                        STANCE.append(line.strip())
            COUNTER = 0
            for arg, stc in zip(ARGUMENT, STANCE):
                result = []
                filepath = "../data/triples/" + str(COUNTER)
                print(f"\nArgument: {arg}, {stc}\n")
                COUNTERTRIPLES = AKG.get_counter_triples(arg, stc, filepath)
                pprint(COUNTERTRIPLES)
                COUNTER += 1

        if COMMAND == "2":
            print("Insert Argument")
            ARG = input(">>> ")
            print("Insert stance")
            STA = input(">>> ")
            COUNTERTRIPLES = AKG.get_counter_triples(ARG, STA)
            print("\n")
            print("Resulting triples:")
            pprint(COUNTERTRIPLES)

        if COMMAND == "3":
            break
