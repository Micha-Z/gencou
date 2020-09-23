#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Provides the class AKG, representing an argument knowledge graph with the methods
needed to create and access it.

The graph works with the args-me corpus but could be used with any corpus,
small adjustments assumed.
Further detail on how to use this class is provided in the Getting started section
of the documentation.

:author: Julius Unverfehrt
:version: 0.9
"""

import json
import functools
import random
import os
import os.path
import logging
import pickle
from difflib import SequenceMatcher
from collections import defaultdict

import igraph
import spacy
import neuralcoref
import progressbar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from allennlp.pretrained import open_information_extraction_stanovsky_2018

# local
import frame_bert_interface


class AKG:
    """This class provides all methods needed to process the args-me corpus into
    an argument knowledge graph. Coreferences in the documents are resolved with
    neuralCoref, the information is extracted with supervised-openIE into triples:
    SUBJ-VERB-OBJ, where SUBJ and OBJ will be nodes in the graph, with VERB as edge.
    When put into the graph, similar nodes (cosine TF_IDF) are merged (only once per existing node,
    to prevent information loss).
    After creation, querys of argument + stance can be made, to recive matching
    triples that oppose the argument.

    :param corpus_path: Path to the corpus (or sample), defaults to "../data/args-me.json"
    :type corpus_path: string
    """

    def __init__(self, corpus_path="../data/args-me.json"):
        self.graph = igraph.Graph(directed=True)
        self.graph.vs["name"] = []
        self.graph.vs["argument"] = []
        self.graph.es["name"] = []
        self.nlp = spacy.load("en")
        neuralcoref.add_to_pipe(self.nlp, greedyness=0.5)
        self.openie = open_information_extraction_stanovsky_2018()
        self.data = defaultdict(list)
        self.data_proc = list()
        self.merges = dict()
        self.corpus_path = corpus_path
        logging.debug("AKG initialized...")

    def load_corpus(self, corpus_path=None):
        """Loads the (args.me-)corpus, provided in .json format.
        Uses the path given from class initialization, alternativly provide one
        here.

        :param corpus_path: Path to the corpus (or sample), defaults to None
        :type corpus_path: string
        :raises TypeError: invalid corpus_path format
        :raises OSError: filesystem errors
        """
        if corpus_path:
            self.corpus_path = corpus_path
        else:
            corpus_path = self.corpus_path

        if not isinstance(corpus_path, str) or not corpus_path.endswith(".json"):
            raise TypeError("Provide a valid .json filepath!")

        with open(corpus_path, "r") as openfile:
            corpus = json.load(openfile)

        if self.data:
            self.data.clear()
        self.data.update(corpus)
        self.data["length"] = len(self.data["arguments"])

    def process_arguments(self):
        """Main routine for processing the corpus.

        Iterates over each argument and executes neuralCoref and supervised-OpenIE.
        Sentences with triple indicator marks are saved for later, when a
        Language model which transforms triples into sentences will be trained (other module).
        Every 1000 arguments, the processed data will be saved, using the location
        provided from corpus_path.

        :raises AttributeError: corpus not loaded previously
        :raises TypeError: args-me structure not detected
        :raises KeyError: args-me arguments key not found
        """

        if not self.data:
            raise AttributeError("No Corpus detected.")
        if not isinstance(self.data, dict):
            raise TypeError("Corpus should be a dictionary.")
        if "arguments" not in self.data.keys():
            raise KeyError("Arguments not found.")

        counter = 0
        length = self.data["length"]

        for argument in self.data["arguments"]:
            argument_proc = defaultdict(list)
            for sentence in self.__filter_sentences(argument["premises"][0]["text"]):
                triple, triple_marked, sentence_marked = self.__select_triple(sentence)
                # Although it shouldn't, supervised-openIE often enough provides
                # empty triples or nodes, we ignore them here:
                if not triple or not triple_marked or not sentence_marked:
                    continue
                if not triple[0] or not triple[1] or not triple[2]:
                    continue

                argument_proc["sentences"].append(sentence)
                argument_proc["triples"].append(triple)
                argument_proc["sentences_marked"].append(sentence_marked)
                argument_proc["triples_marked"].append(triple_marked)

            self.data_proc.append(argument_proc)

            counter += 1
            if not counter % 10:
                logging.debug(
                    "supervised-openIE: %s / %s arguments processed.", counter, length
                )

            if not counter % 1000:
                self.save(data_proc=True)

    def save(self, data_proc=False, graph=False):
        """Save processed data and/or the processed graph, using the location
        provided from corpus_path.

        :param data_proc: should processed data be saved, defaults to False
        :type data_proc: boolean
        :param graph: should the graph be saved, defaults to False
        :type graph: boolean

        :raises OSError: if not able to write
        """
        if data_proc:
            filename = os.path.splitext(self.corpus_path)[0] + "_data_proc.pickle"
            with open(filename, "wb") as openfile:
                pickle.dump(self.data_proc, openfile)
            logging.debug("Processed corpus saved to: %s", filename)
        if graph:
            filename = os.path.splitext(self.corpus_path)[0] + "_graph.pickle"
            with open(filename, "wb") as openfile:
                pickle.dump(self.graph, openfile)
            logging.debug("Graph saved to: %s", filename)

    def load(self, data_proc=False, graph=False):
        """Load processed data and/or the processed graph, using the location
        provided from corpus_path.

        :param data_proc: should processed data be loaded, defaults to False
        :type data_proc: boolean
        :param graph: should the graph be loaded, defaults to False
        :type graph: boolean

        :raises OSError: if files not found / unable to open them
        """
        if data_proc:
            filename = os.path.splitext(self.corpus_path)[0] + "_data_proc.pickle"
            with open(filename, "rb") as openfile:
                self.data_proc = pickle.load(openfile)
            logging.debug("Processed corpus loaded.")
        if graph:
            filename = os.path.splitext(self.corpus_path)[0] + "_graph.pickle"
            with open(filename, "rb") as openfile:
                self.graph = pickle.load(openfile)
            logging.debug("Graph loaded.")

    def construct_graph(self):
        """Iterates over the processed corpus and adds the triples found by
        supervised-openIE to the igraph.

        If similarity (cosine tf-idf) to a existing node is detected, the nodes will be merged
        into the shorter one (only once per existing node, to restrict information loss).
        The merges are therefore memorized, which also improves runtime.

        :raises AttributeError: corpus not loaded previously and/or not processed
        """
        if not self.data or not self.data_proc:
            raise AttributeError("Load and process corpus first!")

        counter = 0

        for index, argument in enumerate(self.data_proc):
            triples = argument["triples"]
            nodes = set()
            for triple in triples:
                args = []
                for arg in [triple[0], triple[2]]:
                    if arg in self.graph.vs["name"]:
                        pass
                    elif arg in self.merges.keys():
                        logging.debug("MERGE lookup: " + arg + "-->" + self.merges[arg])
                        arg = self.merges[arg]
                    elif arg in args:
                        pass
                    else:
                        arg_sim = self.get_cos_sims(arg)
                        arg_sim_max = max(arg_sim)

                        if arg_sim_max > 0.95:
                            max_index = arg_sim.index(arg_sim_max)
                            graph_node = self.graph.vs["name"][max_index]
                            if (len(graph_node) <= len(arg)) or (
                                    graph_node in self.merges.values()
                            ):
                                logging.debug(
                                    "MERGE new: " + arg + " --> " + graph_node
                                )
                                self.merges[arg] = graph_node
                                arg = graph_node
                            else:
                                self.graph.vs[max_index]["name"] = arg
                                self.merges[graph_node] = arg
                                logging.debug(
                                    "MERGE graph change: " + graph_node + " --> " + arg
                                )

                        else:
                            self.graph.add_vertex(name=arg)
                    args.append(arg)
                try:
                    # Somtimes nodes that should be there are not found, the
                    # problem should be the get_cos_sims method which deletes
                    # nodes it should not. Workaround here:
                    self.graph.add_edge(args[0], args[1], name=triple[1])
                except:
                    self.graph.add_vertex(name=args[0])
                    self.graph.add_vertex(name=args[1])
                    self.graph.add_edge(args[0], args[1], name=triple[1])

                arg0_idx = self.graph.vs["name"].index(args[0])
                arg1_idx = self.graph.vs["name"].index(args[1])
                nodes.add(arg0_idx)
                nodes.add(arg1_idx)
                try:
                    self.graph.vs[arg0_idx]["argument"].add(index)
                except:
                    self.graph.vs[arg0_idx]["argument"] = set(index)
                try:
                    self.graph.vs[arg1_idx]["argument"].add(index)
                except:
                    self.graph.vs[arg1_idx]["argument"] = set(index)

            self.data_proc[index]["nodes"] = nodes

            counter += 1
            if not counter % 10:
                logging.debug(
                    "igraph: %s / %s arguments processed", counter, self.data["length"]
                )

            if not counter % 5000:
                self.save(data_proc=True, graph=True)

    def get_cos_sims(self, arg):
        """Aquires the tf-idf cosine similarites for a new node,
        respective to the nodes of all arguments in the graph. Returns a list with the similarities,
        the indices refering to the nodes indices of the graph.
        The new node is temporarily added to the graph, which prevents creating
        new lists every time a similarity check is made.

        :param arg: one single node
        :type arg: string
        :return: similiarities
        :rtype: list(float)
        """

        if len(self.graph.vs["name"]) < 1:
            return [0]

        self.graph.add_vertex(name=arg)
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(self.graph.vs["name"])
        arg_sim = cosine_similarity(matrix[-1], matrix)[0][:-1]
        self.graph.delete_vertices(arg)

        return list(arg_sim)

    def get_counter_triples(self, query, stance, filename=None):
        """Iterates over the graph to identify similar nodes to the query.

        For the sake of runtime, we use a simple sequence match for that task,
        which is way more efficient than cosine similarity and yet yields comparable
        results.
        Triples in the neighborhood of identified nodes are estimated as qualified if their stance
        oposes the queries one.
        They are then sorted by the normalized scores of:

        - respective similarity (cosine TF-IDF), assuming that good counter arguments are similar

        - pagerank, assuming that a high pagerank hints to a meaningful node

        - frames, assuming that good counterarguments are in the same frame

        IMPORTANT: specify a filename if you want to process the triples later,
        since seq2seq isn't integrated.

        :param query: argument to which we want the counter argument
        :type query: string
        :param stance: stance of the argument
        :type stance: string
        :param filename: to saved results on hd, provide a path here, defaults to None
        :type filename: string
        :return: best counter triples (max 10)
        :rtype: list(tuple(string))

        :raises TypeError: query or stance are no strings
        :raises AttributeError: corpus and/or graph not loaded
        :raises ValueError: no triples extracted from query (might be too short)
        :raises OSError: only if filepath is specified
        """

        if not isinstance(query, str) or not isinstance(stance, str):
            raise TypeError("Query and stance need to be strings.")
        if not self.data:
            raise AttributeError("Load corpus, needed for stance lookup.")
        if not self.graph:
            raise AttributeError("No graph found.")

        print("Getting query frame...")
        query_frame = int(frame_bert_interface.classify_single_argument(query))
        print("Search in graph & getting frames (this might take a while)...\n")
        pbar = progressbar.ProgressBar(max_value=len(self.graph.vs["name"]))
        counter_triples = set()
        query_triples = set()
        for sentence_query in self.__filter_sentences(query):
            triple_query, _tm, _sm = self.__select_triple(sentence_query)
            if not triple_query or not triple_query[0] or not triple_query[2]:
                continue
            query_triples.add(triple_query)

        if not query_triples:
            raise ValueError("No triples extracted from query (might be too short).")

        for node_idx, node in enumerate(self.graph.vs["name"]):
            if node.lower() in ["i", "you", "he", "she", "it", "we", "they", "there"]:
                # there is no information gained from these nodes -> ignored
                continue
            for triple in query_triples:
                s_arg0 = SequenceMatcher(0, triple[0], node)
                s_arg1 = SequenceMatcher(0, triple[0], node)
                if s_arg0.ratio() > 0.8 or s_arg1.ratio() > 0.8:
                    for arg_idx in self.graph.vs[node_idx]["argument"]:
                        if (
                                self.data["arguments"][arg_idx]["premises"][0]["stance"]
                                != stance.upper()
                        ):
                            arg_frame = int(
                                frame_bert_interface.classify_single_argument(
                                    self.data["arguments"][arg_idx]["premises"][0][
                                        "text"
                                    ]
                                )
                            )
                            for nbr in self.graph.neighbors(node, mode="OUT"):
                                counter_triples.add(
                                    (
                                        self.graph.vs["name"][node_idx],
                                        self.graph.es["name"][
                                            self.graph.get_eid(node_idx, nbr)
                                        ],
                                        self.graph.vs["name"][nbr],
                                        arg_frame,
                                    )
                                )
                            for nbr in self.graph.neighbors(node, mode="IN"):
                                counter_triples.add(
                                    (
                                        self.graph.vs["name"][nbr],
                                        self.graph.es["name"][
                                            self.graph.get_eid(nbr, node_idx)
                                        ],
                                        self.graph.vs["name"][node_idx],
                                        arg_frame,
                                    )
                                )
            pbar.update(node_idx)
        print("\n\nEvaluate results:\n")
        if not counter_triples:
            print("Sadly no matches found :(")
            return []
        frames = []
        counter_triples = list(counter_triples)
        for idx, triple in enumerate(counter_triples):
            frames.append(triple[3])
            counter_triples[idx] = (triple[0], triple[1], triple[2])

        print("Pagerank...")
        pagerank = [
            self.graph.pagerank()[self.graph.vs["name"].index(triple[0])]
            + self.graph.pagerank()[self.graph.vs["name"].index(triple[2])]
            for triple in counter_triples
        ]
        pagerank_norm = self._normalize(pagerank)
        print("Similarity...")
        cos_list = [" ".join(triple) for triple in counter_triples]
        cos_list.append(" ".join([" ".join(triple) for triple in query_triples]))
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(cos_list)
        arg_sim = cosine_similarity(matrix[-1], matrix)[0][:-1]
        arg_sim_norm = self._normalize(arg_sim)
        print("Frames...")
        frame_norm = list()
        for frame, triple in zip(frames, counter_triples):
            if frame == query_frame:
                frame_norm.append(1)
            else:
                frame_norm.append(0)
        print("Measuring...")
        mesured_list = [
            0.5 * a + b + c for a, b, c in zip(pagerank_norm, arg_sim_norm, frame_norm)
        ]
        result = sorted(
            zip(mesured_list, counter_triples), key=lambda x: x[0], reverse=True
        )
        result = [entry[1] for entry in result if entry[0] > 0]

        if len(result) > 10:
            result = result[:10]

        if filename:
            if os.path.isfile(filename):
                os.remove(filename)
            with open(filename, "a") as openfile:
                for entry in result:
                    openfile.write(" ".join(entry) + "." + "\n")
            print("Result saved.")

        return result

    def _normalize(self, seq):
        amin, amax = min(seq), max(seq)
        for i, val in enumerate(seq):
            if amax - amin == 0:
                seq[i] = 0
            else:
                seq[i] = (val - amin) / (amax - amin)
        return seq

    def write_seq2seq_train(self, marked=True):
        """
        Writes the training data for the seq2seq model, optionally with markers
        for the argument positions. Uses corpus_path for the save path.

        :param marked: if the positions of the arguments should be marked, defaults to True
        :type marked: boolean
        :raises OSError: No access to writefile, or self.data_proc not provided / corrupt
        """
        if marked:
            filename1 = os.path.splitext(self.corpus_path)[0] + "_seq2seq_marked.txt"
            with open(filename1, "w") as openfile:
                for datapoint in self.data_proc:
                    for sentence, triple in zip(
                            datapoint["sentences_marked"], datapoint["triples_marked"]
                    ):
                        triple = " ".join(list(triple))
                        openfile.write(triple + "\t->\t" + sentence + "\n")
        else:
            filename = os.path.splitext(self.corpus_path)[0] + "_seq2seq_plain.txt"
            with open(filename, "w") as openfile:
                for datapoint in self.data_proc:
                    for sentence, triple in zip(
                            datapoint["sentences"], datapoint["triples"]
                    ):
                        triple = " ".join(list(triple))
                        openfile.write(triple + "\t->\t" + sentence + "\n")

    def _write_corpus_sample(self, sample_size=100):
        """Create a shorter sample of the loaded corpus, ment only for developing
        purposes.
        """

        if not self.data or "arguments" not in self.data.keys():
            print("Load corpus first!")
            return

        output = dict()
        idx = random.randint(0, len(self.data["arguments"]) - sample_size)
        output["arguments"] = [
            self.data["arguments"][i] for i in range(idx, idx + sample_size)
        ]
        outputpath = (
            os.path.splitext(self.corpus_path)[0] + "_sample_" + sample_size + ".json"
        )
        with open(outputpath, "w") as openfile:
            json.dump(output, openfile, indent=2)
            logging.debug("Sample of size " + sample_size + "written to\n" + outputpath)

    def __select_triple(self, sentence):
        """Applies supervised-openIE on a given sentence. Returns a list of relevant
        tokens and a list with their tags. Non relevent tokens (tag == 0) are dropped.

        :param input: single sentence
        :type input: string
        :return: list of tokens, list of tags
        :rtype: list, list
        """
        # Mark the positions of verbs
        sent_tokens = self.openie._tokenizer.tokenize(sentence)
        pred_idcs = [i for (i, t) in enumerate(sent_tokens) if t.pos_ == "VERB"]
        instances = [
            {"sentence": sent_tokens, "predicate_index": pred_idx}
            for pred_idx in pred_idcs
        ]
        if len(instances) < 1:
            return None, None, None
        # Predict confidence for different splits
        preds = self.openie.predict_batch_json(instances)
        confs, sents, tags = [], [], []
        for pred in preds:
            pred_tag_indices = [
                self.openie._model.vocab.get_token_index(tag, namespace="labels")
                for tag in pred["tags"]
            ]
            pred_probs = []
            for tag, probs in zip(pred_tag_indices, pred["class_probabilities"]):
                pred_probs.append(probs[tag])
            confs.append(functools.reduce(lambda x, y: x * y, pred_probs))
            sents.append(pred["words"])
            tags.append(pred["tags"])

        conf_max_idx = confs.index(max(confs))
        sent_max = sents[conf_max_idx]
        tag_max = tags[conf_max_idx]

        arg0, pred, arg1, arg2 = [], [], [], []
        sentence_marked = []
        for token, token_tag in zip(sent_max, tag_max):
            if "O" in token_tag:
                sentence_marked.append(token)
            elif "ARG0" in token_tag:
                arg0.append(token)
                sentence_marked.append(token + "**0")
            elif "V" in token_tag:
                pred.append(token)
                sentence_marked.append(token + "**1")
            elif "ARG1" in token_tag:
                arg1.append(token)
                sentence_marked.append(token + "**2")
            elif "ARG2" in token_tag:
                arg2.append(token)
                sentence_marked.append(token + "**3")
            else:
                sentence_marked.append(token)

        arg0_marked = [arg + "**0" for arg in arg0] if arg0 else []
        pred_marked = [arg + "**1" for arg in pred] if pred else []
        arg1_marked = [arg + "**2" for arg in arg1] if arg1 else []
        arg2_marked = [arg + "**3" for arg in arg2] if arg2 else []

        triple_marked = (
            " ".join(arg0_marked),
            " ".join(pred_marked),
            " ".join(arg1_marked + arg2_marked),
        )
        sentence_marked = " ".join(sentence_marked)
        triple = (" ".join(arg0), " ".join(pred), " ".join(arg1 + arg2))

        return triple, triple_marked, sentence_marked

    def my_plot(self):
        """Plot the graph with predifined configuration, might take a while for
        big graphes.
        """
        self.graph.vs["label"] = self.graph.vs["name"]
        self.graph.es["label"] = self.graph.es["name"]
        igraph.plot(
            self.graph,
            target="akg.png",
            bbox=(3000, 3000),
            vertex_frame_width=0,
            edge_color="gray",
            vertex_color="gray",
            vertex_size=10,
            vertex_label_dist=0,
            vertex_shape="circle",
            margin=40,
            autocurve=True,
        )

    def __filter_sentences(self, sentences, resolve_coref=True):
        """Applies neuralCoref to a text and yields its sentences.
        """
        doc = self.nlp(sentences)
        if resolve_coref and doc._.has_coref:
            doc = self.nlp(doc._.coref_resolved)
        for sent in doc.sents:
            # Sentences without verb and punctuation are dropped.
            tokens = [token.pos_ for token in sent]
            if not ("VERB" in tokens and "PUNCT" in tokens):
                continue

            yield sent.text
