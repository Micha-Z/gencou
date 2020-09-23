"""
The Webis-argument-framing-19 corpus is read and preprocessed where stopwords and punctuation gets removed.
Then the arguments are vectorized with TF-IDF and every argument is classified to a cluster.
The most common cluster gets determined and this replaces the frame annotation in the corpus.

:author: Jan Stanicki
"""

import csv
import string
import pickle
import os.path
from collections import defaultdict
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def read_webis(inputpath):
    """
    Reads .csv-file of training corpus.
    :param inputpath: path to the Webis-argument-framing-19 corpus
    :type inputpath: string

    output: list of lists with argument text and frame annotation
    [[frame, argument],[frame, argument]...]
    """

    data = []
    frameSet = set()
    column_names = []
    with open(inputpath, "r") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter = '|')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                column_names = row
                line_count+=1
                pass
            else:
                frame_and_argument = []
                frameSet.add(row[2])
                frame_and_argument.append(row[2])
                frame_and_argument.append(row[5])
                data.append(frame_and_argument)
                line_count+=1

    return data


def preprocessing():
    """
    Removes stopwords and punctuation.

    output: list of lists with argument text and frame annotation
    [[frame, argument],[frame, argument]...]
    """

    data = read_webis("../data/frames/data/Webis-argument-framing.csv")
    stop_words = set(stopwords.words('english'))
    for doc in data:

        doc[1] = ' '.join(word for word in doc[1].split() if word not in stop_words) # remove stopwors from text
        doc[1] = doc[1].lower().strip()
        doc[1] = doc[1].translate(str.maketrans('','', string.punctuation))
    return data


def vectorize_arguments():
    """
    Vectorize and cluster arguments in 10 cluster.

    output: list of lists containing the list of frames, list of arguments, list of predicted clusters
    """

    data = preprocessing()
    tfidf_vectorizer = TfidfVectorizer()
    list_of_all_arguments = []
    list_of_frames = []
    for argument in data:
        list_of_all_arguments.append(argument[1])
        list_of_frames.append(argument[0])
    tfidf = tfidf_vectorizer.fit_transform(list_of_all_arguments)
    kmeans = KMeans(n_clusters=10).fit(tfidf)
    set_of_frames = set(list_of_frames)
    predicted_cluster = []
    for argument in tfidf:
        result = kmeans.predict(argument)
        predicted_cluster.append(int(kmeans.predict(argument)))
    list_of_all_frames_and_arguments_with_cluster = [list(a) for a in zip (list_of_frames, list_of_all_arguments, predicted_cluster)]
    return list_of_all_frames_and_arguments_with_cluster



def frameDict():
    """
    Dictionary with frame as key and as value the list of predicted clusters for every argument belonging to that frame.
    """

    data = vectorize_arguments()
    frameDict = defaultdict(list)
    for argument in data:
        if argument[0] not in frameDict:
            frameDict[argument[0]] = [argument[2]]
        else:
            frameDict[argument[0]].append(argument[2])
    return frameDict

def most_common_cluster():
    """
    Dictionary with frame as key and the most common predicted cluster as value.
    """

    data = frameDict()
    simple_frameDict = {}
    for frame, value in data.items():
        counter = Counter(value)
        most_frequent_cluster = counter.most_common(1) # tuples in list
        simple_frameDict[frame] = most_frequent_cluster
    return simple_frameDict

def create_cluster_sets():
    """
    Creates a set for every cluster and adds frame to the cluster which was the most common prediction, then saves it in .pickle-file.
    """

    data = most_common_cluster()
    cluster0 = set()
    cluster1 = set()
    cluster2 = set()
    cluster3 = set()
    cluster4 = set()
    cluster5 = set()
    cluster6 = set()
    cluster7 = set()
    cluster8 = set()
    cluster9 = set()
    cluster10 = set()
    for frame, cluster in data.items():
        if cluster[0][0] == 0:
            cluster0.add(frame)
        if cluster[0][0] == 1:
            cluster1.add(frame)
        if cluster[0][0] == 2:
            cluster2.add(frame)
        if cluster[0][0] == 3:
            cluster3.add(frame)
        if cluster[0][0] == 4:
            cluster4.add(frame)
        if cluster[0][0] == 5:
            cluster5.add(frame)
        if cluster[0][0] == 6:
            cluster6.add(frame)
        if cluster[0][0] == 7:
            cluster7.add(frame)
        if cluster[0][0] == 8:
            cluster8.add(frame)
        if cluster[0][0] == 9:
            cluster9.add(frame)
        if cluster[0][0] == 10:
            cluster10.add(frame)
    with open(r'../data/frames/data/cluster.pickle', 'wb') as output_file:
        pickle.dump(cluster0, output_file)
        pickle.dump(cluster1, output_file)
        pickle.dump(cluster2, output_file)
        pickle.dump(cluster3, output_file)
        pickle.dump(cluster4, output_file)
        pickle.dump(cluster5, output_file)
        pickle.dump(cluster6, output_file)
        pickle.dump(cluster7, output_file)
        pickle.dump(cluster8, output_file)
        pickle.dump(cluster9, output_file)
    return cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, cluster7, cluster8, cluster9, cluster10


def main():
   read_webis("../data/frames/data/Webis-argument-framing.csv")
   preprocessing()
   vectorize_arguments()
   frameDict()
   most_common_cluster()
   create_cluster_sets()

if __name__ == "__main__":
    main()
