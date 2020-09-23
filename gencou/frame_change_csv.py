"""
Loads clusters from pickle file and writes a csv

:author: Jan Stanicki
"""

import pickle
import csv

def load_cluster(clusterfile_path):
    """
    Loads the clusters saved in .pickle file and returns it.


    :param clusterfile_path: path to clusterfile
    :type clusterfile_path: string
    """
    with open(clusterfile_path, 'rb') as infile:
        cluster0 = pickle.load(infile)
        cluster1 = pickle.load(infile)
        cluster2 = pickle.load(infile)
        cluster3 = pickle.load(infile)
        cluster4 = pickle.load(infile)
        cluster5 = pickle.load(infile)
        cluster6 = pickle.load(infile)
        cluster7 = pickle.load(infile)
        cluster8 = pickle.load(infile)
        cluster9 = pickle.load(infile)
    return [cluster0, cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, cluster7, cluster8, cluster9]

def write_in_csv(original_corpus_path, modified_corpus_path):
    """
    Assigns every frame name in corpus file to one of the seven classes (frames) and writes it in a corpus file (Webis-argument-framing-simplified.csv).

    :param original_corpus_path: path to the original corpus
    :type original_corpus_path: string

    :param modified_corpus_path: path where the modified corpus is saved
    :type modified_corpus_path: string
    """
    data = load_cluster('../data/frames/data/cluster.pickle')
    cluster0 = data[0]
    cluster1 = data[1]
    cluster2 = data[2]
    cluster3 = data[3]
    cluster4 = data[4]
    cluster5 = data[5]
    cluster6 = data[6]
    cluster7 = data[7]
    cluster8 = data[8]
    cluster9 = data[9]
    with open(original_corpus_path,'r') as f:
        reader = list(csv.reader(f, delimiter='|'))
        for row in reader:
            if row[2] in cluster0: #rights
                row[2] = 0
            if row[2] in cluster1: #intern. politics
                row[2] = 1
            if row[2] in cluster2: #intern. politics
                row[2] = 1
            if row[2] in cluster3: #individual rights
                row[2] = 2
            if row[2] in cluster4: #education
                row[2] = 3
            if row[2] in cluster5: #climate change
                row[2] = 4
            if row[2] in cluster6: #climate change
                row[2] = 4
            if row[2] in cluster7: #climate change
                row[2] = 4
            if row[2] in cluster8: #economics
                row[2] = 5
            if row[2] in cluster9: #health care
                row[2] = 6

    with open(modified_corpus_path,'w') as f:
        wr = csv.writer(f, delimiter='|')
        for row in reader:
            wr.writerow(row)

def main():
	load_cluster('../data/frames/data/cluster.pickle')
	write_in_csv('../data/frames/data/Webis-argument-framing.csv', '../data/frames/data/Webis-argument-framing-simplified.csv')

if __name__ == "__main__":
    main()
