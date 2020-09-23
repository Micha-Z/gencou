"""
Splits Corpus in train and test file ( 4:1 )

:author: Jan Stanicki
"""

from numpy.random import RandomState
import pandas as pd

def train_test(in_path, train_out_path, test_out_path):
    """
    reads corpus file and splits data in 80% train and 20% test and saves it in respective files
    """
    df = pd.read_csv(in_path, sep='|')
    rng = RandomState()

    train = df.sample(frac=0.8, random_state=rng)
    test = df.loc[~df.index.isin(train.index)]

    train.to_csv(train_out_path, sep='|', index = None, header=True)
    test.to_csv(test_out_path, sep='|', index = None, header=True)
if __name__ == "__main__":
    train_test('../data/frames/data/Webis-argument-framing-simplified.csv', '../data/frames/data/Webis-argument-framing_train.csv', '../data/frames/data/Webis-argument-framing_test.csv')
