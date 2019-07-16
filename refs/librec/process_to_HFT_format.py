'''
Data pre process part2
@author:
Chong Chen (cstchenc@163.com)

@ created:
27/8/2017
'''

import numpy as np
import re
import csv
import os
import pickle
import argparse

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_HFT_format(data, review_pk, vocab, stopwords, output_file):
    f_data = open(data, 'r')
    with open(output_file, "a") as fp:
        for i, line in enumerate(f_data):
            line = line[:-1].split(' ')
            uid = int(line[0])
            iid = int(line[1])
            rating = float(line[2])
            review = review_pk[(uid, iid)]
            review_ids = []
            text = []
            for sent in review:
                #print(sent)
                s1 = clean_str(sent)
                s1 = s1.split(" ")
                text.extend(s1)
            for word in text:
                if word in stopwords:
                    continue
                if word in vocab:
                    review_ids.append(str(vocab[word]))
                else:
                    vocab[word] = len(vocab)
                    review_ids.append(str(vocab[word]))

            review_str = ":".join(review_ids)
            if not review_str:
                review_str = "0"
            fp.write("%s,%s,%s,%s\n" % (uid, iid, rating, review_str))
    return vocab

def load_data(train_data, valid_data, test_data, review_summary, stopwords, domain):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files

    review_summary = pickle.load(open(review_summary))

    with open(stopwords, 'r') as fp:
        stopwords_list = fp.read().splitlines()

    vocab = {}
    train_file = "data/%s/%s_train_HFT.arff" % (domain, domain)

    print("Creating HFT style training data")
    vocab = get_HFT_format(train_data, review_summary, vocab, stopwords_list, train_file)
    print("Creating HFT style validation data")
    vocab = get_HFT_format(valid_data, review_summary, vocab, stopwords_list, train_file)
    print("Creating HFT style test data")
    vocab = get_HFT_format(test_data, review_summary, vocab, stopwords_list, train_file)

def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Process Amazon data')

    parser.add_argument('-d', '--data_set', type=str, help='Dataset music, tv, electronics, kindle, toys', required=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()


    valid_data = "data/%s/test/%s_valid.csv" % (args.data_set, args.data_set)
    test_data = "data/%s/test/%s_test.csv" % (args.data_set, args.data_set)
    train_data = "data/%s/%s_train.csv" % (args.data_set, args.data_set)
    review_summary = "../../DeepCoNN/data/%s/review_summary" % (args.data_set)
    stopwords = "data/stopwords.txt"

    load_data(train_data, valid_data, test_data, review_summary, stopwords, args.data_set)


