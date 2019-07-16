import numpy as np
import re
import itertools
from collections import Counter
import csv
import os
import pickle
import argparse
import spacy
import sys
import tensorflow as tf
import codecs
from nltk import word_tokenize

def mkdirp(path):
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        pass



def get_items(data_file, review_summaries):
    fp = codecs.open(data_file, "r")
    uid = []
    iid = []
    y = []
    review_list = []
    summary_list = []
    
    for i, line in enumerate(fp):
        line = line.split(',')
        user_id = int(line[0])
        item_id = int(line[1])
        
        review, summary = review_summaries[(user_id, item_id)]
        review_tokens = word_tokenize(review)
        summary_tokens = word_tokenize(summary)
        
        if len(review_tokens) < 10 or len(summary_tokens) < 4:
            continue
        
        uid.append(user_id)
        iid.append(item_id)
        
        review_list.append(" ".join(review_tokens))
        summary_list.append(" ".join(summary_tokens))
        y.append(float(line[2]))
    
    
    print("Length of original data:", i + 1)
    print("Length of data:", len(uid))
    return np.array(uid), np.array(iid), np.array(y), np.array(review_list), np.array(summary_list)

def load_data(train_data, valid_data, test_data, review_summary):

    f1 = open(review_summary)
    review_summaries = pickle.load(f1)

    uid_train, iid_train, y_train, review_train, summary_train = get_items(train_data, review_summaries)
    uid_valid, iid_valid, y_valid, review_valid, summary_valid = get_items(valid_data, review_summaries)
    uid_test, iid_test, y_test, review_test, summary_test = get_items(test_data, review_summaries)
              
    return [y_train, y_valid, y_test, uid_train,
            iid_train, uid_valid, iid_valid, uid_test, iid_test,
             review_train, summary_train, review_valid, summary_valid, review_test, summary_test]


def get_args():
    ''' This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(description='Process Amazon and Yelp data')
    parser.add_argument('-d', '--data_set', type=str, help='Dataset music, tv, electronics, kindle, toys, cds, yelp', required=True)
    args = parser.parse_args()
    return args
                        
if __name__ == '__main__':
    
    args = get_args()
    TPS_DIR = '../data/%s' %(args.data_set)
    
    tf.flags.DEFINE_string("valid_data","../data/%s/%s_valid.csv" % (args.data_set, args.data_set), " Data for validation")
    tf.flags.DEFINE_string("test_data", "../data/%s/%s_test.csv" % (args.data_set, args.data_set), "Data for testing")
    tf.flags.DEFINE_string("train_data", "../data/%s/%s_train.csv" % (args.data_set, args.data_set), "Data for training")
    tf.flags.DEFINE_string("review_summary", "../data/%s/review_summary" % (args.data_set), "Review Summary Pairs")
    
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    y_train, y_valid, y_test,  uid_train, iid_train, uid_valid, iid_valid, uid_test, iid_test, \
    review_train, summary_train, \
    review_valid, summary_valid, review_test, summary_test = load_data(FLAGS.train_data, FLAGS.valid_data, FLAGS.test_data,
                                                 FLAGS.review_summary)
    
    np.random.seed(2017)
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    userid_train = uid_train[shuffle_indices]
    itemid_train = iid_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    review_train = review_train[shuffle_indices]
    summary_train = summary_train[shuffle_indices]
    
    y_train = y_train[:, np.newaxis]
    y_valid = y_valid[:, np.newaxis]
    y_test = y_test[:, np.newaxis]

    userid_train = userid_train[:, np.newaxis]
    itemid_train = itemid_train[:, np.newaxis]
    userid_valid = uid_valid[:, np.newaxis]
    itemid_valid = iid_valid[:, np.newaxis]
    userid_test = uid_test[:, np.newaxis]
    itemid_test = iid_test[:, np.newaxis]

    review_train = review_train[:, np.newaxis]
    summary_train = summary_train[:, np.newaxis]
    review_valid = review_valid[:, np.newaxis]
    summary_valid = summary_valid[:, np.newaxis]
    review_test = review_test[:, np.newaxis]
    summary_test = summary_test[:, np.newaxis]

    batches_summ_train = list(zip(userid_train, itemid_train, review_train, summary_train))
    batches_summ_valid = list(zip(userid_valid, itemid_valid, review_valid, summary_valid))
    batches_summ_test = list(zip(userid_test, itemid_test, review_test, summary_test))
    
    output = open(os.path.join(TPS_DIR, '%s_summ.train' % (args.data_set)), 'wb')
    pickle.dump(batches_summ_train, output)
    output = open(os.path.join(TPS_DIR, '%s_summ.valid' % (args.data_set)), 'wb')
    pickle.dump(batches_summ_valid, output)
    output = open(os.path.join(TPS_DIR, '%s_summ.test' % (args.data_set)), 'wb')
    pickle.dump(batches_summ_test, output)
    