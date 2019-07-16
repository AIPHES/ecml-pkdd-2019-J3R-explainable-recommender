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
import json

def mkdirp(path):
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        pass

def write_to_onmt_files(data_path, data_file, review_summaries, output_type="nnsum"):
    mkdirp(data_path)
    fp = codecs.open(data_file, "r", encoding="utf-8")
    
    src_path = "%s.src" % (data_path)
    tgt_path = "%s.tgt" % (data_path)
        
    with codecs.open(src_path, 'w', encoding="utf-8") as fp1, codecs.open(tgt_path , 'w', encoding="utf-8") as fp2:
        for i, line in enumerate(fp):
            line = line.split(',')
            user_id = int(line[0])
            item_id = int(line[1])

            review, summary = review_summaries[(user_id, item_id)]
            review_tokens = word_tokenize(review)
            summary_tokens = word_tokenize(summary)

            if len(review_tokens) < 10 or len(summary_tokens) < 4:
                continue

            fp1.write(" ".join(review_tokens).lower() + "\n")
            fp2.write(" ".join(summary_tokens).lower() + "\n")

    
def write_to_nnsum_files(data_path, data_file, review_summaries):
    
    mkdirp(data_path)
    fp = codecs.open(data_file, "r")
    
    for i, line in enumerate(fp):
        line = line.split(',')
        user_id = int(line[0])
        item_id = int(line[1])
        
        review, summary = review_summaries[(user_id, item_id)]
        review_tokens = word_tokenize(review)
        summary_tokens = word_tokenize(summary)
        
        if len(review_tokens) < 10 or len(summary_tokens) < 4:
            continue
        
       
        file_path = "%s/%s_%s.json" % (data_path, line[0], line[1])
        data = {}
        data['user_id'] = user_id
        data['item_id'] = item_id
        data['review'] = " ".join(review_tokens).lower()
        data['summary'] = " ".join(summary_tokens).lower()
        with open(file_path, 'w') as outfile:
            json.dump(data, outfile)
        

def load_data(data_path, train_data, valid_data, test_data, review_summary, output_type):

    f1 = open(review_summary)
    review_summaries = pickle.load(f1)
    
    if output_type == "nnsum":
        print("Processing NNSUM training data")    
        write_to_nnsum_files(data_path + '/train', train_data, review_summaries)
        print("Processing NNSUM validation data")
        write_to_nnsum_files(data_path + '/valid', valid_data, review_summaries)
        print("Processing NNSUM test data")
        write_to_nnsum_files(data_path + '/test', valid_data, review_summaries)
    if output_type == "onmt":
        print("Processing ONMT training data")    
        write_to_onmt_files(data_path + '/train', train_data, review_summaries)
        print("Processing ONMT validation data")
        write_to_onmt_files(data_path + '/valid', valid_data, review_summaries)
        print("Processing ONMT test data")
        write_to_onmt_files(data_path + '/test', valid_data, review_summaries)
        
        
def get_args():
    ''' This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(description='Process Amazon and Yelp data for NNSUM')
    parser.add_argument('-d', '--data_set', type=str, help='Dataset music, tv, electronics, kindle, toys, cds, yelp', required=True)
    parser.add_argument('-o', '--output_type', type=str, help='Dataset music, tv, electronics, kindle, toys, cds, yelp', required=False, default="nnsum")
    args = parser.parse_args()
    return args
                        
if __name__ == '__main__':
    
    args = get_args()
    TPS_DIR = '../data/%s' %(args.data_set)
    
    tf.flags.DEFINE_string("valid_data","../data/%s/%s_valid.csv" % (args.data_set, args.data_set), "Data for validation")
    tf.flags.DEFINE_string("test_data", "../data/%s/%s_test.csv" % (args.data_set, args.data_set), "Data for testing")
    tf.flags.DEFINE_string("train_data", "../data/%s/%s_train.csv" % (args.data_set, args.data_set), "Data for training")
    tf.flags.DEFINE_string("review_summary", "../data/%s/review_summary" % (args.data_set), "Review Summary Pairs")
    tf.flags.DEFINE_string("output_type", args.output_type, "Output Type nnsum or onmt")
    
    if args.output_type == "nnsum":
        tf.flags.DEFINE_string("data_path","../../../summarize/nnsum/data/raw/%s/" % (args.data_set), "Data path for NNSUM")
    if args.output_type == "onmt":
        tf.flags.DEFINE_string("data_path","../../../summarize/OpenNMT-py/data/%s/" % (args.data_set), "Data path for NNSUM")
    
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    load_data(FLAGS.data_path, FLAGS.train_data, FLAGS.valid_data, FLAGS.test_data, FLAGS.review_summary, FLAGS.output_type)
    
   