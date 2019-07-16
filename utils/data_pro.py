'''
Data pre process part2
@author:
Chong Chen (cstchenc@163.com)

@ created:
27/8/2017
'''

import numpy as np
import re
import itertools
from collections import Counter

import tensorflow as tf
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


def pad_sentences(u_text, u_len, u2_len, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    review_num = u_len
    review_len = u2_len

    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        padded_u_train = []
        for ri in range(review_num):
            if ri < len(u_reviews):
                sentence = u_reviews[ri]
                if review_len > len(sentence):
                    num_padding = review_len - len(sentence)
                    new_sentence = sentence + [padding_word] * num_padding
                    padded_u_train.append(new_sentence)
                else:
                    new_sentence = sentence[:review_len]
                    padded_u_train.append(new_sentence)
            else:
                new_sentence = [padding_word] * review_len
                padded_u_train.append(new_sentence)
        u_text2[i] = padded_u_train

    return u_text2

def pad_sentences_old(u_text, u_len, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length=u_len
    u_text2={}
    print len(u_text)
    for i in u_text.keys():
        #print i
        sentence = u_text[i]
        if sequence_length>len(sentence):

            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            u_text2[i]=new_sentence
        else:
            new_sentence = sentence[:sequence_length]
            u_text2[i] = new_sentence

    return u_text2


def pad_reviewid(u_train, u_valid, u_test, u_len, num):
    pad_u_train = []

    for i in range(len(u_train)):
        x = u_train[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_train.append(x)
    
    pad_u_valid = []
    for i in range(len(u_valid)):
        x = u_valid[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_valid.append(x)
        
    pad_u_test = []
    for i in range(len(u_test)):
        x = u_test[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_test.append(x)
    return pad_u_train, pad_u_valid, pad_u_test


def build_vocab(sentences1, sentences2):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts1 = Counter(itertools.chain(*sentences1))
    # Mapping from index to word
    vocabulary_inv1 = [x[0] for x in word_counts1.most_common()]
    vocabulary_inv1 = list(sorted(vocabulary_inv1))
    # Mapping from word to index
    vocabulary1 = {x: i for i, x in enumerate(vocabulary_inv1)}

    word_counts2 = Counter(itertools.chain(*sentences2))
    # Mapping from index to word
    vocabulary_inv2 = [x[0] for x in word_counts2.most_common()]
    vocabulary_inv2 = list(sorted(vocabulary_inv2))
    # Mapping from word to index
    vocabulary2 = {x: i for i, x in enumerate(vocabulary_inv2)}
    return [vocabulary1, vocabulary_inv1, vocabulary2, vocabulary_inv2]


def build_input_data(u_text, i_text, vocabulary_u, vocabulary_i):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    l = len(u_text)
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        u = np.array([[vocabulary_u[word] for word in words] for words in u_reviews])
        u_text2[i] = u
    l = len(i_text)
    i_text2 = {}
    for j in i_text.keys():
        i_reviews = i_text[j]
        i = np.array([[vocabulary_i[word] for word in words] for words in i_reviews])
        i_text2[j] = i
    return u_text2, i_text2


def build_input_data_old(u_text,i_text, vocabulary_u,vocabulary_i):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    l = len(u_text)
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        u = np.array([vocabulary_u[word] for word  in u_reviews])
        u_text2[i] = u
    l = len(i_text)
    i_text2 = {}
    for j in i_text.keys():
        i_reviews = i_text[j]
        i = np.array([vocabulary_i[word] for word in i_reviews])
        i_text2[j] = i
    return u_text2, i_text2

def load_data(train_data, valid_data, test_data, user_review, item_review, user_rid, item_rid, stopwords):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    u_text, i_text, u_text_old, i_text_old, y_train, y_valid, y_test, u_len, i_len, u2_len, i2_len, uid_train, iid_train, \
    uid_valid, iid_valid, uid_test, iid_test, user_num, item_num , reid_user_train, reid_item_train, reid_user_valid, \
    reid_item_valid, reid_user_test, reid_item_test= \
        load_data_and_labels(train_data, valid_data, test_data, user_review, item_review, user_rid, item_rid, stopwords)
    print("Finished loading data")
    
    u_text = pad_sentences(u_text, u_len, u2_len)
    u_text_old = pad_sentences_old(u_text_old, u_len)
    reid_user_train, reid_user_valid, reid_user_test = pad_reviewid(reid_user_train, reid_user_valid, reid_user_test, u_len, item_num + 1)
    print("Finished pad user")
    
    i_text = pad_sentences(i_text, i_len, i2_len)
    i_text_old = pad_sentences_old(i_text_old, i_len)
    
    reid_item_train, reid_item_valid, reid_item_test = pad_reviewid(reid_item_train, reid_item_valid, reid_item_test, i_len, user_num + 1)
    print("Finished pad item")

    user_voc = [xx for x in u_text.itervalues() for xx in x]
    item_voc = [xx for x in i_text.itervalues() for xx in x]

    vocabulary_user, vocabulary_inv_user, vocabulary_item, vocabulary_inv_item = build_vocab(user_voc, item_voc)
    print("Finished creating vocabulary")  
    
    print("Length of Vocabulary: User %s, Item %s" % (len(vocabulary_user),len(vocabulary_item)))
    
    u_text, i_text = build_input_data(u_text, i_text, vocabulary_user, vocabulary_item)
    u_text_old, i_text_old = build_input_data_old(u_text_old, i_text_old, vocabulary_user, vocabulary_item)
    print("Finished building Input data")      
    
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)
    uid_train = np.array(uid_train)
    uid_valid = np.array(uid_valid)
    uid_test = np.array(uid_test)
    iid_train = np.array(iid_train)
    iid_valid = np.array(iid_valid)
    iid_test = np.array(iid_test)
    reid_user_train = np.array(reid_user_train)
    reid_user_valid = np.array(reid_user_valid)
    reid_user_test = np.array(reid_user_test)
    reid_item_train = np.array(reid_item_train)
    reid_item_valid = np.array(reid_item_valid)
    reid_item_test = np.array(reid_item_test)

    return [u_text, i_text, u_text_old, i_text_old, y_train, y_valid, y_test, vocabulary_user, vocabulary_item, uid_train, iid_train,
            uid_valid, iid_valid, uid_test, iid_test, user_num, item_num, reid_user_train,
            reid_item_train, reid_user_valid, reid_item_valid, reid_user_test, reid_item_test]


def load_data_and_labels(train_data, valid_data, test_data, user_review, item_review, user_rid, item_rid, stopwords):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files


    f_train = open(train_data, "r")
    f1 = open(user_review)
    f2 = open(item_review)
    f3 = open(user_rid)
    f4 = open(item_rid)

    user_reviews = pickle.load(f1)
    item_reviews = pickle.load(f2)
    user_rids = pickle.load(f3)
    item_rids = pickle.load(f4)

    reid_user_train = []
    reid_item_train = []
    uid_train = []
    iid_train = []
    y_train = []
    u_text = {}
    u_rid = {}
    i_text = {}
    i_rid = {}
    i = 0
    u_text_old = {}
    i_text_old = {}
    for line in f_train:
        i = i + 1
        line = line.split(',')
        uid_train.append(int(line[0]))
        iid_train.append(int(line[1]))
        if u_text.has_key(int(line[0])):
            reid_user_train.append(u_rid[int(line[0])])
        else:
            # For NARRE
            u_text[int(line[0])] = []
            for s in user_reviews[int(line[0])]:
                s1 = clean_str(s)
                s1 = s1.split(" ")
                u_text[int(line[0])].append(s1)
            u_rid[int(line[0])] = []
            for s in user_rids[int(line[0])]:
                u_rid[int(line[0])].append(int(s))
            reid_user_train.append(u_rid[int(line[0])])
            
            # For DeepCoNN
            u_text_old[int(line[0])] = '<PAD/>'
            for s in user_reviews[int(line[0])]:
                u_text_old[int(line[0])] = u_text_old[int(line[0])] + ' ' + s.strip()
            u_text_old[int(line[0])]=clean_str(u_text_old[int(line[0])])
            u_text_old[int(line[0])]=u_text_old[int(line[0])].split(" ")
            
            
        if i_text.has_key(int(line[1])):
            reid_item_train.append(i_rid[int(line[1])])  #####write here
        else:
            # For NARRE
            i_text[int(line[1])] = []
            for s in item_reviews[int(line[1])]:
                s1 = clean_str(s)
                s1 = s1.split(" ")

                i_text[int(line[1])].append(s1)
            i_rid[int(line[1])] = []
            for s in item_rids[int(line[1])]:
                i_rid[int(line[1])].append(int(s))
            reid_item_train.append(i_rid[int(line[1])])
            
            # For DeepCoNN
            i_text_old[int(line[1])] = '<PAD/>'
            for s in item_reviews[int(line[1])]:
                i_text_old[int(line[1])] = i_text_old[int(line[1])] + ' ' + s.strip()
            i_text_old[int(line[1])]=clean_str(i_text_old[int(line[1])])
            i_text_old[int(line[1])]=i_text_old[int(line[1])].split(" ")
            
            
        y_train.append(float(line[2]))
    
    print("Process validation data")
    reid_user_valid = []
    reid_item_valid = []

    uid_valid = []
    iid_valid = []
    y_valid = []
    f_valid = open(valid_data)
    for line in f_valid:
        line = line.split(',')
        uid_valid.append(int(line[0]))
        iid_valid.append(int(line[1]))
        if u_text.has_key(int(line[0])):
            reid_user_valid.append(u_rid[int(line[0])])
        else:
            u_text[int(line[0])] = [['<PAD/>']]
            u_rid[int(line[0])] = [int(0)]
            reid_user_valid.append(u_rid[int(line[0])])
            
            # For DeepCoNN
            u_text_old[int(line[0])] = '<PAD/>'
            u_text_old[int(line[0])]=clean_str(u_text_old[int(line[0])])
            u_text_old[int(line[0])]=u_text_old[int(line[0])].split(" ")
            
        if i_text.has_key(int(line[1])):
            reid_item_valid.append(i_rid[int(line[1])])
        else:
            i_text[int(line[1])] = [['<PAD/>']]
            i_rid[int(line[1])] = [int(0)]
            reid_item_valid.append(i_rid[int(line[1])])
            
            # For DeepCoNN
            i_text_old[int(line[1])] = '<PAD/>'
            i_text_old[int(line[1])]=clean_str(i_text_old[int(line[1])])
            i_text_old[int(line[1])]=i_text_old[int(line[1])].split(" ")

        y_valid.append(float(line[2]))
    
    print("Process test data")
    reid_user_test = []
    reid_item_test = []

    uid_test = []
    iid_test = []
    y_test = []
    f_test = open(test_data)
    for line in f_test:
        line = line.split(',')
        uid_test.append(int(line[0]))
        iid_test.append(int(line[1]))
        if u_text.has_key(int(line[0])):
            reid_user_test.append(u_rid[int(line[0])])
        else:
            u_text[int(line[0])] = [['<PAD/>']]
            u_rid[int(line[0])] = [int(0)]
            reid_user_test.append(u_rid[int(line[0])])
            
            # For DeepCoNN
            u_text_old[int(line[0])] = '<PAD/>'
            u_text_old[int(line[0])]=clean_str(u_text_old[int(line[0])])
            u_text_old[int(line[0])]=u_text_old[int(line[0])].split(" ")

        if i_text.has_key(int(line[1])):
            reid_item_test.append(i_rid[int(line[1])])
        else:
            i_text[int(line[1])] = [['<PAD/>']]
            i_rid[int(line[1])] = [int(0)]
            reid_item_test.append(i_rid[int(line[1])])
            
            # For DeepCoNN
            i_text_old[int(line[1])] = '<PAD/>'
            i_text_old[int(line[1])]=clean_str(i_text_old[int(line[1])])
            i_text_old[int(line[1])]=i_text_old[int(line[1])].split(" ")

        y_test.append(float(line[2]))
    
    review_num_u = np.array([len(x) for x in u_text.itervalues()])
    x = np.sort(review_num_u)
    u_len = x[int(0.9 * len(review_num_u)) - 1]
    review_len_u = np.array([len(j) for i in u_text.itervalues() for j in i])
    x2 = np.sort(review_len_u)
    u2_len = x2[int(0.9 * len(review_len_u)) - 1]

    review_num_i = np.array([len(x) for x in i_text.itervalues()])
    y = np.sort(review_num_i)
    i_len = y[int(0.9 * len(review_num_i)) - 1]
    review_len_i = np.array([len(j) for i in i_text.itervalues() for j in i])
    y2 = np.sort(review_len_i)
    i2_len = y2[int(0.9 * len(review_len_i)) - 1]
    print "u_len:", u_len
    print "i_len:", i_len
    print "u2_len:", u2_len
    print "i2_len:", i2_len
    user_num = len(u_text)
    item_num = len(i_text)
    print "user_num:", user_num
    print "item_num:", item_num
    return [u_text, i_text, u_text_old, i_text_old, y_train, y_valid, y_test, u_len, i_len, u2_len, i2_len, uid_train,
            iid_train, uid_valid, iid_valid, uid_test, iid_test, user_num,
            item_num, reid_user_train, reid_item_train, reid_user_valid, reid_item_valid, reid_user_test, reid_item_test]



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
    tf.flags.DEFINE_string("user_review", "../data/%s/user_review" % (args.data_set), "User's reviews")
    tf.flags.DEFINE_string("item_review", "../data/%s/item_review" % (args.data_set), "Item's reviews")
    tf.flags.DEFINE_string("user_review_id", "../data/%s/user_rid" % (args.data_set), "user_review_id")
    tf.flags.DEFINE_string("item_review_id", "../data/%s/item_rid" % (args.data_set), "item_review_id")
    tf.flags.DEFINE_string("stopwords", "../data/stopwords", "stopwords")
    
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    u_text, i_text, u_text_old, i_text_old, y_train, y_valid, y_test, vocabulary_user, vocabulary_item, \
    uid_train, iid_train, uid_valid, iid_valid, uid_test, iid_test, \
    user_num, item_num, reid_user_train, reid_item_train, \
    reid_user_valid, reid_item_valid, reid_user_test, reid_item_test = load_data(FLAGS.train_data, FLAGS.valid_data, FLAGS.test_data,
                                                 FLAGS.user_review, FLAGS.item_review, 
                                                 FLAGS.user_review_id, FLAGS.item_review_id, FLAGS.stopwords)
    
    np.random.seed(2017)

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    userid_train = uid_train[shuffle_indices]
    itemid_train = iid_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    y_train = y_train[:, np.newaxis]
    y_valid = y_valid[:, np.newaxis]
    y_test = y_test[:, np.newaxis]

    userid_train = userid_train[:, np.newaxis]
    itemid_train = itemid_train[:, np.newaxis]
    userid_valid = uid_valid[:, np.newaxis]
    itemid_valid = iid_valid[:, np.newaxis]
    userid_test = uid_test[:, np.newaxis]
    itemid_test = iid_test[:, np.newaxis]

    print("REID train", reid_user_train[0].shape)
    print("REID valid", reid_user_valid[0].shape)
    print("REID test", reid_user_test[0].shape)
    
    print("REID Item train", reid_item_train[0].shape)
    print("REID Item valid", reid_item_valid[0].shape)
    print("REID Item test", reid_item_test[0].shape)
    
    
    batches_train = list(zip(userid_train, itemid_train, reid_user_train, reid_item_train, y_train))
    batches_valid = list(zip(userid_valid, itemid_valid, reid_user_valid, reid_item_valid, y_valid))
    batches_test = list(zip(userid_test, itemid_test, reid_user_test, reid_item_test, y_test))
    
    
    output = open(os.path.join(TPS_DIR, '%s.train' % (args.data_set)), 'wb')
    pickle.dump(batches_train, output)
    output = open(os.path.join(TPS_DIR, '%s.valid' % (args.data_set)), 'wb')
    pickle.dump(batches_valid, output)
    output = open(os.path.join(TPS_DIR, '%s.test' % (args.data_set)), 'wb')
    pickle.dump(batches_test, output)


    para={}
    para['user_num']=user_num
    para['item_num']=item_num
    para['review_num_u'] = u_text[0].shape[0]
    para['review_num_i'] = i_text[0].shape[0]
    para['review_len_u'] = u_text[1].shape[1]
    para['review_len_i'] = i_text[1].shape[1]
    para['user_length']=u_text[0].shape[0]
    para['item_length'] = i_text[0].shape[0]
    para['user_length_old']=u_text_old[0].shape[0]
    para['item_length_old'] = i_text_old[0].shape[0]
    para['user_vocab'] = vocabulary_user
    para['item_vocab'] = vocabulary_item
    para['train_length']=len(y_train)
    para['test_length']=len(y_test)
    para['valid_length']=len(y_valid)
    para['u_text'] = u_text
    para['i_text'] = i_text
    para['u_text_old'] = u_text_old
    para['i_text_old'] = i_text_old
    
    output = open(os.path.join(TPS_DIR, '%s.para' % (args.data_set)), 'wb')

    pickle.dump(para, output)


