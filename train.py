import argparse
import numpy as np
import tensorflow as tf
import math
from tensorflow.contrib import learn
import datetime
import os

import pickle
from model import DeepCoNN, DeepCoNNPlusPlus, NARRE, MLPTopic, MLP, J3R
import logging 


# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding ")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes ")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability ")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda")
#tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularizaion lambda")

tf.flags.DEFINE_float("l2_reg_V", 0, "L2 regularizaion V")
# Training parameters

tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps ")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps ")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

def mkdirp(path):
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        pass


def write_to_file(filename, y_true, y_pred):
    with open(filename, 'w') as fp:
        for y1, y2 in zip(y_true, y_pred):
            fp.write("%s,%s\n" % (str(y1[0]), str(y2)))
    
            
def train_step(u_batch, i_batch, uid, iid, y_batch, batch_num):
    """
    A single training step
    """
    feed_dict = {
            deep.input_u: u_batch,
            deep.input_i: i_batch,
            deep.input_y: y_batch,
            deep.input_uid: uid,
            deep.input_iid: iid,
            deep.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
        
    _, step, loss, accuracy, mae = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()

    # print("{}: step {}, loss {:g}, rmse {:g},mae {:g}".format(time_str, batch_num, loss, accuracy, mae))
    return accuracy, mae

def train_step_NARRE(u_batch, i_batch, uid, iid, reuid, reiid, y_batch, batch_num):
    """
    A single training step
    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_y: y_batch,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 0.8,

        deep.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, loss, accuracy, mae, u_a, i_a, fm = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae, deep.u_a, deep.i_a, deep.score],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    #print("{}: step {}, loss {:g}, rmse {:g},mae {:g}".format(time_str, batch_num, loss, accuracy, mae))
    return accuracy, mae, u_a, i_a, fm


def dev_step(u_batch, i_batch, uid, iid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step{}, loss {:g}, rmse {:g},mae {:g}".format(time_str, step, loss, accuracy, mae))

    return [loss, accuracy, mae]


def dev_step_NARRE(u_batch, i_batch, uid, iid, reuid, reiid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 1.0,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step{}, loss {:g}, rmse {:g},mae {:g}".format(time_str, step, loss, accuracy, mae))

    return [loss, accuracy, mae]


def test_step(u_batch, i_batch, uid, iid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae, y_pred = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae, deep.predictions],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step{}, loss {:g}, rmse {:g},mae {:g}".format(time_str, step, loss, accuracy, mae))

    return [loss, accuracy, mae, y_pred]


def test_step_NARRE(u_batch, i_batch, uid, iid, reuid, reiid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 1.0,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae, y_pred = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae, deep.predictions],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step{}, loss {:g}, rmse {:g},mae {:g}".format(time_str, step, loss, accuracy, mae))

    return [loss, accuracy, mae, y_pred]


def read_word2vec(initW_user, initW_item, word2vec, embed_type):
    u = 0
    item = 0

    if embed_type == "google":
        with open(word2vec, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                idx = 0
                emb_string = f.read(binary_len)
                if word in vocabulary_user:
                    u = u + 1
                    idx = vocabulary_user[word]
                    initW_user[idx] = np.fromstring(emb_string, dtype='float32')

                if word in vocabulary_item:
                    item = item + 1
                    idx = vocabulary_item[word]
                    initW_item[idx] = np.fromstring(emb_string, dtype='float32')
                    
    if embed_type == "glove":
        with open(word2vec, "rb") as f:
            header = f.readline()
            for i, line in enumerate(f):
                if i == 0:
                    vocab_size, layer1_size = map(int, line.split())
                else:
                    tokens = line.split()
                    word, emb_string = tokens[0], ' '.join(tokens[1:])
                    if word in vocabulary_user:
                        u = u + 1
                        idx = vocabulary_user[word]
                        initW_user[idx] = np.fromstring(emb_string, dtype='float32')

                    if word in vocabulary_item:
                        item = item + 1
                        idx = vocabulary_item[word]
                        initW_item[idx] = np.fromstring(emb_string, dtype='float32')
    
    return initW_user, initW_item
    

def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Parameters for Training Models')
 
    parser.add_argument('-d', '--data_set', type=str, help='Dataset music, tv, electronics, cds, toys, kindle, yelp', required=True)
    parser.add_argument('-m', '--model_type', type=str, help='Model Type DeepCoNN, DeeepCoNN++, NARRE', required=True)
    parser.add_argument('-b', '--batch_size', type=int, help='Batch Size (default=32)', default=32, required=False)
    parser.add_argument('-e', '--epochs', type=int, help='Number of Epochs (default=20)', default=20, required=False)
    parser.add_argument('-f', '--factors', type=int, help='Number of Factors (default=32)', default=32, required=False)
    parser.add_argument('--embed_type', type=str, help='Embedding type ex: google, glove', default="google", required=False)
    args = parser.parse_args()

    return args
        

if __name__ == '__main__':
    
    args = get_args()
    
    TPS_DIR = '../data/%s' %(args.data_set) 

    tf.flags.DEFINE_string("para_data", "../data/%s/%s.para" % (args.data_set, args.data_set), "Data parameters")
    tf.flags.DEFINE_string("train_data", "../data/%s/%s.train" % (args.data_set, args.data_set), "Data for training")
    tf.flags.DEFINE_string("valid_data","../data/%s/%s.valid" % (args.data_set, args.data_set), " Data for validation")
    tf.flags.DEFINE_string("test_data", "../data/%s/%s.test" % (args.data_set, args.data_set), "Data for testing")
    tf.flags.DEFINE_string("model_path", "../saved_models/%s/%s" % (args.data_set, args.model_type), "Model path")
    tf.flags.DEFINE_string("output_path", "../outputs/%s/" % (args.data_set), "Output Path")
    tf.flags.DEFINE_string("logs", "../logs/%s_%s.log" % (args.data_set, args.model_type), "Logging path")
    tf.flags.DEFINE_string("model_type", args.model_type, "model_type")
    tf.flags.DEFINE_integer("batch_size", args.batch_size, "Batch Size ")
    tf.flags.DEFINE_integer("num_epochs", args.epochs, "Number of training epochs ")
    tf.flags.DEFINE_integer("num_factors", args.factors, "Number of Factors ")
    if args.embed_type == "google":
        tf.flags.DEFINE_string("word2vec", "../data/embeddings/google.bin", "Word2vec file with pre-trained embeddings (default: None)")
    if args.embed_type == "glove":
        tf.flags.DEFINE_string("word2vec", "../data/embeddings/glove.bin", "Word2vec file with pre-trained embeddings (default: None)")
    
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    
    
    mkdirp(FLAGS.model_path)
    mkdirp(FLAGS.output_path)
    
    logging.basicConfig(filename=FLAGS.logs, filemode='w', level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    
    logging.info("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        logging.info("{}={}".format(attr.upper(), value))
    logging.info("")

    logging.info("Loading data...")

    pkl_file = open(FLAGS.para_data, 'rb')

    para = pickle.load(pkl_file)
    user_num = para['user_num']
    item_num = para['item_num']
    
    vocabulary_user = para['user_vocab']
    vocabulary_item = para['item_vocab']
    train_length = para['train_length']
    valid_length = para['valid_length']
    test_length = para['test_length']
    
    best_rmse_valid = 10000
    
    #For NAARE      
    if args.model_type == "NARRE":
        review_num_u = para['review_num_u']
        review_num_i = para['review_num_i']
        review_len_u = para['review_len_u']
        review_len_i = para['review_len_i']
        u_text = para['u_text']
        i_text = para['i_text']
        user_length = para['user_length']
        item_length = para['item_length']
    else:
        u_text = para['u_text_old']
        i_text = para['i_text_old']
        user_length = para['user_length_old']
        item_length = para['item_length_old']
        

    valid_rmse_scores, valid_mae_scores = [], []  
    test_rmse_scores, test_mae_scores = [], []
    
    np.random.seed(2017)
    random_seed = 2017
	
    with tf.device('/gpu:0'):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=False)
            session_conf.gpu_options.allow_growth = False
            
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                if args.model_type== "DeepCoNN":
                    deep = DeepCoNN.DeepCoNN(
                        user_num=user_num,
                        item_num=item_num,
                        user_length=user_length,
                        item_length=item_length,
                        num_classes=1,
                        user_vocab_size=len(vocabulary_user),
                        item_vocab_size=len(vocabulary_item),
                        embedding_size=FLAGS.embedding_dim,
                        fm_k=8,
                        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                        num_filters=FLAGS.num_filters,
                        l2_reg_lambda=FLAGS.l2_reg_lambda,
                        l2_reg_V=FLAGS.l2_reg_V,
                        n_latent=32)
                
                if args.model_type == "DeepCoNNPlusPlus":
                    deep = DeepCoNNPlusPlus.DeepCoNNPlusPlus(
                        user_num=user_num,
                        item_num=item_num,
                        user_length=user_length,
                        item_length=item_length,
                        num_classes=1,
                        user_vocab_size=len(vocabulary_user),
                        item_vocab_size=len(vocabulary_item),
                        embedding_size=FLAGS.embedding_dim,
                        fm_k=8,
                        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                        num_filters=FLAGS.num_filters,
                        l2_reg_lambda=FLAGS.l2_reg_lambda,
                        l2_reg_V=FLAGS.l2_reg_V,
                        n_latent=FLAGS.num_factors)
                    
                if args.model_type == "NARRE":
                    deep = NARRE.NARRE(
                        review_num_u=review_num_u,
                        review_num_i=review_num_i,
                        review_len_u=review_len_u,
                        review_len_i=review_len_i,
                        user_num=user_num,
                        item_num=item_num,
                        num_classes=1,
                        user_vocab_size=len(vocabulary_user),
                        item_vocab_size=len(vocabulary_item),
                        embedding_size=FLAGS.embedding_dim,
                        embedding_id=32,
                        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                        num_filters=FLAGS.num_filters,
                        l2_reg_lambda=FLAGS.l2_reg_lambda,
                        attention_size=32,
                        n_latent=32)
                
                tf.set_random_seed(random_seed)
                global_step = tf.Variable(0, name="global_step", trainable=False)

                # optimizer = tf.train.AdagradOptimizer(learning_rate=0.1, initial_accumulator_value=1e-8).minimize(deep.loss)

                optimizer = tf.train.AdamOptimizer(0.002, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(deep.loss)
                '''optimizer=tf.train.RMSPropOptimizer(0.002)
                grads_and_vars = optimizer.compute_gradients(deep.loss)'''
                train_op = optimizer  # .apply_gradients(grads_and_vars, global_step=global_step)

                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                
                if FLAGS.word2vec:
                    # initial matrix with random uniform

                    initW_user = np.random.uniform(-1.0, 1.0, (len(vocabulary_user), FLAGS.embedding_dim))
                    initW_item = np.random.uniform(-1.0, 1.0, (len(vocabulary_item), FLAGS.embedding_dim))
                    # load any vectors from the word2vec
                    logging.info("Load word2vec file {} to initialize user and item".format(FLAGS.word2vec))
                    
                    initW_user, initW_item  = read_word2vec(initW_user, initW_item, FLAGS.word2vec, args.embed_type)
                    
                    sess.run(deep.W1.assign(initW_user))
                    sess.run(deep.W2.assign(initW_item))

                l = (train_length / FLAGS.batch_size) + 1
                ll = 0
                epoch = 1
                best_mae = 5
                best_rmse = 5
                train_mae = 0
                train_rmse = 0

                pkl_file = open(FLAGS.train_data, 'rb')
                
                logging.info("Load Train Data")
                train_data = pickle.load(pkl_file)
                train_data = np.array(train_data)
                pkl_file.close()
                
                logging.info("Load Validation Data")
                pkl_file = open(FLAGS.valid_data, 'rb')
                valid_data = pickle.load(pkl_file)
                valid_data = np.array(valid_data)
                pkl_file.close()
                                
                logging.info("Load Test Data")
                pkl_file = open(FLAGS.test_data, 'rb')
                test_data = pickle.load(pkl_file)
                test_data = np.array(test_data)
                pkl_file.close()
                
                data_size_train = len(train_data)
                data_size_valid = len(valid_data)
                data_size_test = len(test_data)
                batch_size = FLAGS.batch_size
                ll = int(len(train_data) / batch_size)

                for epoch in range(FLAGS.num_epochs):
                    logging.info("Epoch: %s" % (epoch))
                    # Shuffle the data at each epoch
                    shuffle_indices = np.random.permutation(np.arange(data_size_train))
                    shuffled_data = train_data[shuffle_indices]
                    for batch_num in range(ll):
                        start_index = batch_num * batch_size
                        end_index = min((batch_num + 1) * batch_size, data_size_train)
                        data_train = shuffled_data[start_index:end_index]

                        uid, iid, reuid, reiid, y_batch = zip(*data_train)
                        #print("REUID TRAIN", reuid[0].shape)
                        """
                        print(uid[0].shape)
                        print(iid[0].shape)
                        print(reuid[0].shape)
                        print(reiid[0].shape)
                        print(y_batch[0].shape)
                        """
                        u_batch = []
                        i_batch = []
                        for i in range(len(uid)):
                            u_batch.append(u_text[uid[i][0]])
                            i_batch.append(i_text[iid[i][0]])
                        u_batch = np.array(u_batch)
                        i_batch = np.array(i_batch)
                        
                        if FLAGS.model_type == "NARRE":
                            t_rmse, t_mae, u_a, i_a, fm = train_step_NARRE(u_batch, i_batch, uid, iid, reuid, reiid, y_batch, batch_num)
                        else:
                            t_rmse, t_mae = train_step(u_batch, i_batch, uid, iid, y_batch, batch_num)
                            
                        current_step = tf.train.global_step(sess, global_step)
                        train_rmse += t_rmse
                        train_mae += t_mae

                        if batch_num % 1000 == 0 and batch_num > 1:
                            logging.info("Evaluation Batch: %s" %(batch_num))
                            loss_s = 0
                            accuracy_s = 0
                            mae_s = 0

                            ll_valid = int(len(valid_data) / batch_size) + 1
                            for batch_num2 in range(ll_valid):
                                start_index = batch_num2 * batch_size
                                end_index = min((batch_num2 + 1) * batch_size, data_size_valid)
                                data_valid = valid_data[start_index:end_index]

                                userid_valid, itemid_valid, reuid, reiid, y_valid = zip(*data_valid)
                                
                                u_valid = []
                                i_valid = []
                                for i in range(len(userid_valid)):
                                    u_valid.append(u_text[userid_valid[i][0]])
                                    i_valid.append(i_text[itemid_valid[i][0]])
                                u_valid = np.array(u_valid)
                                i_valid = np.array(i_valid)

                                if FLAGS.model_type == "NARRE":
                                    loss, accuracy, mae = dev_step_NARRE(u_valid, i_valid, userid_valid, itemid_valid, reuid, reiid, y_valid)
                                else:
                                    loss, accuracy, mae = dev_step(u_valid, i_valid, userid_valid, itemid_valid, y_valid)
                                    
                                loss_s += len(u_valid) * loss
                                accuracy_s += len(u_valid) * np.square(accuracy)
                                mae_s += len(u_valid) * mae
                            logging.info("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}".format(loss_s / valid_length,
                                                                                             np.sqrt(
                                                                                                 accuracy_s / valid_length),
                                                                                             mae_s / valid_length))

                    logging.info("Evaluation:")
                    logging.info("Train: rmse {:g},mae: {:g}".format(train_rmse / ll, train_mae / ll))
                    
                    
                    if FLAGS.model_type == "NARRE":
                        u_a = np.reshape(u_a[0], (1, -1))
                        i_a = np.reshape(i_a[0], (1, -1))
                        print("User Attention", u_a)
                        print("Item Attention", i_a)
                    
                    train_rmse = 0
                    train_mae = 0

                    loss_s = 0
                    accuracy_s_valid = 0
                    mae_s_valid = 0

                    ll_valid = int(len(valid_data) / batch_size) + 1
                    for batch_num in range(ll_valid):
                        start_index = batch_num * batch_size
                        end_index = min((batch_num + 1) * batch_size, data_size_valid)
                        data_valid = valid_data[start_index:end_index]

                        userid_valid, itemid_valid, reuid, reiid, y_valid = zip(*data_valid)
                        #print("REUID_valid", reuid[0].shape)    
                        
                        u_valid = []
                        i_valid = []
                        for i in range(len(userid_valid)):
                            u_valid.append(u_text[userid_valid[i][0]])
                            i_valid.append(i_text[itemid_valid[i][0]])
                        u_valid = np.array(u_valid)
                        i_valid = np.array(i_valid)

                        if FLAGS.model_type == "NARRE":
                            loss, accuracy, mae = dev_step_NARRE(u_valid, i_valid, userid_valid, itemid_valid, reuid, reiid, y_valid)
                        else:
                            loss, accuracy, mae = dev_step(u_valid, i_valid, userid_valid, itemid_valid, y_valid)
                        
                        loss_s += len(u_valid) * loss
                        accuracy_s_valid += len(u_valid) * np.square(accuracy)
                        mae_s_valid += len(u_valid) * mae
                    logging.info("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}".format(loss_s / valid_length,
                                                                                     np.sqrt(accuracy_s_valid / valid_length),
                                                                                     mae_s_valid / valid_length))
                    
                    
                    rmse_valid = np.sqrt(accuracy_s_valid / valid_length)
                    if rmse_valid < best_rmse_valid:
                        save_path = saver.save(sess, "%s/model_%s.ckpt" % (FLAGS.model_path, str(epoch)))
                        logging.info("Model saved in path: %s" % save_path)
                        best_rmse_valid = rmse_valid
                    
                    loss_s = 0
                    accuracy_s_test = 0
                    mae_s_test = 0
                    
                    ll_test = int(len(test_data) / batch_size) + 1
                    for batch_num in range(ll_test):
                        start_index = batch_num * batch_size
                        end_index = min((batch_num + 1) * batch_size, data_size_test)
                        data_test = test_data[start_index:end_index]
                        
                        userid_test, itemid_test, reuid, reiid, y_test = zip(*data_test)
                        
                        #print("REUID test", reuid[0].shape)
                        u_test = []
                        i_test = []
                        for i in range(len(userid_test)):
                            u_test.append(u_text[userid_test[i][0]])
                            i_test.append(i_text[itemid_test[i][0]])
                        u_test = np.array(u_test)
                        i_test = np.array(i_test)
                        
                        
                        if FLAGS.model_type == "NARRE":
                            loss, accuracy, mae = dev_step_NARRE(u_test, i_test, userid_test, itemid_test, reuid, reiid, y_test)
                        else:
                            loss, accuracy, mae = dev_step(u_test, i_test, userid_test, itemid_test, y_test)
                        loss_s += len(u_test) * loss
                        accuracy_s_test +=  len(u_test) * np.square(accuracy)
                        mae_s_test += len(u_test) * mae
                    logging.info("loss_test {:g}, rmse_test {:g}, mae_test {:g}".format(loss_s / test_length,
                                                                                     np.sqrt(accuracy_s_test / test_length),
                                                                                     mae_s_test / test_length))
                    
                    valid_rmse = np.sqrt(accuracy_s_valid / valid_length)
                    valid_mae = mae_s_valid / valid_length
                    
                    test_rmse = np.sqrt(accuracy_s_test / test_length)
                    test_mae = mae_s_test / test_length
                
                    valid_rmse_scores.append(valid_rmse)
                    valid_mae_scores.append(valid_mae)
                    
                    test_rmse_scores.append(test_rmse)
                    test_mae_scores.append(test_mae)
                    
                # Find the best model on validation data
                index = valid_rmse_scores.index(min(valid_rmse_scores))
                
                logging.info("Best Epoch {}".format(index))
                logging.info("Best valid: RMSE {:g}, MAE {:g}".format(valid_rmse_scores[index], valid_mae_scores[index])) 
                logging.info("Best test: RMSE {:g}, MAE {:g}".format(test_rmse_scores[index], test_mae_scores[index])) 
                
                # Load the best model
                saver.restore(sess, "%s/model_%s.ckpt" % (FLAGS.model_path, str(index)))
                
                y_true = []
                y_pred = []
                loss_s = 0
                accuracy_s_test = 0
                mae_s_test = 0
                for batch_num in range(ll_test):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_test)
                    data_test = test_data[start_index:end_index]

                    userid_test, itemid_test, reuid, reiid, y_test = zip(*data_test)
                    u_test = []
                    i_test = []
                    for i in range(len(userid_test)):
                        u_test.append(u_text[userid_test[i][0]])
                        i_test.append(i_text[itemid_test[i][0]])
                    u_test = np.array(u_test)
                    i_test = np.array(i_test)

                    if FLAGS.model_type == "NARRE":
                        loss, accuracy, mae, predictions = test_step_NARRE(u_test, i_test, userid_test, itemid_test, reuid, reiid, y_test)
                    else:
                        loss, accuracy, mae, predictions = test_step(u_test, i_test, userid_test, itemid_test, y_test)
                    loss_s += len(u_test) * loss
                    accuracy_s_test +=  len(u_test) * np.square(accuracy)
                    mae_s_test += len(u_test) * mae
                    
                    y_true.extend(y_test)
                    preds = np.reshape(predictions, (len(y_test),))                 
                    y_pred.extend(preds)
                    
                logging.info("loss_test {:g}, rmse_test {:g}, mae_test {:g}".format(loss_s / test_length,
                                                                                    np.sqrt(accuracy_s_test / test_length),
                                                                                    mae_s_test / test_length))
                write_to_file("%s/%s_epoch_%s_pred.txt" % (FLAGS.output_path, args.model_type, str(index)), y_true, y_pred)
                
    print('End Experiment')
