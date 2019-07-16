import tensorflow as tf
import logging 
import pickle
import argparse

def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Parameters for Training Models')
 
    parser.add_argument('-d', '--data_set', type=str, help='Dataset music, tv, electronics, cds, toys, kindle, yelp', required=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = get_args()
    
    data_set = args.data_set
    
    print("Dataset:", data_set)
    tf.flags.DEFINE_string("para_data", "../data/%s/%s.para" % (data_set, data_set), "Data parameters")
    tf.flags.DEFINE_string("train_data", "../data/%s/%s.train" % (data_set, data_set), "Data for training")
    tf.flags.DEFINE_string("valid_data","../data/%s/%s.valid" % (data_set, data_set), " Data for validation")
    tf.flags.DEFINE_string("test_data", "../data/%s/%s.test" % (data_set, data_set), "Data for testing")
    tf.flags.DEFINE_string("output_path", "../outputs/%s/" % (data_set), "Output Path")
    tf.flags.DEFINE_string("logs", "../logs/%s_stats.log" % (data_set), "Logging path")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

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

    print("User Num", user_num)
    print("Item Num", item_num)
    print("User Vocab", len(vocabulary_user))
    print("Item Vocab", len(vocabulary_item))    
    
    logging.info("User Num %s" % user_num)
    logging.info("Item Num %s" % item_num)
    logging.info("User Vocab %s" % len(vocabulary_user))
    logging.info("Item Vocab %s" % len(vocabulary_item))    
    