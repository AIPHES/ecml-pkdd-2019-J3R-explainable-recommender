import os
import json
import pandas as pd
import pickle
import numpy as np
import argparse

def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Load the Amazon dataset and splpt the data')
 
    parser.add_argument('-d', '--data_set', type=str, help='Dataset: music, tv, toys, electronics, kindle, cds', required=True)

    args = parser.parse_args()

    return args
            
            
args = get_args()

TPS_DIR = '../data/%s' % (args.data_set)

data_dict = {"music": "Digital_Music_5.json", 
            "tv": "reviews_Movies_and_TV_5.json",
            "toys": "reviews_Toys_and_Games_5.json",
            "kindle": "reviews_Kindle_Store_5.json",
            "electronics": "reviews_Electronics_5.json",
            "cds": "reviews_CDs_and_Vinyl_5.json"}

TP_file = os.path.join(TPS_DIR, data_dict[args.data_set])

f= open(TP_file)
users_id=[]
items_id=[]
ratings=[]
reviews=[]
summaries=[]
np.random.seed(2017)

for line in f:
    js=json.loads(line)
    if str(js['reviewerID'])=='unknown':
        print "unknown"
        continue
    if str(js['asin'])=='unknown':
        print "unknown2"
        continue
    reviews.append(js['reviewText'])
    users_id.append(str(js['reviewerID'])+',')
    items_id.append(str(js['asin'])+',')
    ratings.append(str(js['overall']))
    summaries.append(str(js['summary']))


data=pd.DataFrame({'user_id':pd.Series(users_id),
                   'item_id':pd.Series(items_id),
                   'ratings':pd.Series(ratings),
                   'reviews':pd.Series(reviews),
                   'summary':pd.Series(summaries)})[['user_id','item_id','ratings','reviews', 'summary']]

def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count
usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')
unique_uid = usercount.index
unique_sid = itemcount.index
item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
def numerize(tp):
    uid = map(lambda x: user2id[x], tp['user_id'])
    sid = map(lambda x: item2id[x], tp['item_id'])
    tp['user_id'] = uid
    tp['item_id'] = sid
    return tp

data=numerize(data)
tp_rating=data[['user_id','item_id','ratings']]


n_ratings = tp_rating.shape[0]
test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_1 = tp_rating[test_idx]
tp_train= tp_rating[~test_idx]

data2=data[test_idx]
data=data[~test_idx]


n_ratings = tp_1.shape[0]
test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)

test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_test = tp_1[test_idx]
tp_valid = tp_1[~test_idx]
tp_train.to_csv(os.path.join(TPS_DIR, '%s_train.csv' % (args.data_set)), index=False,header=None)
tp_valid.to_csv(os.path.join(TPS_DIR, '%s_valid.csv' % (args.data_set)), index=False,header=None)
tp_test.to_csv(os.path.join(TPS_DIR, '%s_test.csv' % (args.data_set)), index=False,header=None)

user_reviews={}
item_reviews={}
user_rid={}
item_rid={}
review_summary = {}

# Looping over train data
for i in data.values:
    review_summary[(i[0], i[1])] = [i[3], i[4]]
    if user_reviews.has_key(i[0]):
        user_reviews[i[0]].append(i[3])
        user_rid[i[0]].append(i[1])
    else:
        user_rid[i[0]]=[i[1]]
        user_reviews[i[0]]=[i[3]]
    if item_reviews.has_key(i[1]):
        item_reviews[i[1]].append(i[3])
        item_rid[i[1]].append(i[0])
    else:
        item_reviews[i[1]] = [i[3]]
        item_rid[i[1]]=[i[0]]

# Looping over test data
for i in data2.values:
    review_summary[(i[0], i[1])] = [i[3], i[4]]
    if user_reviews.has_key(i[0]):
        l=1
    else:
        user_rid[i[0]]=[0]
        user_reviews[i[0]]=['0']
        
    if item_reviews.has_key(i[1]):
        l=1
    else:
        item_reviews[i[1]] = [0]
        item_rid[i[1]]=['0']
        
pickle.dump(user_reviews, open(os.path.join(TPS_DIR, 'user_review'), 'wb'))
pickle.dump(item_reviews, open(os.path.join(TPS_DIR, 'item_review'), 'wb'))
pickle.dump(user_rid, open(os.path.join(TPS_DIR, 'user_rid'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TPS_DIR, 'item_rid'), 'wb'))
pickle.dump(review_summary, open(os.path.join(TPS_DIR, 'review_summary'), 'wb'))
usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')

print np.sort(np.array(usercount.values))

print np.sort(np.array(itemcount.values))
