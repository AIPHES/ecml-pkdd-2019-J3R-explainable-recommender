import os
import json
import pandas as pd
import pickle
import numpy as np
import codecs

def load_tips(TPS_DIR):
    tip_hash = {}
    TP_file = os.path.join(TPS_DIR, 'tip.json')
    with codecs.open(TP_file, 'r', encoding = 'utf-8') as fp:
        for line in fp:
            line = line.encode('ascii','ignore')
            js = json.loads(line)
            user_id = str(js['user_id'])
            business_id = str(js['business_id'])
            tip_hash[(user_id, business_id)] = str(js['text'])
    return tip_hash

TPS_DIR = '../data/yelp'
TP_file = os.path.join(TPS_DIR, 'review.json')
tips_hash = load_tips(TPS_DIR)


f= open(TP_file)
users_id,items_id, ratings, reviews, summaries = [], [], [], [], []

np.random.seed(2017)

for line in f:
    line=line.decode('utf-8','ignore').encode("utf-8")
    js=json.loads(line)
    if str(js['user_id'])=='unknown':
        print "unknown"
        continue
    if str(js['business_id'])=='unknown':
        print "unknown2"
        continue
        
    id_tip = (str(js['user_id']), str(js['business_id']))

    reviews.append(js['text'])
    users_id.append(str(js['user_id'])+',')
    items_id.append(str(js['business_id'])+',')
    ratings.append(str(js['stars']))
    if id_tip in tips_hash:        
        summaries.append(str(tips_hash[id_tip]))
    else:
        summaries.append(str("None"))
    
data=pd.DataFrame({'user_id':pd.Series(users_id),
                   'item_id':pd.Series(items_id),
                   'ratings':pd.Series(ratings),
                   'reviews':pd.Series(reviews),
                   'summary':pd.Series(summaries)})[['user_id','item_id','ratings','reviews', 'summary']]

def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

MIN_USER_COUNT = 5
MIN_SONG_COUNT = 5
def filter_triplets(tp, min_uc=MIN_USER_COUNT, min_sc=MIN_SONG_COUNT):
    # Only keep the triplets for songs which were listened to by at least min_sc users.
    songcount = get_count(tp, 'item_id')

    tp = tp[tp['item_id'].isin(songcount.index[songcount >= min_sc])]

    # Only keep the triplets for users who listened to at least min_uc songs
    # After doing this, some of the songs will have less than min_uc users, but should only be a small proportion
    usercount = get_count(tp, 'user_id')

    tp = tp[tp['user_id'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, songcount = get_count(tp, 'user_id'), get_count(tp, 'item_id')
    return tp, usercount, songcount

data, usercount, itemcount=filter_triplets(data)

print("Print Data_shape", data.shape[0])
print("User counts", usercount.shape[0])
print("Item counts", itemcount.shape[0])

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
print("Ratings", tp_rating.shape[0])

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
tp_train.to_csv(os.path.join(TPS_DIR, 'yelp_train.csv'), index=False,header=None)
tp_valid.to_csv(os.path.join(TPS_DIR, 'yelp_valid.csv'), index=False,header=None)
tp_test.to_csv(os.path.join(TPS_DIR, 'yelp_test.csv'), index=False,header=None)

user_reviews={}
item_reviews={}
user_rid={}
item_rid={}
review_summary = {}

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
