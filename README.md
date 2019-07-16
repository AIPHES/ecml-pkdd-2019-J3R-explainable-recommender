# J3R: Joint Multi-task Learning of Ratings and Review Summaries

In this project, we develop a J3R: Joint Multi-task Learning of Ratings and Review Summaries. This repository contains preprocessing codes and models for training the models in the paper. 

If you reuse this software, please use the following citation:

```
@inproceedings{J3R-multi-task,
    title = {J3R: Joint Multi-task Learning of Ratings and Review Summaries},
    author = {P.V.S., Avinesh and Ren, Yongli and Meyer, Christian M. and Jeffrey Chan and Zhifeng Bao and Mark Sanderson},
    booktitle = {Machine Learning and Knowledge Discovery in Databases - European Conference,
               {ECML} {PKDD} 2019, Wurzburg, Germany, September 16-20, 2019, Proceedings,
    pages = {to appear},
    Xmonth = sep,
    year = {2019},
    location = {Wurzburg, Germany},
}
```
> **Abstract:** We learn user preferences from ratings and reviews by using multi-task learning (MTL) of rating prediction and summarization of item reviews. 
Reviews of an item tend to describe detailed user preferences (e.g., the cast, genre, or screenplay of a movie). A summary of such a review or a rating describes an overall user experience of the item. Our objective is to learn latent vectors which are shared across rating prediction and review summary generation.
Additionally, the learned latent vectors and the generated summary act as explanations for the recommendation. Our MTL-based approach J3R uses a multi-layer perceptron for rating prediction, combined with pointer-generator networks with attention mechanism for the summarization component. We provide empirical evidence for joint learning of rating prediction and summary generation being beneficial for recommendation by conducting experiments on the Yelp dataset and six domains of the Amazon 5-core dataset. Additionally, we provide two ways of explanations visualizing (a) the user vectors on different topics of a domain, computed from our J3R approach and (b) a ten-word review summary of a review and the attention highlights generated on the review based on the user--item vectors.

**Contact person:**
* Avinesh P.V.S., first_name AT aiphes.tu-darmstadt.de, first_name.last_name AT gmail.com
* http://www.ukp.tu-darmstadt.de/
* http://www.tu-darmstadt.de/

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.


## Environments

- python 2.7
- Tensorflow (version: 0.12.1)
- numpy
- pandas


## Dataset

In our experiments, we use the datasets from [Amazon 5-core](http://jmcauley.ucsd.edu/data/amazon) and [Yelp Challenge 2018](https://www.yelp.com/dataset_challenge).

## Data preprocessing:

```
# Processed data for all models for music, toys, kindle, electronics, tv, cds and yelp
python utils/loaddata.py -d music
python utils/data_pro.py -d music

# Processed data for Seq2Seq and Pointer-Networks
python utils/data_pro_for_nnsum.py -d music -o onmt
```

## Train and evaluate the model:

```
CUDA_VISIBLE_DEVICES=5 python train.py -d music -m DeepCoNN -b 32
CUDA_VISIBLE_DEVICES=5 python train.py -d music -m NAARE -b 32
CUDA_VISIBLE_DEVICES=5 python train.py -d music -m J3R -b 32
```

## Baselines:

```
# Rating Prediction
python refs/librec/run.py
python refs/HFT/run.py 

# Summarization
sh summarize/OpenNMT-py/train.sh
python summarize/baseline.py
```

