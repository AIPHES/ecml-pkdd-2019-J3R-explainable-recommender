import sys
import os
import argparse
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import mkdirp
from utils.misc import set_logger
from utils.data_helpers import load_data

from summarize.upper_bound import ExtractiveUpperbound
from summarize.sume_wrap import SumeWrap
from summarize.sumy.nlp.tokenizers import Tokenizer
from summarize.sumy.parsers.plaintext import PlaintextParser
from summarize.sumy.summarizers.lsa import LsaSummarizer
from summarize.sumy.summarizers.kl import KLSummarizer
from summarize.sumy.summarizers.luhn import LuhnSummarizer
from summarize.sumy.summarizers.lex_rank import LexRankSummarizer
from summarize.sumy.summarizers.text_rank import TextRankSummarizer
from summarize.sumy.nlp.stemmers import Stemmer
from nltk.corpus import stopwords
from rouge.rouge import Rouge

import pandas as pd
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Upper Bound for Summarization')
    # -- summary_size: 100, 200, 400
    parser.add_argument('--summary_size', type=str, help='Summary Length', required=True)

       # --dataset: amazon, yelp, tripadvisor
    parser.add_argument('--dataset', type=str, help='Dataset ex: amazon, yelp, tripadvisor',
                        required=True)

    # --domain: movies, books, electronics, restaurants, travel
    parser.add_argument('--domain', type=str, help='Domain Type: movies, electronics, books,\
                         restaurants, travel', required=True)
    # --split: split type to load "users", "items", "ratings"
    parser.add_argument('--split', type=str,
                        help='Split type: users, items, ratings', required=False,
                        default="users")

    # --language: english, german
    parser.add_argument('--language', type=str, help='Language: english, german', required=False,
                        default='english')

    parser.add_argument('-io', '--iobasedir', type=str, help='IO base directory', required=False,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    args = parser.parse_args()

    return args

def print_scores(algo_name, summary_sents, refs, rouge, summary_size, best_summary, best_score):    
    score = rouge(' '.join(summary_sents), refs, summary_size)
    
    if score['rouge_1_recall'] >= best_score:
        best_summary = summary_sents
        best_score = score['rouge_1_recall']
    
    logger.info('%s: ROUGE-1: %4f %4f %4f, ROUGE-2: %4f %4f %4f, ROUGE-SU4: %4f %4f %4f' % (algo_name, \
        score['rouge_1_f_score'], score['rouge_1_precision'], score['rouge_1_recall'], \
        score['rouge_2_f_score'], score['rouge_2_precision'], score['rouge_2_recall'], \
        score['rouge_su4_f_score'], score['rouge_su4_precision'], score['rouge_su4_recall']))

    return best_summary, best_score

def get_summary_scores(algo, docs, refs, summary_size, language, rouge, best_summary, best_score):
    summary = []
    if algo == 'UB1' or algo == 'UB2':
        temp_summary_size = summary_size
        while not summary:
            if algo == 'UB1':
                summarizer = ExtractiveUpperbound(language)
                summary = summarizer(docs, refs, temp_summary_size, ngram_type=1)    
            elif algo == 'UB2':
                summarizer = ExtractiveUpperbound(language)
                summary = summarizer(docs, refs, temp_summary_size, ngram_type=2)
                
            temp_summary_size += 5
            if temp_summary_size >= 30:
                break
        
        # If summary is empty assign the best summary     
        if not summary:
            summary = best_summary
        else:
            # If the summary is not empty and the score is less than the best summary 
            # then replace the summary with best summary
            score = rouge(' '.join(summary), refs, summary_size)
            if score['rouge_1_recall'] < best_score:
                summary = best_summary
    elif algo == "ICSI":
        summarizer = SumeWrap(language)
        summary = summarizer(docs, summary_size) 
    else:
        doc_string = u'\n'.join([u'\n'.join(doc_sents) for doc_sents in docs]) 
        parser = PlaintextParser.from_string(doc_string, Tokenizer(language))
        stemmer = Stemmer(language)
        if algo == 'LSA':
            summarizer = LsaSummarizer(stemmer)
        if algo == 'KL':
            summarizer = KLSummarizer(stemmer)
        if algo == 'Luhn':
            summarizer = LuhnSummarizer(stemmer)
        if algo == 'LexRank':
            summarizer = LexRankSummarizer(stemmer)
        if algo == 'TextRank':
            summarizer = TextRankSummarizer(stemmer)

        summarizer.stop_words = frozenset(stopwords.words(language))
        summary = summarizer(parser.document, summary_size)
        if not summary:
            summary = []
    logger.info('Summary: %s' % summary)
    best_summary, best_score = print_scores(algo, summary, refs, rouge, summary_size, best_summary, best_score)
    return best_summary, best_score

def main():

    args = get_args()
    rouge_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rouge/RELEASE-1.5.5/')

    data_path = os.path.join(args.iobasedir, 'processed/', args.dataset, args.domain, args.split)
    log_path = os.path.join(args.iobasedir, 'logs')
    log_file = os.path.join(args.iobasedir, 'logs', 'baselines_rsumm_%s_%s_%s_%s.log' % (args.dataset, args.domain, args.split, str(args.summary_size)))
    mkdirp(log_path)
    set_logger(log_file)

    data_file = os.path.join(data_path, 'test0.csv')
    df = pd.read_csv(data_file, sep=",", quotechar='"', engine='python', header=None, skiprows=1, names=["user_id","product_id", "rating", "review", "nouns", "summary", 'time'])

#   check_index = 1099
    for index, row in df.iterrows():  
#        if index != check_index:
#            continue
        topic = row['user_id'] + '_' + row['product_id']
        docs = [ [sent] for sent in  sent_tokenize(row['review'].strip()) ]
        refs = [ sent_tokenize(row['summary'].strip()) ]
        if not refs:
            continue

        if not args.summary_size:
            summary_size = len(" ".join(refs[0]).split(' '))
        else:
            summary_size = int(args.summary_size)
        
        logger.info('Topic ID: %s', topic)
        logger.info('###')
        logger.info('Summmary_len: %d', summary_size)
        
        rouge = Rouge(rouge_dir)
        algos = ['Luhn', 'LexRank', 'TextRank', 'LSA', 'KL', "ICSI", 'UB1', 'UB2']
        best_summary = []
        best_score = 0.0
        for algo in algos:
            best_summary, best_score = get_summary_scores(algo, docs, refs, summary_size, args.language, rouge, best_summary, best_score)
        
        rouge._cleanup()
        logger.info('###')
    

if __name__ == '__main__':
    main()
