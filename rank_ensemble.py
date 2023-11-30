import json
from collections import defaultdict
import math
from nltk.corpus import stopwords
import numpy as np
import argparse
from utils import *

from nltk.corpus import wordnet

def are_words_closely_related(word1, word2):
    # Get synsets for the words
    synsets_word1 = wordnet.synsets(word1)
    synsets_word2 = wordnet.synsets(word2)

    hypernym = []
    # Check if one word is a hypernym of the other or they share a common hypernym
    for synset1 in synsets_word1:
        for synset2 in synsets_word2:
            temp = 0
            list1 = list(synset1.closure(lambda s: s.hypernyms())) + list(synset1.closure(lambda s: s.hyponyms()))
            list2 = list(synset2.closure(lambda s: s.hypernyms())) + list(synset2.closure(lambda s: s.hyponyms()))
            for term1 in list1:
              if term1 in list2:
                temp += 1
            hypernym.append(temp)
    if len(hypernym) == 0:
      return 0
    return max(hypernym)


def get_similarity(word1, word2):
    similarity = None
    # Get synsets for the word
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)

    temp = []
    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1 and synset2:
                temp.append(synset1.path_similarity(synset2))
    
    if len(temp) == 0:
      similarity = 0.
    similarity = max(temp)
    return similarity

def rank_ensemble(args, topk=20):

    ## embedding got from CatE
    word2emb = load_cate_emb(f'datasets/{args.dataset}/emb_{args.topic}_w.txt')
    ## embedding got from bert(PLM)
    word2bert = load_bert_emb(f'datasets/{args.dataset}/{args.dataset}_bert')

    caseolap_results = []
    with open(f'datasets/{args.dataset}/intermediate_2.txt') as fin:
        for line in fin:
            data = line.strip()
            _, res = data.split(':')
            caseolap_results.append(res.split(','))
            
    cur_seeds = []
    with open(f'datasets/{args.dataset}/{args.topic}_seeds.txt') as fin:
        for line in fin:
            data = line.strip().split(' ')
            cur_seeds.append(data)


    with open(f'datasets/{args.dataset}/{args.topic}_seeds.txt', 'w') as fout:
        for seeds, caseolap_res in zip(cur_seeds, caseolap_results):
            word2mrr = defaultdict(float)

            # cate mrr
            word2cate_score = {word:np.mean([np.dot(word2emb[word], word2emb[s]) for s in seeds]) for word in word2emb}
            r = 1. ## r suppose to means rank
            for w in sorted(word2cate_score.keys(), key=lambda x: word2cate_score[x], reverse=True)[:topk]:
                if w not in word2bert: continue
                word2mrr[w] += 1./r
                r += 1
                 
            # bert mrr
            word2bert_score = {word:np.mean([np.dot(word2bert[word], word2bert[s]) for s in seeds]) for word in word2bert}
            r = 1.
            for w in sorted(word2bert_score.keys(), key=lambda x: word2bert_score[x], reverse=True)[:topk]:
                if w not in word2emb: continue
                word2mrr[w] += 1./r
                r += 1
            
            # caseolap mrr
            r = 1.
            for w in caseolap_res[:topk]:
                word2mrr[w] += 1./r
                r += 1
            

            def wordnet_score(word2mrr):
                r = 1.
                similarity = 0.
                temp2 = []
                for w in word2mrr.keys():
                    for s in seeds:
                        temp2.append(get_similarity(w, s))                    
                    similarity = max(temp2)
                word2mrr[w]  += similarity
                r += 1

            
            r = 1.
            for w in word2mrr.keys():
                for s in seeds:                  
                    word2mrr[w] = word2mrr[w]+ are_words_closely_related(w, s)/100
            r += 1

            wordnet_score(word2mrr)   
            ## word2mrr 在这里包含了三种办法的平均rank 如果能用这个平均rank去和seed的关系做判断再去修改这个rank会准确（prompt-based）（有点难）
            ## word2mrr 如果给rank再加上一个wordnet的score也许会更准确
            ## word2mrr 这个rank也相当于一种normalized过后的score
            score_sorted = sorted(word2mrr.items(), key=lambda x: x[1], reverse=True)  ##从大到小sort
            top_terms = [x[0].replace(' ', '') for x in score_sorted if x[1] > args.rank_ens and x[0] != '']
            fout.write(' '.join(top_terms) + '\n')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='nyt', type=str)
    parser.add_argument('--topic', default='topic', type=str)
    parser.add_argument('--topk', default=20, type=int)
    parser.add_argument('--rank_ens', default=0.3, type=float)
    args = parser.parse_args()

    rank_ensemble(args, args.topk)