# -*- coding: utf-8 -*-
#__auther__: LI BOYANG
#__date__: 2020.07.15

import re
import os
import json
import math
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction,brevity_penalty

def bleu_compute_without_sw(refs,hyps,smf,bleu_weight,stopwords):
    #param:refs:list of the ref sentences, if number of refs is one, should also be a list.
    #param:hyps:list of the hyp sentences, if number of hyps is one, should also be a list.

    gains=[]
    for ref,hyp in zip(refs,hyps):
        ref=sentence_pretreatment(ref,stopwords,True)
        hyp=sentence_pretreatment(hyp,stopwords,False)

        ref=[[x for x in ref]]
        hyp=[y for y in hyp]
        bleu=sentence_bleu(ref,hyp,weights=bleu_weight,smoothing_function=smf.method1)
        score=bleu
        #gains.append((math.exp(score)-1)/(math.exp(1)-1))
        gains.append((math.exp(score)-1)/(math.exp(1)-1))
    return gains

def bleu_compute(refs,hyps):
    only_word=re.compile(r'\?|？|，')
    
    dictionary_path=os.path.join(os.path.abspath('.'),'config','similarity_dictionary.json')
    with open(dictionary_path,'r') as f:
        dictionary=f.read()
        dictionary=json.loads(dictionary)
    
    smf=SmoothingFunction()
    bleu_weight=(0.05,0.5,0.4,0.05)

    new_refs=[]
    new_hyps=[]

    for ref,hyp in zip(refs,hyps):
        ref=only_word.sub('',ref).lower()
        hyp=only_word.sub('',hyp).lower()

        for each,value in dictionary.items():
            if each in ref:
                ref=ref.replace(each,value)
            if each in hyp:
                hyp=hyp.replace(each,value)

        new_refs.append(ref)
        new_hyps.append(hyp)


    stopwords_multi_path=os.path.join(os.path.abspath('.'),'config','stopwords','stopwords_multi_gram')
    stopwords_all_path=os.path.join(os.path.abspath('.'),'config','stopwords','new_stopwords')
    stopwords_multi=torch.load(stopwords_multi_path)
    stopwords_all=torch.load(stopwords_all_path)

    gain1=bleu_compute_without_sw(new_refs,new_hyps,smf,bleu_weight,stopwords_multi)
    gain2=bleu_compute_without_sw(new_refs,new_hyps,smf,bleu_weight,stopwords_all)
    
    gain=[(x+y)/2 for x,y in zip(gain1,gain2)]
    return gain,max(gain),gain.index(max(gain))

def bleu_inference(refs,hyps):
    only_word=re.compile(r'\?|？|，')
    
    dictionary_path=os.path.join(os.path.abspath('.'),'config','similarity_dictionary.json')
    with open(dictionary_path,'r') as f:
        dictionary=f.read()
        dictionary=json.loads(dictionary)
    
    smf=SmoothingFunction()
    bleu_weight=(0.05,0.5,0.4,0.05)

    new_refs=[]
    new_hyps=[]

    for ref,hyp in zip(refs,hyps):
        ref=only_word.sub('',ref).lower()
        hyp=only_word.sub('',hyp).lower()

        for each,value in dictionary.items():
            if each in ref:
                ref=ref.replace(each,value)
            if each in hyp:
                hyp=hyp.replace(each,value)

        new_refs.append(ref)
        new_hyps.append(hyp)


    stopwords_multi_path=os.path.join(os.path.abspath('.'),'config','stopwords','stopwords_multi_gram')
    stopwords_all_path=os.path.join(os.path.abspath('.'),'config','stopwords','new_stopwords')
    stopwords_multi=torch.load(stopwords_multi_path)
    stopwords_all=torch.load(stopwords_all_path)

    gain1=bleu_compute_without_sw(new_refs,new_hyps,smf,bleu_weight,stopwords_multi)
    gain2=bleu_compute_without_sw(new_refs,new_hyps,smf,bleu_weight,stopwords_all)
    
    gain=[(x+y)/2 for x,y in zip(gain1,gain2)]
    if max(gain)<0.0325:
        gain=[-0.4]*len(gain)
    return gain

def sentence_pretreatment(sentence,stopwords,is_ref=False):
    if not is_ref:
        for stopword in stopwords:
            if stopword in sentence:
                sentence=sentence.replace(stopword,'、')
    else:
        for stopword in stopwords:
            if stopword in sentence:
                sentence=sentence.replace(stopword,'|')

    if sentence[0]=='、':
        new_sentence=''
    else:
        new_sentence=sentence[0]

    pos=1
    while pos<len(sentence):
        if sentence[pos-1]!=sentence[pos]:
            new_sentence+=sentence[pos]
        pos+=1
    if new_sentence[-1]=='、':
        new_sentence=new_sentence[:-1]
    return new_sentence

