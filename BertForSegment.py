# -*- coding: utf-8 -*-
#__auther__: LI BOYANG
#__date__: 2020.07.15

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import os
import json
from transformers import *
import re
import math
import logging
logging.basicConfig(level=logging.ERROR)

class BertForSegment(nn.Module):
    def __init__(self,config_path,DEVICE,maxlen=0,dropout=0.05):
        super(BertForSegment,self).__init__()

        #Parameters for BERT Model:
        self.device=DEVICE
        self.model=BertModel.from_pretrained(config_path).to(self.device)
        self.tokenizer=BertTokenizer.from_pretrained(config_path)
        self.maxlen=150 if not maxlen else maxlen
        self.hidden_size=self.model.config.hidden_size
        self.num_hidden_heads=self.model.config.num_attention_heads
        self.d_ff=2048
        self.dropout=nn.Dropout(dropout)
        #Parameters for BLEU:
        self.only_word=re.compile(r'[^\w]')

        #Paramters for top-compute-layer
        self.FFN1=nn.Linear(self.hidden_size,self.d_ff,bias=True)
        self.FFN2=nn.Linear(self.d_ff,self.hidden_size,bias=True)

        #self.linear1=nn.Linear(self.hidden_size*3,self.hidden_size,bias=True)
        self.conv1=nn.Conv1d(self.hidden_size*3,self.hidden_size,2,padding=1)
        self.conv2=nn.Conv1d(self.hidden_size*3,self.hidden_size,3,padding=1)
        self.conv3=nn.Conv1d(self.hidden_size*3,self.hidden_size,4,padding=1)
        #self.conv1=nn.Conv1d(self.hidden_size*2,self.num_hidden_layers,3,padding=1)
        self.linear2=nn.Linear(self.hidden_size*3,self.hidden_size,bias=True)
        self.linear3=nn.Linear(self.hidden_size,1,bias=True)

        dictionary_path=os.path.join(os.path.abspath('.'),'config','similarity_dictionary.json')
        with open(dictionary_path,'r') as f:
            dictionary=f.read()
            self.dictionary=json.loads(dictionary)

    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))

    def forward(self,sent1,sent2):
        #@param:sent1:ref
        #@param:sent2:hyp
        if not self.training:
            emb1=self.sentence_embedding_test(sent1)
            emb2=self.sentence_embedding_test(sent2)
        else:
            emb1=self.sentence_embedding_train(sent1)
            emb2=self.sentence_embedding_train(sent2)

        emb_score=self.evaluate_func(emb1,emb2)
        return emb_score

    def evaluate_func(self,emb1,emb2):
        #print(emb1.size())
        #print(emb2.size())
        #Distance
        if emb2.size(0)!=emb1.size(0):
            emb2=emb2.repeat(emb1.size(0),1,1)

        SUB=torch.abs(emb1-emb2)
        SUB=torch.cat((SUB,emb1,emb2),dim=-1)
        SUB=self.dropout(SUB)
        SUB1=self.conv1(SUB.transpose(1,2).contiguous())
        SUB2=self.conv2(SUB.transpose(1,2).contiguous())
        SUB3=self.conv3(SUB.transpose(1,2).contiguous())
        SUB1=F.max_pool1d(SUB1,SUB1.size(-1),padding=0).squeeze(-1)
        SUB2=F.max_pool1d(SUB2,SUB2.size(-1),padding=0).squeeze(-1)
        SUB3=F.max_pool1d(SUB3,SUB3.size(-1),padding=0).squeeze(-1)

        #Here the size:[batch,hidden_heads_num]
        SUB=torch.cat((SUB1,SUB2,SUB3),dim=-1)
        SUB=torch.relu(self.linear2(SUB))
        SUB=self.linear3(SUB)

        return SUB

    def sentence_embedding_test(self,sentence):
        #Because of test and inference process, 
        #the input of this function should be a list with only one sentence.
        sentence=sentence[0]

        sentence=self.only_word.sub('',sentence).lower()
        for each,value in self.dictionary.items():
            if each in sentence:
                sentence=sentence.replace(each,value)
        input=self.tokenizer.encode(sentence)

        length=len(input)
        input=input+(self.maxlen-length)*[self.tokenizer.pad_token_id]
        segment=torch.tensor([0]*self.maxlen).unsqueeze(0).to(self.device)
        atten_mask=[1.]*length+[0.]*(self.maxlen-length)
        atten_mask=torch.tensor(atten_mask).unsqueeze(0).to(self.device)
        input=torch.tensor(input).unsqueeze(0).to(self.device)
        #print(input.size())
        #print(segment.size())
        #print(atten_mask.size())
        sentence_embedding=self.model(input_ids=input,\
                            attention_mask=atten_mask,\
                            token_type_ids=segment)[0]

        assert sentence_embedding.size()==torch.Size([1,self.maxlen,self.model.config.hidden_size])
        return sentence_embedding

    def sentence_embedding_train(self,sentences):
        #@Because of train process,
        #the sentences here should be a list of str
        sentences_ids=[]
        atten_masks=[]
        segments=[]


        for each_sent in sentences:
            each_sent=self.only_word.sub('',each_sent).lower()
            for each,value in self.dictionary.items():
                if each in each_sent:
                    each_sent=each_sent.replace(each,value)

            each_ids=self.tokenizer.encode(each_sent)
            length=len(each_ids)
            each_ids=each_ids+(self.maxlen-length)*[self.tokenizer.pad_token_id]
            atten_mask=[1.]*length+[0.]*(self.maxlen-length)
            segment=[0]*self.maxlen

            sentences_ids.append(each_ids)
            atten_masks.append(atten_mask)
            segments.append(segment)

        sentence_input=torch.tensor(sentences_ids).to(self.device)
        atten_masks=torch.tensor(atten_masks).to(self.device)
        segments=torch.tensor(segments).to(self.device)

        sentence_embeddings=self.model(input_ids=sentence_input,\
                                                token_type_ids=segments,\
                                                attention_mask=atten_masks)[0]

        return sentence_embeddings

    def inference(self,sentence,ref_embs):
        #@param:ref_embs:the tensor of all standard questions which should be stored as a document.
        #                   to reduce the compute time.
        #@param:standard_q:list of all standard questions in str.
        with torch.no_grad():
            hyp_emb=self.sentence_embedding_test([sentence])
            emb_score=self.evaluate_func(ref_embs,hyp_emb)

            top3_score=[x.item() for x in torch.topk(score,k=3,dim=0)[0]]
            top3_index=[x.item() for x in torch.topk(score,k=3,dim=0)[1]]
        return top3_score,top3_index
    
    def sentences2ids(self,sentences):
        #@param:sentences: the list of sentences
        #@return: list: each element is a tensor with size(batch_size,seq_len(with CLS and SEP))
        with torch.no_grad():
            sentences_ids=[]
            for each_sent in sentences:
                each_sent=self.only_word.sub('',each_sent).lower()
                for each,value in self.dictionary.items():
                    if each in each_sent:
                        each_sent=each_sent.replace(each,value)
                #each_sent=['[CLS]']+each_sent+['[SEP]']
                each_sent=self.tokenizer.encode(each_sent)
                each_sent=torch.tensor(each_sent).unsqueeze(0).to(self.device)
                sentences_ids.append(each_sent)
        return sentences_ids

    @staticmethod
    def load_weights(bertpath,path,DEVICE):
        #@ params:bertpath:path to initialization of this model
        #@ params:path:load the trained weights of this model
        model_params=torch.load(path,map_location=DEVICE)
        model=BertForSegment(bertpath,DEVICE)
        model.load_state_dict(state_dict=model_params['BertForSegment'])
        model=model.to(DEVICE)
        return model

