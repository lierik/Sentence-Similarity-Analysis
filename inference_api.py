# -*- coding: utf-8 -*-
#__auther__: LI BOYANG
#__date__: 2020.07.15

import os
import sys
import time
import torch
from BertForSegment import BertForSegment
from bleu_pretest_multi_gram import bleu_inference

'''
***********INTRODUCTION**************

This python code is the api for inference.

If the received question can be matched with any standard questions in dataset,
return [str1,str2,str3]->str,[score1, score2,score2]->float

If the If the received question can NOT be matched with any standard questions in dataset,
then return ['Not in Dataset'],[0]

'''

def inference_api(to_be_searach):

    config_path=os.path.join(os.path.abspath('.'),'config')
    output_path=os.path.join(os.path.abspath('.'),'After_training')

    DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu'
    threshold=0.06

    model_config=os.path.join(output_path,'newmodel_aftertraining.pth')
    model=BertForSegment.load_weights(config_path,model_config,DEVICE)
    model.eval()

    standard_dataset=torch.load(os.path.join(os.path.abspath('.'),'data','standard_qa'))
    standard_q=[x['Q'] for x in standard_dataset]
    standard_vec_path=os.path.join(os.path.abspath('.'),'data','standard_vec')
    
    if os.path.exists(standard_vec_path):
        standard_q_embs=torch.load(standard_vec_path)
    else:
        standard_q_embs=[]
        with torch.no_grad():
            for each_q in standard_q:
                standard_q_emb=model.sentence_embedding_test([each_q])
                standard_q_embs.append(standard_q_emb)
        
            standard_q_embs=torch.cat(standard_q_embs,dim=0).to(DEVICE)
            torch.save(standard_q_embs,standard_vec_path)

    starttime=time.time()
    hyps=[to_be_searach]*len(standard_q)
    hyp_embs=model.sentence_embedding_test([to_be_searach])
    emb_score=model.evaluate_func(standard_q_embs,hyp_embs)
    bleu_score=bleu_inference(standard_q,hyps)
    bleu_score=torch.tensor(bleu_score).unsqueeze(-1).to(DEVICE)
    score=bleu_score+emb_score

    top3_score_index=[x.item() for x in torch.topk(score,k=3,dim=0)[1]]
    top3_score=[x.item() for x in torch.topk(score,k=3,dim=0)[0]]

    endtime=time.time()
    used_time=endtime-starttime

    if top3_score[0]>threshold:
        return [standard_dataset[top3_score_index[x]]['A'] for x in range(3)],top3_score
    else:
        return ['Not in Dataset'],[0]



if __name__=='__main__':

    result=inference_api('今天天气好不好？')
    print(result)

    result=inference_api('什么是通货膨胀啊？')
    print(result)
