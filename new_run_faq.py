
# -*- coding: utf-8 -*-
#__auther__: LI BOYANG
#__date__: 2020.07.15

import torch
import os
import re
import sys
import time
import random
import logging
import torch.nn as nn 
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
from transformers import AdamW
from BertForSegment import BertForSegment
from bleu_pretest_multi_gram import bleu_inference
logging.basicConfig(level=logging.ERROR)


def train_loop_1(train_dataset,test_dataset, model,batch_size,output_path):

    alldata=torch.load(train_dataset[0])
    full_batch_num=len(alldata)//batch_size
    epochs=3
    
    #criterion=nn.MSELoss(reduction='sum')
    criterion=nn.MSELoss()
    optimizer=AdamW(model.parameters(),lr=3e-6,betas=(0.9,0.999),eps=1e-9)

    print('Start Train Process No.1:')
    starttime=time.time()
    for epoch in range(epochs):
        model.train()
        random.shuffle(alldata)
        all_standard=[x['SQ'] for x in alldata]
        all_similar=[y['similar'] for y in alldata]
        all_label=[z['label'] for z in alldata]
        count=0
        for step in range(full_batch_num):
            #print
            if not step:
                batch_standard=all_standard[:batch_size]
                batch_similar=all_similar[:batch_size]
                batch_label=all_label[:batch_size]
            else:
                batch_standard=all_standard[step*batch_size:(step+1)*batch_size]
                batch_similar=all_similar[step*batch_size:(step+1)*batch_size]
                batch_label=all_label[step*batch_size:(step+1)*batch_size]
            assert len(batch_standard)==len(batch_similar)==len(batch_label)

            optimizer.zero_grad()
            emb_score=model(batch_standard,batch_similar)
            batch_label=torch.tensor(batch_label).view(-1,1).float().to(DEVICE)
            loss=criterion(emb_score,batch_label)

            if not (step+1)%500:
                endtime=time.time()
                already_spend_time=int(endtime-starttime)/3600
                now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print('Epoch: %d, step: %d, batch.loss: %.4f, used time: %.2fh, now: %s'% (epoch+1,step+1,loss,already_spend_time,now))
            loss.backward()
            optimizer.step()

        # the last batch of this epoch:
        if len(alldata)%batch_size:
            final_batch_standard=all_standard[full_batch_num*batch_size:]
            final_batch_similar=all_similar[full_batch_num*batch_size:]
            final_batch_label=all_label[full_batch_num*batch_size:]

            optimizer.zero_grad()
            scores=model(final_batch_standard,final_batch_similar)
            batch_label=torch.tensor(final_batch_label).view(-1,1).float().to(DEVICE)
            loss=criterion(scores,batch_label)
            loss.backward()
            optimizer.step()

    print('Finish the training process No.1 and Save the final state of model')
    final_state={'BertForSegment':model.state_dict()}
    torch.save(final_state,os.path.join(output_path,'newmodel_aftertraining.pth'))
    test_loop_2(test_dataset,model)

def train_loop_2(train_dataset,test_dataset, model,batch_size,output_path):
    alldata=torch.load(train_dataset[1])
    epochs=3
    criterion=nn.MSELoss()
    optimizer=AdamW(model.parameters(),lr=3e-6,betas=(0.9,0.999),eps=1e-9)

    starttime=time.time()
    dataset=[]
    standard_questions=[x['Q'] for x in alldata]
    for idx in range(len(alldata)):
        for each_similar in alldata[idx]['similar']:
            dataset.append((each_similar,idx))
    assert len(alldata)==len(standard_questions)

    print('Start Train Process No.2:')
    print('Num of all sentences to be trained in each epoch: %d' %(len(dataset),))

    full_batch_num=len(dataset)//batch_size
    label_length=len(alldata)
    for epoch in range(epochs):
        
        model.train()
        random.shuffle(dataset)

        dataset_similars=[]
        dataset_values=[]
        for unit in dataset:
            dataset_similars.append(unit[0])
            dataset_values.append(unit[1])

        for step in range(full_batch_num):
            
            with torch.no_grad():
                standard_q_embs=[]
                
                for standard_question in standard_questions:
                    q_emb=model.sentence_embedding_test([standard_question])
                    standard_q_embs.append(q_emb)
                standard_q_embs=torch.cat(standard_q_embs,dim=0).to(DEVICE)
                
            if not step:
                batch_similar=dataset_similars[:batch_size]
                batch_label=dataset_values[:batch_size]
            else:
                batch_similar=dataset_similars[step*batch_size:(step+1)*batch_size]
                batch_label=dataset_values[step*batch_size:(step+1)*batch_size]

            scores=[]
            labels=[]
            for i in range(batch_size):
                
                label=batch_label[i]
                label_tensor=[0]*label_length
                label_tensor[label]=1

                label_tensor=torch.tensor(label_tensor).float().unsqueeze(0)
                labels.append(label_tensor)

                hyp=[batch_similar[i]]
                hyps=hyp*len(standard_questions)
                hyp_embs=model.sentence_embedding_test(hyp)
                score=model.evaluate_func(standard_q_embs,hyp_embs)
                scores.append(score.transpose(1,0).contiguous())
            scores=torch.cat(scores,dim=0).to(DEVICE)
            labels=torch.cat(labels,dim=0).to(DEVICE)
            assert scores.size()==labels.size()

            optimizer.zero_grad()
            loss=criterion(scores,labels)

            if not (step+1)%100:
                endtime=time.time()
                already_spend_time=int(endtime-starttime)/3600
                now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print('Epoch: %d, step: %d, batch.loss: %.5f, lr: %s, used time: %.2fh , now: %s '% (epoch+1,step+1,loss,optimizer.param_groups[0]['lr'],already_spend_time,now))
                
            loss.backward()
            optimizer.step()
            if not (step+1) %500:
                now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print('Now save the checkpoint and do the test')
                final_state={'BertForSegment':model.state_dict()}
                torch.save(final_state,os.path.join(output_path,'1_checkpoint_newmodel.pth'))
                test_loop_2(test_dataset,model)

        test_loop_2(test_dataset,model)
    print('Finish the training process No.2 and Save the final state of model')
    final_state={'BertForSegment':model.state_dict()}
    torch.save(final_state,os.path.join(output_path,'newmodel_aftertraining.pth'))

def test_loop_1(test_dataset,model,threshold=0.85):
    test_start_time=time.time()
    alldata=torch.load(test_dataset[0])
    random.shuffle(alldata)
    model.eval()

    all_true_num=0
    all_false_num=0

    right_true_num=0
    right_false_num=0

    true_scores=[]
    false_scores=[]

    print('Start Test Process No.1 :')
    with torch.no_grad():
        for each in tqdm(alldata,ascii=True):
            ref=[each['SQ']]
            hyp=[each['similar']]
            label=each['label']

            score=model(ref,hyp).item()
            y_hat=score>threshold

            if label==True:
                true_scores.append(score)
                all_true_num+=1
                if y_hat==label:
                    right_true_num+=1
            else:
                false_scores.append(score)
                all_false_num+=1
                if y_hat==label:
                    right_false_num+=1


    acc_true=right_true_num/all_true_num
    acc_false=right_false_num/all_false_num
    test_end_time=time.time()
    test_time=int(test_end_time-test_start_time)

    print('After %d seconds, the Result of Test No.1 :'% (test_time))
    print('Num_similar: %d, Num_right_similar: %d, True Accuracy: %.2f %%' % (all_true_num,right_true_num,acc_true*100))
    print('Num_not_similar: %d, Num_right_not_similar: %d, False Accuracy: %.2f %%' % (all_false_num,right_false_num,acc_false*100))
    print('Max true score: %.4f, Min true score: %.4f' % (max(true_scores),min(true_scores)))
    print('Max false score: %.4f, Min false score: %.4f' % (max(false_scores),min(false_scores)))

def test_loop_2(test_dataset,model,threshold=0.85):
    alldata=torch.load(test_dataset[1])
    model.eval()

    all_pair=0
    top1_true_num=0
    top3_true_num=0
    standard_q_embs=[]
    standard_questions=[x['Q'] for x in alldata]
    all_top1_value=[]

    print('Start Test Process No.2 :')
    with torch.no_grad():
        for standard_question in standard_questions:
            q_emb=model.sentence_embedding_test([standard_question])
            standard_q_embs.append(q_emb)
        standard_q_embs=torch.cat(standard_q_embs,dim=0).to(DEVICE)

        for right_id in tqdm(range(len(alldata)),ascii=True):
            for hyp in alldata[right_id]['similar']:
                all_pair+=1
                hyps=[hyp]*len(standard_questions)

                hyp_embs=model.sentence_embedding_test([hyp])
                emb_score=model.evaluate_func(standard_q_embs,hyp_embs)
                score=emb_score
                top3_score_index=[x.item() for x in torch.topk(score,k=3,dim=0)[1]]
                top3_score=[x.item() for x in torch.topk(score,k=3,dim=0)[0]]
                top1_score_index=top3_score_index[0]
                all_top1_value.append(top3_score[0])

                if right_id in top3_score_index:
                    top3_true_num+=1
                    if top1_score_index==right_id:
                        top1_true_num+=1

    top1_true=top1_true_num/all_pair
    top3_true=top3_true_num/all_pair
    torch.save(all_top1_value,os.path.join(os.path.abspath('.'),'score_1'))
    print('The Result of Test No.2 :')
    #torch.save(all_top1_value,os.path.join(os.path.abspath('.'),'score_1'))
    print('Max right score: %.3f, Min right score: %.3f' %(max(all_top1_value),min(all_top1_value)))
    print('Number of correct in Top 3: %d, Top 3 Accuracy: %.3f %%,' % (top3_true_num,top3_true*100))
    print('Number of correct in Top 1: %d, Top 1 Accuracy: %.3f %%,' % (top1_true_num,top1_true*100))

def test_loop_3(test_dataset,dataset,model,threshold=0.85,printall=False):
    dataset=torch.load(dataset)
    alldata=torch.load(test_dataset)
    only_word=re.compile(r'[^\w]')
    model.eval()
    all_pair=0
    top1_true_num=0
    top3_true_num=0
    standard_q_embs=[]
    standard_questions=[x['Q'] for x in dataset]
    tested_q=[x['similar'] for x in alldata]
    tested_sq=[x['Q'] for x in alldata]

    count=0

    print('Start Test Process No.3 :')
    with torch.no_grad():
        for standard_question in standard_questions:
            q_emb=model.sentence_embedding_test([standard_question])
            standard_q_embs.append(q_emb)
        standard_q_embs=torch.cat(standard_q_embs,dim=0).to(DEVICE)


        for hyp,answer in zip(tested_q,tested_sq):
            print('\rnow start the count:{}'.format(count),end='')
            count+=1
            hyp=only_word.sub('',hyp[0]).lower()
            all_pair+=1
            hyps=[hyp]*len(standard_questions)

            hyp_embs=model.sentence_embedding_test([hyp])
            emb_score=model.evaluate_func(standard_q_embs,hyp_embs)
            score=emb_score+hyp_embs
            top3_score_index=[x.item() for x in torch.topk(score,k=3,dim=0)[1]]
            top3_score=[x.item() for x in torch.topk(score,k=3,dim=0)[0]]
            top1_score_index=top3_score_index[0]

            top3_sq=[standard_questions[m] for m in top3_score_index]

            if top3_score[0]<threshold:
                if answer=='unknown':
                    top3_true_num+=1
                    top1_true_num+=1
                    if printall:
                        print('--'*20)
                        print('unknown RIGHT')
                        print('hyp:',hyp)
                        print('answer:',answer)
                        #print('bleuscore:',[bleu_score[i] for i in top3_score_index])
                        print('top3_sq:',top3_sq)
                        print('top3 score:',top3_score)
                else:
                    print('--'*20)
                    print('UNKNOWN')
                    print('hyp:',hyp)
                    print('answer:',answer)
                    print('embscore:',[emb_score[i] for i in top3_score_index])
                    #print('bleuscore:',[bleu_score[i] for i in top3_score_index])
                    print('top3_sq:',top3_sq)
                    print('top3 score:',top3_score)


            elif answer in top3_sq:
                top3_true_num+=1
                if top3_sq[0]==answer:
                    top1_true_num+=1
                    if printall:
                        print('--'*20)
                        print('top1 RIGHT')
                        print('hyp:',hyp)
                        print('answer:',answer)
                        #print('bleuscore:',[bleu_score[i] for i in top3_score_index])
                        print('top3_sq:',top3_sq)
                        print('top3 score:',top3_score)
                else:
                    print('--'*20)
                    print('TOP 3 RIGHT')
                    print('hyp:',hyp)
                    print('answer:',answer)
                    #print('embscore:',[emb_score[i] for i in top3_score_index])
                    #print('bleuscore:',[bleu_score[i] for i in top3_score_index])
                    print('top3_sq:',top3_sq)
                    print('top3 score:',top3_score)
            else:
                print('--'*20)
                print('Not in TOP3')
                print('hyp:',hyp)
                print('answer:',answer)
                #print('embscore:',[emb_score[i] for i in top3_score_index])
                #print('bleuscore:',[bleu_score[i] for i in top3_score_index])
                print('top3_sq:',top3_sq)
                print('top3 score:',top3_score)


    top1_true=top1_true_num/all_pair
    top3_true=top3_true_num/all_pair
    print('The Result of Test No.2 :')
    #print('Max right score: %.3f, Min right score: %.3f' %(max(all_top1_value),min(all_top1_value)))
    print('Number of correct in Top 3: %d, Top 3 Accuracy: %.3f %%,' % (top3_true_num,top3_true*100))
    print('Number of correct in Top 1: %d, Top 1 Accuracy: %.3f %%,' % (top1_true_num,top1_true*100))

def test_loop_with_bleu(test_dataset,dataset,model,threshold=0.85,printall=False):
    dataset=torch.load(dataset)
    alldata=torch.load(test_dataset)
    only_word=re.compile(r'[^\w]')
    model.eval()
    all_pair=0
    top1_true_num=0
    top3_true_num=0
    standard_q_embs=[]
    standard_questions=[x['Q'] for x in dataset]
    tested_q=[x['similar'] for x in alldata]
    tested_sq=[x['Q'] for x in alldata]

    count=0
    a=0.9
    b=0.1
    print('Start Test Process No.3 :')
    with torch.no_grad():
        for standard_question in standard_questions:
            q_emb=model.sentence_embedding_test([standard_question])
            standard_q_embs.append(q_emb)
        standard_q_embs=torch.cat(standard_q_embs,dim=0).to(DEVICE)


        for hyp,answer in zip(tested_q,tested_sq):
            print('\rnow start the count:{}'.format(count),end='')
            count+=1
            hyp=only_word.sub('',hyp[0]).lower()
            all_pair+=1
            hyps=[hyp]*len(standard_questions)

            hyp_embs=model.sentence_embedding_test([hyp])
            emb_score=model.evaluate_func(standard_q_embs,hyp_embs)
            bleu_score=bleu_inference(standard_questions,hyps)
            bleu_score=torch.tensor(bleu_score).unsqueeze(-1).to(DEVICE)

            assert bleu_score.size()==emb_score.size()
            #score=emb_score+bleu_score
            score=b*emb_score+a*bleu_score
            top3_score_index=[x.item() for x in torch.topk(score,k=3,dim=0)[1]]
            top3_score=[x.item() for x in torch.topk(score,k=3,dim=0)[0]]
            top1_score_index=top3_score_index[0]

            top3_sq=[standard_questions[m] for m in top3_score_index]

            if top3_score[0]<threshold:
                if answer=='unknown':
                    top3_true_num+=1
                    top1_true_num+=1
                    if printall:
                        print('--'*20)
                        print('unknown RIGHT')
                        print('hyp:',hyp)
                        print('answer:',answer)
                        print('embscore:',[emb_score[i] for i in top3_score_index])
                        print('bleuscore:',[bleu_score[i] for i in top3_score_index])
                        print('top3_sq:',top3_sq)
                        print('top3 score:',top3_score)
                else:
                    print('--'*20)
                    print('UNKNOWN')
                    print('hyp:',hyp)
                    print('answer:',answer)
                    print('embscore:',[emb_score[i] for i in top3_score_index])
                    print('bleuscore:',[bleu_score[i] for i in top3_score_index])
                    print('top3_sq:',top3_sq)
                    print('top3 score:',top3_score)


            elif answer in top3_sq:
                top3_true_num+=1
                if top3_sq[0]==answer:
                    top1_true_num+=1
                    if printall:
                        print('--'*20)
                        print('top1 RIGHT')
                        print('hyp:',hyp)
                        print('answer:',answer)
                        print('embscore:',[emb_score[i] for i in top3_score_index])
                        print('bleuscore:',[bleu_score[i] for i in top3_score_index])
                        print('top3_sq:',top3_sq)
                        print('top3 score:',top3_score)
                else:
                    print('--'*20)
                    print('TOP 3 RIGHT')
                    print('hyp:',hyp)
                    print('answer:',answer)
                    print('embscore:',[emb_score[i] for i in top3_score_index])
                    print('bleuscore:',[bleu_score[i] for i in top3_score_index])
                    print('top3_sq:',top3_sq)
                    print('top3 score:',top3_score)
            else:
                print('--'*20)
                print('Not in TOP3')
                print('hyp:',hyp)
                print('answer:',answer)
                print('embscore:',[emb_score[i] for i in top3_score_index])
                print('bleuscore:',[bleu_score[i] for i in top3_score_index])
                print('top3_sq:',top3_sq)
                print('top3 score:',top3_score)


    top1_true=top1_true_num/all_pair
    top3_true=top3_true_num/all_pair
    print()
    print('The Result of Test No.2 :')
    #print('Max right score: %.3f, Min right score: %.3f' %(max(all_top1_value),min(all_top1_value)))
    print('Number of correct in Top 3: %d, Top 3 Accuracy: %.3f %%,' % (top3_true_num,top3_true*100))
    print('Number of correct in Top 1: %d, Top 1 Accuracy: %.3f %%,' % (top1_true_num,top1_true*100))

if __name__=='__main__':

    dataset=os.path.join(os.path.abspath('.'),'data','standard_qa')
    train_dataset_1=os.path.join(os.path.abspath('.'),'data','final_mix_train_data')
    train_dataset_2=os.path.join(os.path.abspath('.'),'data','train_data')
    train_dataset=[train_dataset_1,train_dataset_2]

    test_dataset_1=os.path.join(os.path.abspath('.'),'data','final_mix_test_data')
    test_dataset_2=os.path.join(os.path.abspath('.'),'data','test_data')
    test_dataset_3=os.path.join(os.path.abspath('.'),'data','A_test_known')
    test_dataset_4=os.path.join(os.path.abspath('.'),'data','A_test')
    test_dataset_5=os.path.join(os.path.abspath('.'),'data','B_test')
    test_dataset=[test_dataset_1,test_dataset_2]

    config_path=os.path.join(os.path.abspath('.'),'config')
    output_path=os.path.join(os.path.abspath('.'),'After_training')

    DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE_1=15
    BATCH_SIZE_2=5
    #threshold=0.06
    threshold=0.0395

    if len(sys.argv)>1:
        if sys.argv[1]=='train1':
            model=BertForSegment(config_path,DEVICE).to(DEVICE)
            train_loop_1(train_dataset,test_dataset,model,BATCH_SIZE_1,output_path)

        elif sys.argv[1]=='train2':
            all_config=os.path.join(output_path,'newmodel_aftertraining.pth')
            model=BertForSegment.load_weights(config_path,all_config,DEVICE)
            train_loop_2(train_dataset,test_dataset,model,BATCH_SIZE_2,output_path)

        elif sys.argv[1]=='test1':
            all_config=os.path.join(output_path,'newmodel.pth')
            model=BertForSegment.load_weights(config_path,all_config,DEVICE)
            test_loop_1(test_dataset,model,threshold)

        elif sys.argv[1]=='test2':
            all_config=os.path.join(output_path,'newmodel_aftertraining.pth')
            model=BertForSegment.load_weights(config_path,all_config,DEVICE)
            test_loop_2(test_dataset,model,threshold)

        elif sys.argv[1]=='test3':
            all_config=os.path.join(output_path,'newmodel_aftertraining.pth')
            model=BertForSegment.load_weights(config_path,all_config,DEVICE)
            if len(sys.argv)==2:
                test_loop_3(test_dataset_3,dataset,model,-10)
            else:
                test_loop_3(test_dataset_3,dataset,model,-10,True)

        elif sys.argv[1]=='test4':
            all_config=os.path.join(output_path,'newmodel_aftertraining.pth')
            model=BertForSegment.load_weights(config_path,all_config,DEVICE)
            test_loop_with_bleu(test_dataset_5,dataset,model,threshold)

        elif sys.argv[1]=='final':
            all_config=os.path.join(output_path,'newmodel_aftertraining.pth')
            model=BertForSegment.load_weights(config_path,all_config,DEVICE)
            test_loop_with_bleu(test_dataset_4,dataset,model,threshold)
    else:
        print('No input sentence')
