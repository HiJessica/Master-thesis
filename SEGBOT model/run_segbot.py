# import tools
import re
from nltk.tokenize import word_tokenize
import pickle
import numpy as np

import torch
from solver import TrainSolver

from model import PointerNetworks

from iteration_utilities import flatten

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"RE_DIGITS":1,"UNKNOWN":2,"PADDING":0} 
        self.word2count = {"RE_DIGITS":1,"UNKNOWN":1,"PADDING":1}
        self.index2word = {0: "PADDING", 1: "RE_DIGITS", 2: "UNKNOWN"}
        self.n_words = 3  # Count SOS and EOS. start from 3 to add new words' number

    def addSentence(self, sentence): # remove \n,\r and split a sentenceto words
        for word in sentence.strip('\n').strip('\r').split(' '):
            self.addWord(word)

    def addWord(self, word): # add new words to dictionary
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# tokenize input string
def mytokenizer(inS,all_dict): # ins=input string; all_dict=dictionary

    repDig = re.sub(r'\d*[\d,]*\d+', 'RE_DIGITS', inS) # replace setence digits and comma besides digits with RE_DIGITS in ins
    toked = word_tokenize(repDig) # NLTK tokenize repDig 
    or_toked = word_tokenize(inS) # NLTK tokenize ins(original one)
    re_unk_list = []
    ori_list = []
    
    for (i,t) in enumerate(toked): # index, word
        # dictionary or unknwon; original split token
        if t not in all_dict and t not in ['RE_DIGITS']:
            re_unk_list.append('UNKNOWN')
            ori_list.append(or_toked[i])
        else:
            re_unk_list.append(t)
            ori_list.append(or_toked[i])
    labey_edus = [0]*len(re_unk_list) # [0,0,...,0]
    labey_edus[-1] = 1  # [0,0,0...,0,1]


    return ori_list,re_unk_list,labey_edus #original split words, add unkown words, [0,0,...,1]


# map words in X to corresponding dictionary word/unkown and convert to array
def get_mapping(X,Y,D): # X,Y,D=word2index vocab

    X_map = []
    for w in X:
        if w in D: 
            X_map.append(D[w])
        else:
            X_map.append(D['UNKNOWN'])

    X_map = np.array([X_map])
    Y_map = np.array([Y])


    return X_map,Y_map


# pre-processed input data
def preprocessing(file):
    all_voc = r'all_vocabulary.pickle' # insert all_vocabulary.pickle
    voca = pickle.load(open(all_voc, 'rb')) 
    voca_dict = voca.word2index # word2index
    t1=[]
    arr_all=[[]]
    tmp_seg=[]
    tmp_sen=''
    for line in file:
        text=word_tokenize(line)[1:]
        if text==[]: # skip blank lines
            continue
        if text[0].isdigit(): # skip id lines
            continue        
        text=' '.join(text)
        if '< s >' in text: # end of sentence
            text=text.replace('< s >','')
            tmp_sen+=' '+text
            ori_X, X, Y = mytokenizer(tmp_sen, voca_dict)
            arr_all[-1]+=[0]*(len(ori_X)-1)+[1] 
            for i in tmp_seg:
                arr_all[-1][i]=1
            X_in, Y_in = get_mapping(X, Y, voca_dict)
            t1.append(list(flatten(X_in.tolist())))
            arr_all.append([])
            tmp_sen='' # restart to memorize sequence
            tmp_seg=[]
        else: 
            tmp_sen+=' '+text
            ori_X, X, Y=mytokenizer(tmp_sen, voca_dict)
            tmp_seg+=[len(ori_X)-1] # memorize positions for end of segments
    arr_all.remove([])
    t1=np.array(t1)
    arr_all=np.array(arr_all)
    return t1,arr_all


def main_input_output():
      
# model.py
    mymodel = PointerNetworks(voca_size =2, voc_embeddings=np.ndarray(shape=(2,500), dtype=float),word_dim=500, hidden_dim=10,is_bi_encoder_rnn=True,rnn_type='GRU',rnn_layers=3,
                 dropout_prob=0.5,use_cuda=False,finedtuning=True,isbanor=True)

    mymodel = torch.load(r'trained_model.torchsave', map_location=lambda storage, loc: storage) # load torchsave document onto CPU, using a function
    # can change parameters in saved model, e.g mymodel.dropout_prob=0.4
 
    mymodel.use_cuda = False

    mymodel.eval()
    
# solver.py
    # insert training, validation and test set
    # text file could changes to other dataset file
    f1 = open("GUM_mix_train_without.txt", "r").readlines() 
    train_x,train_y=preprocessing(f1)
    f2 = open("GUM_mix_dev_without.txt", "r").readlines() 
    dev_x,dev_y=preprocessing(f2)
    f3 = open("GUM_mix_test_without.txt", "r").readlines() 
    test_x,test_y=preprocessing(f3)
    
    mysolver = TrainSolver(mymodel,train_x,train_y,train_x,train_y, save_path=r'C:\Users\Jessica Geng\Desktop\segbot',
                           batch_size=200, eval_size=1, epoch=20, lr=1e-5, lr_decay_epoch=1, weight_decay=1e-4,
                           use_cuda=False)

    mysolver.train()
    # get metrics
    test_batch_ave_loss, test_pre, test_rec, test_f1, visdata = mysolver.check_accuracy(test_x,test_y) 

    return test_batch_ave_loss, test_pre, test_rec, test_f1


if __name__ == '__main__':
    [test_batch_ave_loss, test_pre, test_rec, test_f1]=main_input_output()
        
    print(test_batch_ave_loss, test_pre, test_rec, test_f1)
