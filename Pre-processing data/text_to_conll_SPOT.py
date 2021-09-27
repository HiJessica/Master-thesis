# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 06:55:15 2021

@author: Jessica Geng
"""

import spacy

import re
from nltk.tokenize import word_tokenize
import pickle
import numpy as np

import torch

# can change the f1 file for training, dev, test files
# can change the name of new created files 

nlp = spacy.load("en_core_web_sm")

f1 = open("dev.txt", "r").readlines()

t1=[]
tmp_seg=[]
tmp_sen=''
tmp_index=0
for line in f1:
    text=word_tokenize(line)[1:]
    if len(text)==1:
        continue
    if text==[]:
        continue
    text_sen=' '.join(text)
    text = nlp(text_sen)

    if '< s >' in text_sen: # end of sentence
        text_sen=text_sen.replace('< s >','')
        text = nlp(text_sen)
        tmp_sen+=' '+text_sen
        doc = nlp(tmp_sen[1:])
        tmp_index+=len(text)
        tmp_seg+=[tmp_index]
        tmp_seg+=[0]
        pos=0
        # add features
        for token in doc:
            with open('dev_tag.txt', 'a') as the_file:
                if pos in tmp_seg: # start of new segment
                    the_file.write(str(pos+1)+ "\t"+ token.text+ "\t"+token.lemma_+ "\t"+token.pos_+ "\t"+ token.tag_+ "\t"+'_'+ "\t"+str(token.head.i+1)+ "\t"+token.dep_+ "\t"+'_'+ "\t"+'BeginSeg=Yes'+ "\n")
                else:
                    the_file.write(str(pos+1)+ "\t"+token.text+ "\t"+token.lemma_+ "\t"+token.pos_+ "\t"+ token.tag_+ "\t"+'_'+ "\t"+str(token.head.i+1)+ "\t"+token.dep_+ "\t"+'_'+ "\t"+'_'+ "\n")
            pos+=1
        tmp_sen=''
        tmp_seg=[]   
        tmp_index=0
        with open('dev_tag.txt', 'a') as the_file:
            the_file.write('\n')
    else:
        tmp_sen+=' '+text_sen
        tmp_index+=len(text)
        tmp_seg+=[tmp_index]
        
   



