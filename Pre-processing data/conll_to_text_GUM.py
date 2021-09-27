# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 04:49:58 2021

@author: Jessica Geng
"""
import re
from nltk.tokenize import word_tokenize
import pickle
import numpy as np

import torch
from solver import TrainSolver

from model import PointerNetworks

from iteration_utilities import flatten

# can change files for f2
# can change the name for new file

f2 = open("GUM_mix_test.txt", "rb").readlines()


seg='0'

with open('GUM_mix_test_without.txt', 'a') as the_file:
    the_file.write('0'+' '+'0') # first line for polarity score
for line in f2:
    print(line)
    if '#' in str(line):
        continue
    # delete separate code
    line=str(line).replace('\\t',' ')
    line=line.replace('\\n','')
    line=line.replace('\\r','')
    text=str(line).split(' ')
    text[0].replace("b'",'')
    text[0]=text[0].replace("b'",'')
    text[0]=text[0].replace('b"','')
    if not text[0].isdigit():
        if seg=='0':
            with open('GUM_mix_test_without.txt', 'a') as the_file:
                the_file.write(' '+'< s >'+'\n') # add label for end of sentence
                the_file.write('\n')
                the_file.write('0'+' '+'0')
        else:
            with open('GUM_mix_test_without.txt', 'a') as the_file:
                the_file.write('\n'+seg+' '+'< s >'+'\n') # change the line and add label for end of sentence
                the_file.write('\n')
                the_file.write('0'+' '+'0')
                seg='0'
            continue
    else:
        text[9]=text[9].replace("'",'')
        text[9]=text[9].replace('"','')
        if text[9]=='BeginSeg=Yes':
        #if 'Discourse' in text[9]:
            if seg!='0':
                with open('GUM_mix_test_without.txt', 'a') as the_file:
                    the_file.write('\n'+seg)
                seg='0'
        seg+=' '+str(text[1])
