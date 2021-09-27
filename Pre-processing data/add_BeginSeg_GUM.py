import re
from nltk.tokenize import word_tokenize
import pickle
import numpy as np

import torch
from solver import TrainSolver

from model import PointerNetworks

from iteration_utilities import flatten

# can change the files for f2
# can change the name of new file

f2 = open("GUM_whow_test.txt", "rb").readlines()

for line in f2:
    line=str(line).replace('\\t',' ')
    line=line.replace('\\n','')
    line=line.replace('\\r','')
    if '#' in str(line):
        line=str(line).replace("b'",'')
        line.replace('b"','')
        with open('GUM_whow_3.txt', 'a') as the_file:
            the_file.write(str(line)+'\n')  
        continue
    text=str(line).split(' ')
    text[0].replace("b'",'')
    text[0]=text[0].replace("b'",'')
    text[0]=text[0].replace('b"','')
    if not text[0].isdigit():
         with open('GUM_whow_3.txt', 'a') as the_file:
             the_file.write('\n')
    else:
        text[9]=text[9].replace("'",'')
        text[9]=text[9].replace('"','')
        if 'Discourse' in text[9]:
            with open('GUM_whow_3.txt', 'a') as the_file: # add label for the start of new segment
                the_file.write('\t'.join(text[:-1])+'\t'+'BeginSeg=Yes'+'\n')
        else:
            with open('GUM_whow_3.txt', 'a') as the_file:
                the_file.write('\t'.join(text)+'\n')