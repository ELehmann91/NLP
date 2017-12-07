# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:50:04 2017

@author: cb2lehk
"""


import re
from collections import Counter
import pandas as pd
import pickle

#kuko = pd.read_csv('data/kuko_01.csv', sep=';',header = 0, encoding='mbcs')
#kuko = ''.join(kuko['CBKTNOTZ'])


def words(text): return re.findall(r'\w+', text.lower())

path ='P:/Zentral_Einheiten/ZPK/2_Projektdaten/DataBaseMarketing/Data Science/08 - Textmining/LISTEN/'
WORDS = pickle.load(open(path+'wiki_woerterbuch.p', 'rb'))

words = []
fake_counts = []
with open(path+'fachbegriffe.txt') as fin:
    for line in fin:
        try: 
            k = line.split()[0]
            words.append(k)
            fake_counts.append(9999)
        except IndexError: pass
dict_fach = dict(zip(words,fake_counts))

WORDS.update(dict_fach)

#if False:
#    WORDS = Counter(words(kuko))
#    with open('woerterbuch', 'wb') as outputfile:
#        pickle.dump(WORDS, outputfile)
      
def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    try: p = WORDS[word] / N
    except KeyError: p = 1
    return p

def correction(word): 
    "Most probable spelling correction for word."
    if word in WORDS: pass
    best = max(candidates(word), key=P)
    #for i in range(len(word)-5):
    #    i = i + 3
    #    if word[:i].lower() in WORDS: 
    #        if word[i:].lower() in WORDS:
    #            best = word[:i] + ' ' + word[i:]   
    return best

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or {word})

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyzäöüß'
    aeoeue     = {'ü':'ue','ö':'oe','ä':'ae','ß':'ss'}
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    umlaut     = [word.replace(aeoeue[c],c) for c in aeoeue.keys()] 
    return set(deletes + transposes + replaces + inserts + umlaut)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
    

#with open('woerterbuch', 'rb') as inputfile:
#     print(pickle.load(inputfile))