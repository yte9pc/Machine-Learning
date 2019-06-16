#!/usr/bin/python
from __future__ import division

import sys
import os
import numpy as np
import pandas as pd
#import nltk
from sklearn.naive_bayes import MultinomialNB
#from nltk.tokenize import word_tokenize
#from nltk.stem import PorterStemmer
from collections import OrderedDict

###############################################################################
# Create Dictionary
vocabulary1 = ('love', 'loving', 'loved', 'loves')
vocabulary2 = ('wonderful',)
vocabulary3 = ('best' , 'bests', 'bested', 'besting')
vocabulary4 = ('great', 'greater', 'greatest', 'greats')
vocabulary5 = ('superb',)
vocabulary6 = ('still', 'stiller', 'stillest', 'stills', 'stilled', 'stilling')
vocabulary7 = ('beautiful',)
vocabulary8 = ('bad', 'badder', 'baddest')
vocabulary9 = ('worst', 'worsts', 'worsted', 'worsting')
vocabulary10 = ('stupid', 'stupider', 'stupidest', 'stupids')
vocabulary11 = ('waste', 'wastes', 'wasted', 'wasting')
vocabulary12 = ('boring', 'bored')
vocabulary13 = ('?',)
vocabulary14 = ('!',) 
vocabulary15 = ('confuse', 'confused', 'confusing', 'confuses')
vocabulary16 = ('like', 'liking', 'likes', 'liked')
vocabulary17 = ('hate', 'hated', 'hates')
vocabulary18 = ('good',)
vocabulary19 = ('terrible',)
vocabulary20 = ('fun', 'funnier', 'funnest')
vocabulary21 = ('happy', 'happier')
vocabulary22 = ('poor', 'poorly', 'poorest')
vocabulary23 = ('avoid', 'mistakes')
vocabulary24 = ('enjoy', 'enjoyed', 'enjoys', 'enjoying', 'happy')
vocabularyUNK = ('love', 'loving', 'loved', 'loves', 'wonderful', 'best' , 'bests', 
                'bested', 'besting','great', 'greater', 'greatest', 'greats',
                'superb', 'still', 'stiller', 'stillest', 'stills', 'stilled', 'stilling',
                'beautiful', 'bad', 'badder', 'baddest', 'worst', 'worsts', 'worsted', 
                'worsting', 'stupid', 'stupider', 'stupidest', 'stupids', 'waste', 'wastes', 
                'wasted', 'wasting', 'boring', '?', '!', 'confuse', 'confused', 'confusing', 
                'confuses', 'like', 'liking', 'likes', 'liked', 'hate', 'hated', 'hates',
                'good','terrible', 'fun', 'funnier', 'funnest', 
                'poor', 'poorly', 'poorest', 'avoid', 'mistake', 'enjoy', 'enjoyed', 'enjoys', 
                'enjoying', 'happy', 'happier')
vocabularyALL = [vocabulary1, vocabulary2, vocabulary3,vocabulary4, vocabulary5,
                 vocabulary6, vocabulary7, vocabulary8,  vocabulary9, vocabulary10, 
                 vocabulary11, vocabulary12, vocabulary13, vocabulary14, vocabulary15,
                 vocabulary16, vocabulary17, vocabulary18, vocabulary19, vocabulary20,
                 vocabulary21, vocabulary22, vocabulary23, vocabulary24, vocabularyUNK]

predefined = ['love', 'wonderful', 'best', 'great', 'superb', 'still', 'beautiful', 
         'bad', 'worst', 'stupid', 'waste', 'boring', '?', '!', 'confuse', 'likes',
         'hate', 'good', 'terrible', 'fun', 'happy', 'poor', 'avoid', 'enjoy', 'UNK']

vocabulary = OrderedDict()

for i in range(len(predefined)):
    vocabulary[predefined[i]] = vocabularyALL[i]
#----------------------------------------------------------------------------#

def transfer(fileDj, vocabulary):
    tokens = open(fileDj).read().split()

    BOWDj = np.zeros(len(vocabulary))
    
    for i in tokens:
        for j in vocabulary.keys():
            vocab = vocabulary.get(j)
            
            if i in vocab and j != 'UNK':
                BOWDj[vocabulary.keys().index(j)] += 1
            elif j == 'UNK' and i not in vocabulary.get(j):
                BOWDj[-1] += 1

    return BOWDj


def loadData(Path):
    Xtrain =[]
    ytrain = []
    Xtest = [] 
    ytest = []

        
    for filename in os.listdir(Path + '/training_set/neg/'):
        BOWDj = transfer(Path + '/training_set/neg/' + filename, vocabulary)
        Xtrain.append(BOWDj)
        ytrain.append(0)
        
    for filename in os.listdir(Path + '/training_set/pos/'):
        BOWDj = transfer(Path + '/training_set/pos/' + filename, vocabulary)
        Xtrain.append(BOWDj)
        ytrain.append(1)
    
    for filename in os.listdir(Path + '/test_set/neg/'):
        BOWDj = transfer(Path + '/test_set/neg/' + filename, vocabulary)
        Xtest.append(BOWDj)
        ytest.append(0)
        
    for filename in os.listdir(Path + '/test_set/pos/'):
        BOWDj = transfer(Path + '/test_set/pos/' + filename, vocabulary)
        Xtest.append(BOWDj)
        ytest.append(1)
        
    Xtrain =  np.asmatrix(np.asarray(Xtrain))
    ytrain = np.asarray(ytrain)
    
    Xtest = np.asmatrix(np.asarray(Xtest))
    ytest = np.asarray(ytest) 
    
    return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):  
    thetaPos = []
    thetaNeg = []
    
    train = pd.DataFrame(Xtrain)
    train['Class'] = ytrain
    
    pos = train[train['Class'] == 1]
    neg = train[train['Class'] == 0]
    
    p_total_word = pos.groupby('Class').sum().sum(axis=1).values[0]
    n_total_word = neg.groupby('Class').sum().sum(axis=1).values[0]
    
    alpha = 1
    for i in range(len(vocabulary)):
        p_num_occurrences = pos[i].sum(axis=0)
        n_num_occurrences = neg[i].sum(axis=0)
                
        thetaPos.append((p_num_occurrences + alpha)/(p_total_word + alpha * (len(vocabulary))))
        thetaNeg.append((n_num_occurrences + alpha)/(n_total_word + alpha * (len(vocabulary))))

    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg):
    yPredict = []
    
    Xtest = pd.DataFrame(Xtest)
    
    for i in range(len(Xtest)):
        row = Xtest.iloc[i]

        pos = np.log(np.power(np.transpose(np.asarray(thetaPos)), row)).sum(axis=0)
        neg = np.log(np.power(np.transpose(np.asarray(thetaNeg)), row)).sum(axis=0)
        
        if pos > neg:
            yPredict.append(1)
        else:
            yPredict.append(0)
            
    Accuracy = float(np.sum(ytest == yPredict))/len(ytest)
    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    clf = MultinomialNB()
    clf.fit(Xtrain, ytrain)
    yPredict = clf.predict(Xtest)
   
    Accuracy = float(np.sum(ytest == yPredict))/len(ytest)

    return Accuracy



#def naiveBayesMulFeature_testDirectOne(path,thetaPos, thetaNeg):
#   return yPredict


def naiveBayesMulFeature_testDirect(path,thetaPos, thetaNeg):
    Xtest = [] 
    ytest = []
    
    for filename in os.listdir(path + '/neg/'):
        BOWDj = transfer(path + '/neg/' + filename, vocabulary)
        Xtest.append(BOWDj)
        ytest.append(0)
        
    for filename in os.listdir(path + '/pos/'):
        BOWDj = transfer(path + '/pos/' + filename, vocabulary)
        Xtest.append(BOWDj)
        ytest.append(1)
     
    Xtest = np.asmatrix(np.asarray(Xtest))
    ytest = np.asarray(ytest) 
    
    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)

    return yPredict, Accuracy

def naiveBayesBernFeature_train(Xtrain, ytrain):
    thetaPosTrue = []
    thetaNegTrue = []
    
    train = pd.DataFrame(Xtrain)
    train['Class'] = ytrain
    
    pos = train[train['Class'] == 1]
    neg = train[train['Class'] == 0]
    
    count_pos = pos.count().values[0]
    count_neg = neg.count().values[0]

    for i in range(len(vocabulary)):
        word_in_pos = pos.where(pos[i] > 0).count().values[0]
        word_in_neg = neg.where(neg[i] > 0).count().values[0]
        
        thetaPosTrue.append((word_in_pos + 1)/(count_pos + 2))
        thetaNegTrue.append((word_in_neg + 1)/(count_neg +2))

    return thetaPosTrue, thetaNegTrue

    
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []
    
    Xtest = pd.DataFrame(Xtest)
    
    for i in range(len(Xtest)):
        row = Xtest.iloc[i]
        pos_value = 1/2
        neg_value = 1/2
        for j in range(len(row)):
            if row[j] > 0:
                pos_value *= thetaPosTrue[j]
                neg_value *= thetaNegTrue[j]
            else:
                pos_value *= (1-thetaPosTrue[j])
                neg_value *= (1-thetaNegTrue[j])
            
            if j == 14:
                if pos_value > neg_value:
                    yPredict.append(1)
                else:
                    yPredict.append(0)
    
    Accuracy = float(np.sum(ytest == yPredict)/len(ytest))
    
    return yPredict, Accuracy


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python naiveBayes.py dataSetPath testSetPath"
        sys.exit()

    print "--------------------"
    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]
    textDataSetsDirectoryFullPath = 'data_sets/'
    testFileDirectoryFullPath = 'data_sets/test_set/'

    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)

    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print "thetaPos =", thetaPos
    print "thetaNeg =", thetaNeg
    print "--------------------"

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print "MNBC classification accuracy =", Accuracy

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print "Sklearn MultinomialNB accuracy =", Accuracy_sk

    yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg)
    print "Directly MNBC tesing accuracy =", Accuracy
    print "--------------------"

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print "thetaPosTrue =", thetaPosTrue
    print "thetaNegTrue =", thetaNegTrue
    print "--------------------"

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print "BNBC classification accuracy =", Accuracy
    print "--------------------"