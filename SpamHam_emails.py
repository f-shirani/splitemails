# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:31:58 2019
@author: shirani

"""
import os, os.path
import numpy as np
import re
import itertools

from numpy.lib.tests.test_format import dtype
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn import svm
import timeit
from sklearn import metrics

# seprate the spam and ham emails


def make_dictionary(stopword, email):
    dictionary = []
    with open(email, errors='ignore') as h:
        email = h.read()
    with open(stopword) as s:
        stopwords = s.readlines()
    for line in email.splitlines():
        for word in line.split():
            word = re.sub(r'[^a-zA-Z]', "", word)
            if word not in stopwords:
                dictionary.append(word)
    return dictionary


# the number of ham_emails
path, dirs, files = next(os.walk("G:/python-practice/emails/enron1/ham"))
ham_count = len(files)
# list of ham emails
ham_emails = [os.path.join("G:/python-practice/emails/enron1/ham", f) for f in
              os.listdir("G:/python-practice/emails/enron1/ham")]
# the number of spam_emails
path, dirs, files = next(os.walk("G:/python-practice/emails/enron1/spam"))
spam_count = len(files)
# list of spam emails
spam_emails = [os.path.join("G:/python-practice/emails/enron1/spam", f) for f in
               os.listdir("G:/python-practice/emails/enron1/spam")]
# 60% data for training and 40% for testing
nspam_train = 900
nham_train = 2200
# counter
nham_test = 1
nspam_test = 1
#######################################
# train
dictionary = []
ham_train = []
spam_train = []
spam_test = []
ham_test = []
total = []
uniqueDic = []
stopwords_txt = "G:\python-practice\emails\stop-word-list.txt"
# split train and test of ham emails
for hams in range(len(ham_emails)):
    if hams <= nham_train - 1:
        # Delete stopwords from ham email && Create dictionaries
        dictionary = make_dictionary(stopwords_txt, ham_emails[hams])
        ham_train.append(ham_emails[hams])
        total.append(list(set(dictionary)))
    else:
        ham_test.append(ham_emails[hams])
        nham_test = nham_test + 1

# split train and test of spam emails
for spams in range(len(spam_emails)):
    if spams <= nspam_train - 1:
        # Delete stopwords from ham email && Create dictionaries
        dictionary = make_dictionary(stopwords_txt, spam_emails[spams])
        spam_train.append(spam_emails[spams])
        total.append(list(set(dictionary)))

    else:
        spam_test.append(spam_emails[spams])
        nspam_test = nspam_test + 1

uniqueDic = [j for i in total for j in i]  # merge all sublists to a list
uniqueDic = list(set(uniqueDic))  # remove duplicate items from list


def features(uniqueDic, emailDir):
    docID = 0
    features = (len(emailDir), len(uniqueDic))
    features_matrix = np.zeros(features, dtype=int)
    for fil in emailDir:
        with open(fil, errors='ignore') as fi:
            h = fi.read()
        for line in h.splitlines():
            word_split = line.split()
            for word in line.split():
               if word in uniqueDic:
                   wordId = uniqueDic.index(word)
                   features_matrix[docID,wordId] = word_split.count(word)
        docID = docID + 1
    return features_matrix


# count the number of dictionary words in ham-train

cham_train = features(uniqueDic, ham_train)
cspam_train = features(uniqueDic, spam_train)
train_label = np.zeros(3100,dtype=int)
train_label[2200:3099] = 1

train_feature = np.concatenate((cham_train, cspam_train), 0)

model1 = LinearSVC()
model1.fit(train_feature, train_label)

test_label = []
cham_test = []
cspam_test = []
# count the number of words dictionary in ham_test mail

cham_test = features(uniqueDic, ham_test)
cspam_test = features(uniqueDic, spam_test)
test_label = np.zeros(2072,dtype=int)
test_label[1471:2072] = 1
test_feature = np.concatenate((cham_test, cspam_test), 0)

y_pred = model1.predict(test_feature)
# calculate the accuracy
print('accuracy: ', metrics.accuracy_score(test_label,y_pred))

# show the time of execution
start = timeit.default_timer()
stop = timeit.default_timer()
print('Time: ', (stop - start))
