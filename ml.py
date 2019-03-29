import string
import pandas as pd
import nltk
# nltk.download('stopwords')
import random
import argparse
from nltk.corpus import stopwords
import numpy as np
punctuation = string.punctuation
    
parser = argparse.ArgumentParser(description='ndsc')
import argparse
parser.add_argument('--translated', action='store_true', help='translated')
parser.add_argument('--model', type=str, default='naivebayes',choices=['naivebayes', 'dtree'], help='choice of model')
opt = parser.parse_args()
import pickle

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

MANUAL_FEATURES = ['CD', 'CDA']

questions = pd.read_csv('data/big_train.csv', encoding='utf-8')
original = pd.read_csv('data/msf_baby_bonus.csv',header=None,encoding='utf-8').iloc[:,1]
test = pd.read_csv('data/predicted_transcipts_sorted.csv',encoding='utf-8')
print(test)
experiment = 'original_'+ opt.model

print(experiment)
question_titles = [x[1] for x in list(questions.loc[:,'question'].items())]
question_titles = [nltk.word_tokenize(x) for x in question_titles]


original_titles = [x[1] for x in list(original.iloc[:].items())]
original_titles = [nltk.word_tokenize(x) for x in original_titles]


test_titles = [x[1] for x in list(test.loc[:,'MessageText'].items())]
test_titles = [nltk.word_tokenize(x) for x in test_titles]

question_titles += original_titles
question_words = []
for title in question_titles[43012:]:
    question_words += title
question_cat = [x[1] for x in list(questions.loc[:,'cls'].items())]
question_cat += [1 for x in range(294)]
assert(len(question_cat) == len(question_titles))
# question_data = [(question_titles[i], question_cat[i]) for i in range(len(question_titles))]
# random.shuffle(question_data)
all_words = nltk.FreqDist(w.lower() for w in question_words)
print('number of positive classes:', question_cat.count(1)/len(question_cat))
word_features = list(all_words.most_common())[:200]
word_features = [x for x in word_features if x[0] not in stopwords.words('english')]
word_features = [x for x in word_features if x[0] not in punctuation]
word_features = [x for x in word_features if len(x[0]) > 1]
word_features_only = [x[0] for x in word_features]
word_features_only += MANUAL_FEATURES
print(word_features_only)
print('number of word features:', len(word_features_only))

pred = np.array([not set(x).isdisjoint(set(word_features_only)) for x in question_titles])
question_cat = np.array(question_cat)==1
unique, counts = np.unique( pred == question_cat,return_counts=True)
pred = dict(zip(unique,counts))


# print(pred)
print('prediction accuracy:', pred[False]/(pred[False]+pred[True]))

print('\n\n')
test_pred = np.array([not set(x).isdisjoint(set(word_features_only)) for x in test_titles])
print(test_pred[:16211].tolist().count(True))
test['string_matching'] = test_pred
test.to_csv('data/predicted_transcript_sorted_matched.csv',index=None)
# featuresets = [(document_features(d),c) for (d,c) in question_data]
# num_samples = len(featuresets)
# train_set, test_set = featuresets[:int(0.8*num_samples)], featuresets[int(0.8*num_samples):]
# if opt.model == 'naivebayes':
#     print('training naive bayes')
#     classifier = nltk.NaiveBayesClassifier.train(train_set)
# else:
#     print('training decision tree')
#     classifier = nltk.DecisionTreeClassifier.train(train_set)
# with open('data/'+experiment+ '.pkl', 'wb') as f:
#     pickle.dump(classifier,f)  
# print(nltk.classify.accuracy(classifier, test_set))
# 
# classifier.show_most_informative_features(5)