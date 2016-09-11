__author__ = 'Daniela, Pat'
__date__ = '03.09.2016'
#Creates models and returns accuracies for index vector input


import math
import numpy as np
import pandas as pd
from keras.layers import Embedding, Convolution1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.models import Sequential
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# raw data and labels of CSC and SAD corpus
d_CSC = pd.read_csv('../data_CSDMC2010_SPAM/CSC_mails.csv')
CSC_labels = d_CSC.iloc[:, 1].as_matrix()

d_SAD = pd.read_csv('../data_SpamAssassin/SAD_mails.csv')
SAD_labels = d_SAD.iloc[:, 1].as_matrix()


# index vectors of CSC and SAD corpus
df_CSC = pd.read_csv('CSC_to_int.csv')
CSC_data_features = df_CSC.iloc[:, :500].as_matrix()

df_SAD = pd.read_csv('SAD_to_int.csv')
SAD_data_features = df_SAD.iloc[:, :500].as_matrix()


print("################### CountVectorizer CSC_corpus DATA: Logistic Regression C=1000000 (10-fold cross validation) ###################")
texts_CSC = list()
for mail_content_CSC in d_CSC.content:
    texts_CSC.append(str(mail_content_CSC))
coVe_CSC = CountVectorizer(ngram_range=(1, 2))

# To restrict feature size to 500/5.000 which makes the model more similar to CNN,
# uncomment (and, for 5,000, change) the following line:
#coVe_CSC = CountVectorizer(ngram_range=(1, 2), vocabulary=None, max_features=500)

CSC_data_features_coVe = coVe_CSC.fit_transform(texts_CSC)

print("labels", len(CSC_labels))
print("data_features", CSC_data_features_coVe.shape)

# create and fit the model
lrm_cv_coVe = LogisticRegression(penalty='l2', C=1000000)
lrm_cv_coVe.fit(CSC_data_features_coVe, CSC_labels)

# report resulting measures
cv_scores_lrm_CSC_coVe = cross_val_score(lrm_cv_coVe, CSC_data_features_coVe, CSC_labels, cv=10)
print("cross-validation scores: ", cv_scores_lrm_CSC_coVe)
cv_mean_lrm_CSC_coVe = np.mean(cv_scores_lrm_CSC_coVe)
print("mean accuracy cv: ", cv_mean_lrm_CSC_coVe)
sterr_lrm_CSC_coVe = np.std(cv_scores_lrm_CSC_coVe) / (math.sqrt(len(cv_scores_lrm_CSC_coVe)))
print("standard error: ", sterr_lrm_CSC_coVe)


print("################### CountVectorizer SAD_corpus DATA: Logistic Regression C=1000000 (10-fold cross validation) ###################")
texts_SAD = list()
for mail_content_SAD in d_SAD.content:
    texts_SAD.append(str(mail_content_SAD))
coVe_SAD = CountVectorizer(ngram_range=(1, 2))

# To restrict the feature size to 500/5.000 which makes the model more similar to CNN,
# uncomment (and, for 5,000, change) the following line:
#coVe_SAD = CountVectorizer(ngram_range=(1, 2), vocabulary=None, max_features=500)

SAD_data_features_coVe = coVe_SAD.fit_transform(texts_SAD)
print("labels", len(SAD_labels))
print("data_features", SAD_data_features_coVe.shape)

# create and fit the model
lrm_SAD_cv_coVe = LogisticRegression(penalty='l2', C=1000000)
lrm_SAD_cv_coVe.fit(SAD_data_features_coVe, SAD_labels)

# report resulting measures
cv_scores_lrm_SAD_coVe = cross_val_score(lrm_SAD_cv_coVe, SAD_data_features_coVe, SAD_labels, cv=10)
print("cross-validation scores: ", cv_scores_lrm_SAD_coVe)
cv_mean_lrm_SAD_coVe = np.mean(cv_scores_lrm_SAD_coVe)
print("mean accuracy cv: ", cv_mean_lrm_SAD_coVe)
sterr_lrm_SAD_coVe = np.std(cv_scores_lrm_SAD_coVe) / (math.sqrt(len(cv_scores_lrm_SAD_coVe)))
print("standard error: ", sterr_lrm_SAD_coVe)


print("################### CSC_corpus DATA: Logistic Regression C=1000000 (10-fold cross validation) ###################")
print("labels", len(CSC_labels))
print("data_features", CSC_data_features.shape)

# create and fit the model
lrm_cv_CSC = LogisticRegression(penalty='l2', C=1000000)
lrm_cv_CSC.fit(CSC_data_features, CSC_labels)

# report resulting measures
cv_scores_lrm_CSC = cross_val_score(lrm_cv_CSC, CSC_data_features, CSC_labels, cv=10)
print("cross-validation scores: ", cv_scores_lrm_CSC)
cv_mean_lrm_CSC = np.mean(cv_scores_lrm_CSC)
print("mean accuracy cv: ", cv_mean_lrm_CSC)
sterr_lrm_CSC = np.std(cv_scores_lrm_CSC) / (math.sqrt(len(cv_scores_lrm_CSC)))
print("standard error: ", sterr_lrm_CSC)


print("################### SAD_corpus DATA: Logistic Regression C=1000000 (10-fold cross validation) ###################")
print("labels", len(SAD_labels))
print("data_features", SAD_data_features.shape)

# create and fit the model
lrm_SAD_cv = LogisticRegression(penalty='l2', C=1000000)
lrm_SAD_cv.fit(SAD_data_features, SAD_labels)

# report resulting measures
cv_scores_lrm_SAD = cross_val_score(lrm_SAD_cv, SAD_data_features, SAD_labels, cv=10)
print("cross-validation scores: ", cv_scores_lrm_SAD)
cv_mean_lrm_SAD = np.mean(cv_scores_lrm_SAD)
print("mean accuracy cv: ", cv_mean_lrm_SAD)
sterr_lrm_SAD = np.std(cv_scores_lrm_SAD) / (math.sqrt(len(cv_scores_lrm_SAD)))
print("standard error: ", sterr_lrm_SAD)


print("################### CSC_corpus DATA: CNN (10-fold cross validation) ###################")
print("labels", len(CSC_labels))
print("data_features", CSC_data_features.shape)

# set variables
max_features_CSC = 5001 # vocabulary size: 5,000 most frequent words
max_len_CSC = CSC_data_features.shape[1] # maximum document/sequence length
embedding_dims_CSC = 100 # vocabulary mapped onto x dimensions
feature_maps_CSC = 25 # number of feature maps for each filter size
filter_size_CSC = 5 # size of applied filter, covering at least bigrams = 2
hidden_dims_CSC = 50
batch_size_CSC = 16
pool_length_CSC = 2

### create and fit the model
kfold_cnn_CSC = StratifiedKFold(y=CSC_labels, n_folds=10, shuffle=True, random_state=True)
cv_scores_cnn_CSC = []
for i, (train_CSC, test_CSC) in enumerate(kfold_cnn_CSC):
    print("Looping through cross-validation (CSC corpus)...")
    cnn_cv_CSC = Sequential()
    cnn_cv_CSC.add(Embedding(max_features_CSC, embedding_dims_CSC, input_length=max_len_CSC, dropout=0.5))
    cnn_cv_CSC.add(Convolution1D(nb_filter=feature_maps_CSC, filter_length=filter_size_CSC, activation='relu'))
    cnn_cv_CSC.add(MaxPooling1D(pool_length=cnn_cv_CSC.output_shape[2]))
    cnn_cv_CSC.add(Flatten())
    cnn_cv_CSC.add(Dense(hidden_dims_CSC, activation='relu'))
    cnn_cv_CSC.add(Dropout(0.2))
    cnn_cv_CSC.add(Dense(1, activation='sigmoid'))
    # compile model
    cnn_cv_CSC.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # fit model
    cnn_cv_CSC.fit(CSC_data_features[train_CSC], CSC_labels[train_CSC], batch_size=batch_size_CSC, nb_epoch=2, verbose=0)
    #, validation_data=(data_features_03[test], labels_03[test]))
    # evaluate model
    cv_scores_CSC = cnn_cv_CSC.evaluate(CSC_data_features[test_CSC], CSC_labels[test_CSC], verbose=0)
    cv_scores_cnn_CSC.append(cv_scores_CSC[1])

# report resulting measures
print("cross-validation scores: ", cv_scores_cnn_CSC)
cv_mean_cnn = np.mean(cv_scores_cnn_CSC)
print("mean accuracy cv: ", cv_mean_cnn)
sterr_cnn = np.std(cv_scores_cnn_CSC) / (math.sqrt(len(cv_scores_cnn_CSC)))
print("standard error: ", sterr_cnn)


print("################### SAD_corpus DATA: CNN (10-fold cross validation) ###################")
print("labels", len(SAD_labels))
print("data_features", SAD_data_features.shape)

# set variables
max_features_SAD = 5001 # vocabulary size: 5,000 most frequent words
max_len_SAD = SAD_data_features.shape[1] # maximum document/sequence length
embedding_dims_SAD = 100 # vocabulary mapped onto x dimensions
feature_maps_SAD = 25 # number of feature maps for each filter size
filter_size_SAD = 5 # size of applied filter, covering at least bigrams = 2
hidden_dims_SAD = 50
batch_size_SAD = 16
pool_length_SAD = 2

### create and fit the model
kfold_cnn_SAD = StratifiedKFold(y=SAD_labels, n_folds=10, shuffle=True, random_state=True)
cv_scores_cnn_SAD = []
for i, (train_SAD, test_SAD) in enumerate(kfold_cnn_SAD):
    print("Looping through cross-validation (SAD corpus)...")
    cnn_cv_SAD = Sequential()
    cnn_cv_SAD.add(Embedding(max_features_SAD, embedding_dims_SAD, input_length=max_len_SAD, dropout=0.5))
    cnn_cv_SAD.add(Convolution1D(nb_filter=feature_maps_SAD, filter_length=filter_size_SAD, activation='relu'))
    cnn_cv_SAD.add(MaxPooling1D(pool_length=cnn_cv_SAD.output_shape[2]))
    cnn_cv_SAD.add(Flatten())
    cnn_cv_SAD.add(Dense(hidden_dims_SAD, activation='relu'))
    cnn_cv_SAD.add(Dropout(0.2))
    cnn_cv_SAD.add(Dense(1, activation='sigmoid'))
    # compile model
    cnn_cv_SAD.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # fit model
    cnn_cv_SAD.fit(SAD_data_features[train_SAD], SAD_labels[train_SAD], batch_size=batch_size_SAD, nb_epoch=2, verbose=0)
    #, validation_data=(data_features_03[test], labels_03[test]))
    # evaluate model
    cv_scores_SAD = cnn_cv_SAD.evaluate(SAD_data_features[test_SAD], SAD_labels[test_SAD], verbose=0)
    cv_scores_cnn_SAD.append(cv_scores_SAD[1])

# report resulting measures
print("cross-validation scores: ", cv_scores_cnn_SAD)
cv_mean_cnn_SAD = np.mean(cv_scores_cnn_SAD)
print("mean accuracy cv: ", cv_mean_cnn_SAD)
sterr_cnn_SAD = np.std(cv_scores_cnn_SAD) / (math.sqrt(len(cv_scores_cnn_SAD)))
print("standard error: ", sterr_cnn_SAD)