__author__ = 'Daniela Stier'

# IMPORT STATEMENTS
import csv
import itertools
import numpy as np
import pandas as pd
import math as mt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Convolution1D, Flatten, MaxPooling1D

############################################# APPLICATIONAL PART #############################################

# FOR CSC: merging data vectors, stored in 'CSC_main_vectors.csv'
csv_names = ["01_CSC_vectors/CSC_sent_vector.csv", "01_CSC_vectors/CSC_word_vector.csv", "01_CSC_vectors/CSC_char_vector.csv", "01_CSC_vectors/CSC_punct_vector.csv"]
readers = [csv.reader(open(r, 'r')) for r in csv_names]
writer = csv.writer(open('01_CSC_vectors/CSC_main_vectors.csv', 'w'))
for row in zip(*readers):
    writer.writerow(list(itertools.chain.from_iterable(row)))


# FOR SAD: merging data vectors for test, stored in 'SAD_main_vectors.csv'
csv_names = ["02_SAD_vectors/SAD_sent_vector.csv", "02_SAD_vectors/SAD_word_vector.csv", "02_SAD_vectors/SAD_char_vector.csv", "02_SAD_vectors/SAD_punct_vector.csv"]
readers = [csv.reader(open(r, 'r')) for r in csv_names]
writer = csv.writer(open('02_SAD_vectors/SAD_main_vectors.csv', 'w'))
for row in zip(*readers):
    writer.writerow(list(itertools.chain.from_iterable(row)))


# input data FOR CSC
clean_data = pd.read_csv('01_CSC_vectors/CSC_main_vectors.csv', sep=',', header=1)
clean_data = clean_data.iloc[np.random.permutation(len(clean_data))]
data_features = clean_data.iloc[:, 1:].as_matrix()
labels = clean_data.iloc[:, 0].as_matrix()
labels = np.array(labels)
data_features = np.array(data_features)


# input test data for SAD
clean_test_data = pd.read_csv('02_SAD_vectors/SAD_main_vectors.csv', sep=',', header=1)
clean_test_data = clean_test_data.iloc[np.random.permutation(len(clean_test_data))]
test_data_features = clean_test_data.iloc[:, 1:].as_matrix()
test_labels = clean_test_data.iloc[:, 0].as_matrix()
test_labels = np.array(test_labels)
test_data_features = np.array(test_data_features)


print("################### CSC: Logistic Regression C=1000000 (10-fold cross validation) ###################")
print("labels", len(labels))
print("data_features", data_features.shape)

# create and fit the model
lrm_cv = LogisticRegression(penalty='l2', C=1000000)
lrm_cv.fit(data_features, labels)

# report resulting accuracies
cv_scores_lrm = cross_val_score(lrm_cv, data_features, labels, cv=10)
print("cross-validation scores: ", cv_scores_lrm)
cv_mean_lrm = np.mean(cv_scores_lrm)
print("mean accuracy cv: ", cv_mean_lrm)
sterr_lrm = np.std(cv_scores_lrm)/(mt.sqrt(len(cv_scores_lrm)))
print("standard error: ", sterr_lrm)


print("################### SAD: Logistic Regression C=1000000 (10-fold cross validation) ###################")
print("labels", len(test_labels))
print("data_features", test_data_features.shape)

# create and fit the model
lrm_test_cv = LogisticRegression(penalty='l2', C=1000000)
lrm_test_cv.fit(test_data_features, test_labels)

# report resulting accuracies
cv_scores_lrm_test = cross_val_score(lrm_test_cv, test_data_features, test_labels, cv=10)
print("cross-validation scores: ", cv_scores_lrm_test)
cv_mean_lrm_test = np.mean(cv_scores_lrm_test)
print("mean accuracy cv: ", cv_mean_lrm_test)
sterr_lrm_test = np.std(cv_scores_lrm_test)/(mt.sqrt(len(cv_scores_lrm_test)))
print("standard error: ", sterr_lrm_test)


print("################### CSC: CNN (10-fold cross validation) ###################")
print("labels", len(labels))
print("data_features", data_features.shape)

# set variables
max_features = int(clean_data.values.max())+1 # vocabulary size
max_len = data_features.shape[1] # maximum document/sequence length
embedding_dims = 100 # vocabulary mapped onto x dimensions
feature_maps = 25 # number of feature maps for each filter size
filter_size = 5 # size of applied filter, covering at least bigrams = 2
hidden_dims = 50
batch_size = 16

### create and fit the model
kfold_cnn = StratifiedKFold(y=labels, n_folds=10, shuffle=True, random_state=True)
cv_scores_cnn = []
for i, (train, test) in enumerate(kfold_cnn):
    cnn_cv = Sequential()
    cnn_cv.add(Embedding(max_features, embedding_dims, input_length=max_len, dropout=0.5))
    cnn_cv.add(Convolution1D(nb_filter=feature_maps, filter_length=filter_size, activation='relu'))
    cnn_cv.add(MaxPooling1D(pool_length=cnn_cv.output_shape[2]))
    cnn_cv.add(Flatten())
    cnn_cv.add(Dense(hidden_dims, activation='relu'))
    cnn_cv.add(Dropout(0.2))
    cnn_cv.add(Dense(1, activation='sigmoid'))
    # compile model
    cnn_cv.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # fit model
    cnn_cv.fit(data_features[train], labels[train], batch_size=batch_size, nb_epoch=2, verbose=0)#, validation_data=(data_features_03[test], labels_03[test]))
    # evaluate model
    cv_scores = cnn_cv.evaluate(data_features[test], labels[test], verbose=0)
    cv_scores_cnn.append(cv_scores[1] * 100)

# report resulting accuracies
print("cross-validation scores: ", cv_scores_cnn)
cv_mean_cnn = np.mean(cv_scores_cnn)
print("mean accuracy cv: ", cv_mean_cnn)
sterr_cnn = np.std(cv_scores_cnn)/(mt.sqrt(len(cv_scores_cnn)))
print("standard error: ", sterr_cnn)


print("################### SAD: CNN (10-fold cross validation) ###################")
print("labels", len(test_labels))
print("data_features", test_data_features.shape)

# set variables
max_features_test = int(clean_test_data.values.max())+1 # vocabulary size
max_len_test = test_data_features.shape[1] # maximum document/sequence length
embedding_dims_test = 100 # vocabulary mapped onto x dimensions
feature_maps_test = 25 # number of feature maps for each filter size
filter_size_test = 5 # size of applied filter, covering at least bigrams = 2
hidden_dims_test = 50
batch_size_test = 16

### create and fit the model
kfold_cnn_test = StratifiedKFold(y=test_labels, n_folds=10, shuffle=True, random_state=True)
cv_scores_cnn_test = []
for i, (train_test, test_test) in enumerate(kfold_cnn_test):
    cnn_cv_test = Sequential()
    cnn_cv_test.add(Embedding(max_features_test, embedding_dims_test, input_length=max_len_test, dropout=0.5))
    cnn_cv_test.add(Convolution1D(nb_filter=feature_maps_test, filter_length=filter_size_test, activation='relu'))
    cnn_cv_test.add(MaxPooling1D(pool_length=cnn_cv_test.output_shape[2]))
    cnn_cv_test.add(Flatten())
    cnn_cv_test.add(Dense(hidden_dims_test, activation='relu'))
    cnn_cv_test.add(Dropout(0.2))
    cnn_cv_test.add(Dense(1, activation='sigmoid'))
    # compile model
    cnn_cv_test.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # fit model
    cnn_cv_test.fit(test_data_features[train_test], test_labels[train_test], batch_size=batch_size_test, nb_epoch=2, verbose=0)#, validation_data=(data_features_03[test], labels_03[test]))
    # evaluate model
    cv_scores_test = cnn_cv_test.evaluate(test_data_features[test_test], test_labels[test_test], verbose=0)
    cv_scores_cnn_test.append(cv_scores_test[1] * 100)

# report resulting accuracies
print("cross-validation scores: ", cv_scores_cnn_test)
cv_mean_cnn_test = np.mean(cv_scores_cnn_test)
print("mean accuracy cv: ", cv_mean_cnn_test)
sterr_cnn_test = np.std(cv_scores_cnn_test)/(mt.sqrt(len(cv_scores_cnn_test)))
print("standard error: ", sterr_cnn_test)