__author__ = 'Daniela Stier'

### IMPORT STATEMENTS
import string
import pandas as pd
import numpy as np
from collections import defaultdict

# read preprocessed data, store in lists (clean_data)
d = pd.read_csv('../data_CSDMC2010_SPAM/CSC_mails.csv')
#d = pd.read_csv('../data_SpamAssassin/SAD_mails.csv')
labels = d.iloc[:, 1].as_matrix()

reader = open("CSC.txt", 'r')
#reader = open("SAD.txt", 'r')
clean_data = list()
for line in reader.readlines():
    line_split = line.split("\t")
    sentence = list()
    for sent in line_split:
        sent_split = sent.split(", ")
        temp = list()
        for tagged_word in sent_split:
            if tagged_word.startswith("['"):
                temp.append(tagged_word[2:-1])
            elif tagged_word.endswith("']"):
                temp.append(tagged_word[1:-2])
            else:
                temp.append(tagged_word[1:-1])
        if len(temp) > 1:
            sentence.append(temp)
    clean_data.append(sentence)

# clean data excluding punctuation marks
puncts = string.punctuation
clean_data = [[[token for token in sent if not token[0:token.index("/")].lower() in puncts] for sent in text] for text in clean_data]

# list of words only (excluding pos-tags)
pure_words = [[[word[0:word.index("/")] for word in sent] for sent in text] for text in clean_data]

# number of words
num_words = [[len(sent) for sent in text] for text in pure_words]
num_words_summed = [sum(sent) for sent in num_words]

# number of characters
num_chars = [[[len(word) for word in sent] for sent in text] for text in pure_words]
num_chars_summed_sent = [[sum(sent) for sent in text] for text in num_chars]
num_chars_summed = [sum(text) for text in num_chars_summed_sent]


######### MEASURES CONCERNING SENTENCES ##############################

# number of sentences, output: int(number)
num_sentences = [len(text) for text in clean_data]

# average sentence length (according to the whole text, based on words and characters), output: float(number)
# dividing the total number of words|chars in a text by the total number of sentences
av_sentence_length_words = list()
for i, j in zip(num_words_summed, num_sentences):
    if j > 0:
        av_sentence_length_words.append(i/j)
    else:
        av_sentence_length_words.append(0)

av_sentence_length_chars = list()
for i, j in zip(num_chars_summed, num_sentences):
    if j > 0:
        av_sentence_length_chars.append(i/j)
    else:
        av_sentence_length_chars.append(0)

# n-word sentences distribution (according to the whole text, based on words), output: float(number)
# rel. freq of each sentence-length: dividing total number of sentences of that length by total number of sentences
max_num_word = [np.mean(sent) for sent in num_words]
counter = defaultdict(int)
c = 0
for i in range(int(max_num_word[c])-15, int(max_num_word[c])+15):
    counter[i] = [text.count(i) for text in num_words]
    c += 1

sentence_length_dist_words = list()
for value in counter.values():
    temp = list()
    for i in range(len(num_sentences)):
        if num_sentences[i] > 0:
            temp.append(value[i]/num_sentences[i])
        else:
            temp.append(0)
    sentence_length_dist_words.append(temp)

# n-character sentences distribution (according to the whole text, based on characters)), output: float(number)
# rel. freq of each sentence-length: dividing total number of sentences of that length by total number of sentences
max_num_char = [np.mean(sent) for sent in num_chars_summed_sent]
counter = defaultdict(int)
c = 0
for i in range(int(max_num_char[c])-60, int(max_num_char[c])+60):
    counter[i] = [text.count(i) for text in num_chars_summed_sent]
    c += 1

sentence_length_dist_chars = list()
for value in counter.values():
    temp = list()
    for i in range(len(num_sentences)):
        if num_sentences[i] > 0:
            temp.append(value[i]/num_sentences[i])
        else:
            temp.append(0)
    sentence_length_dist_chars.append(temp)


# store vectorized data in 'test_data_sentence_vector.csv' file, including labels at first position
data_matrix = pd.concat([pd.DataFrame(labels), pd.DataFrame(num_sentences), pd.DataFrame(av_sentence_length_words), pd.DataFrame(av_sentence_length_chars), pd.DataFrame(sentence_length_dist_words).transpose(), pd.DataFrame(sentence_length_dist_chars).transpose()], axis=1)
data_matrix.to_csv('01_CSC_vectors/CSC_sent_vector.csv', index=False, delimiter=',')
#data_matrix.to_csv('02_SAD_vectors/SAD_sent_vector.csv', index=False, delimiter=',')