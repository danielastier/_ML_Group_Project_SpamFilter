__author__ = 'Daniela Stier'

### IMPORT STATEMENTS
import pandas as pd
from nltk.tokenize import TweetTokenizer, sent_tokenize
from nltk import pos_tag


############### IMPLEMENTATION OF HELPER METHODS ###############

### return individual sentences of an input text
def extract_sentences(input_text):
    sentences = sent_tokenize(input_text)
    return sentences

### tokenize input text - nltk.TweetTokenizer does not split contractions
def tokenize_sentences(input_text):
    sent_tokens = [TweetTokenizer().tokenize(sent) for sent in input_text]
    return sent_tokens

### add pos-information to sentences of an input text - nltk as reliable
def pos_tag_sentences(input_text):
    sent_tagged = [pos_tag(sent) for sent in input_text]
    return sent_tagged

### change pos-notation
def change_pos_notation(input_text):
    sent_tagged = [[str(word + "/" + tag) for (word, tag) in sent] for sent in input_text]
    return sent_tagged





############### PREPROCESSING ###############
d = pd.read_csv('../data_CSDMC2010_SPAM/CSC_mails.csv')
#d = pd.read_csv('../data_SpamAssassin/SAD_mails.csv')
data = d.iloc[:, 2].as_matrix()
labels = d.iloc[:, 1].as_matrix()

# read in data, pre-process input documents
# output: all sentences of one document to a single line, excluding class labels
with open("CSC.txt", 'a', newline='') as w:
#with open("SAD.txt", 'a', newline='') as w:
    for entry in data:

        # extract sentences
        cont_sentences = extract_sentences(entry)
        # tokenize sentences
        cont_tokens = tokenize_sentences(cont_sentences)
        # pos-tag sentences
        cont_tags = pos_tag_sentences(cont_tokens)
        # rewrite pos-tag information
        cont_tags = change_pos_notation(cont_tags)
        # store pre-processed data in 'data.txt' file
        for sent in cont_tags:
            if not len(sent) == 0:
                w.write(str(sent) + "\t")
        w.write("\n")
w.close()
