__author__ = 'Daniela Stier'

### IMPORT STATEMENTS
import string
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from nltk import corpus

# read preprocessed data, store in lists (clean_data)
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

# clean_data excluding stopwords
stops = corpus.stopwords.words("english")
clean_data_wo_stops = [[[token for token in sent if not token[0:token.index("/")].lower() in stops] for sent in text] for text in clean_data]

# list of words only (excluding pos-tags)
pure_words = [[[word[0:word.index("/")] for word in sent] for sent in text] for text in clean_data]

# list of words only (excluding pos-tags) on text-level
pure_words_text = list()
for text in pure_words:
    temp = ""
    for sent in text:
        for word in sent:
            temp += word + " "
    pure_words_text.append(temp)

# list of words only excluding stopwords (excluding pos-tags)
pure_words_wo_stops = [[[word[0:word.index("/")] for word in sent] for sent in text] for text in clean_data_wo_stops]

# list tags only (excluding tokens)
pure_tags_sent = [[[tag[tag.index("/")+1:] for tag in sent] for sent in text] for text in clean_data]
pure_tags_text = list()
for text in pure_tags_sent:
    temp = list()
    for sent in text:
        temp.append(sent)
    pure_tags_text.append(temp)
num_tags = [[len(sent) for sent in text] for text in pure_tags_sent]
num_tags_summed = [sum(sent) for sent in num_tags]

# number of words
num_words = [[len(sent) for sent in text] for text in clean_data]
num_words_summed = [sum(sent) for sent in num_words]

# number of words excluding stopwords
num_words_wo_stops = [[len(sent) for sent in text] for text in clean_data_wo_stops]
num_words_wo_stops_summed = [sum(sent) for sent in num_words_wo_stops]

# number of characters
num_chars = [[[len(word) for word in sent] for sent in text] for text in pure_words]
num_chars_summed_sent = [[sum(sent) for sent in text] for text in num_chars]
num_chars_summed = [sum(text) for text in num_chars_summed_sent]

# number of characters excluding stopwords
num_chars_wo_stops = [[[len(word) for word in sent] for sent in text] for text in pure_words_wo_stops]
num_chars_wo_stops_summed_sent = [[sum(sent) for sent in text] for text in num_chars_wo_stops]
num_chars_wo_stops_summed = [sum(text) for text in num_chars_wo_stops_summed_sent]


######### MEASURES CONCERNING WORDS ##############################

# average word length including stop words
# dividing the total number of characters in a text by the total number of words in a text
av_word_length = list()
for i, j in zip(num_chars_summed, num_words_summed):
    if j > 0:
        av_word_length.append(i/j)
    else:
        av_word_length.append(0)

# average word length excluding stopwords, output: float(number)
av_word_length_wo_stops = list()
for i, j in zip(num_chars_wo_stops_summed, num_words_wo_stops_summed):
    if j > 0:
        av_word_length_wo_stops.append(i/j)
    else:
        av_word_length_wo_stops.append(0)

# frequency of part-of-speech tags
# rel. freq of each pos tag: dividing the frequency of that tag in a text by the total frequencies of tags
pos_tags = "CC,CD,DT,EX,FW,IN,JJ,JJR,JJS,LS,MD,NN,NNS,NNP,NNPS,PDT,POS,PRP,PRP$,RB,RBR,RBS,RP,SYM,TO,UH,VB,VBD,VBG,VBN,VBP,VBZ,WDT,WP,WP$,WRB"
counter = defaultdict(str)
for i in pos_tags.split(","):
    counter[i] = [text.count(i) for text in pure_tags_text]
counter = OrderedDict(sorted(counter.items(), key=lambda x: x[0]))

pos_freq_dist = list()
for value in counter.values():
    temp = list()
    for i in range(len(num_tags_summed)):
        if num_tags_summed[i] > 0:
            temp.append(value[i]/num_tags_summed[i])
        else:
            temp.append(0)
    pos_freq_dist.append(temp)

# n-character words distribution (according to the whole text, based on characters)
# rel. freq of each word-length: dividing total number of words of that length by total number of words
wos = list()
for text in num_chars:
    temp = list()
    for sent in text:
        for word in sent:
            temp.append(word)
    wos.append(temp)
max_num_char = [np.mean(sent) for sent in wos]
counter = defaultdict(int)
c = 0
for i in range(int(max_num_char[c])-10, int(max_num_char[c])+15):
    counter[i] = [text.count(i) for text in num_chars_summed_sent]
    c += 1

word_length_dist_chars = list()
for value in counter.values():
    temp = list()
    for i in range(len(num_words_summed)):
        if num_words_summed[i] > 0:
            temp.append(value[i]/num_words_summed[i])
        else:
            temp.append(0)
    word_length_dist_chars.append(temp)


# store vectorized data in 'test_data_sentence_vector.csv' file, including labels at first position
data_matrix = pd.concat([pd.DataFrame(num_words_summed), pd.DataFrame(num_words_wo_stops_summed), pd.DataFrame(av_word_length), pd.DataFrame(av_word_length_wo_stops), pd.DataFrame(pos_freq_dist).transpose(), pd.DataFrame(word_length_dist_chars).transpose()], axis=1)
data_matrix.to_csv('01_CSC_vectors/CSC_word_vector.csv', index=False, delimiter=',')
#data_matrix.to_csv('02_SAD_vectors/SAD_word_vector.csv', index=False, delimiter=',')