__author__ = 'Daniela Stier'

### IMPORT STATEMENTS
import string
import pandas as pd
from collections import defaultdict, OrderedDict, Counter

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
clean_data = [[[token  for token in sent if not token[0:token.index("/")].lower() in puncts] for sent in text] for text in clean_data]

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


######### MEASURES CONCERNING CHARACTERS ##############################

# number of characters
num_chars = [[[len(word) for word in sent] for sent in text] for text in pure_words]
num_chars_summed_sent = [[sum(sent) for sent in text] for text in num_chars]
num_chars_summed = [sum(text) for text in num_chars_summed_sent]

# frequency of characters
# rel. freq of each char: dividing the frequency of that char in a text by the total frequencies of chars
letters = string.ascii_letters
counter = defaultdict(list)
for i in letters:
    for text in pure_words_text:
        temp = Counter(text)
        counter[i].append(temp[i])
counter = OrderedDict(sorted(counter.items(), key=lambda x: x[0]))

char_freq_dist = list()
for value in counter.values():
    temp = list()
    for i in range(len(num_chars_summed)):
        if num_chars_summed[i] > 0:
            temp.append(value[i]/num_chars_summed[i])
        else:
            temp.append(0)
    char_freq_dist.append(temp)


# store vectorized data in 'test_data_sentence_vector.csv' file, including labels at first position
data_matrix = pd.concat([pd.DataFrame(num_chars_summed), pd.DataFrame(char_freq_dist).transpose()], axis=1)
data_matrix.to_csv('01_CSC_vectors/CSC_char_vector.csv', index=False, delimiter=',')
#data_matrix.to_csv('02_SAD_vectors/SAD_char_vector.csv', index=False, delimiter=',')