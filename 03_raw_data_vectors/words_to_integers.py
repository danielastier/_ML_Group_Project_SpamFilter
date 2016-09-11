__author__ = 'Pat'
__date__ = '09.08.2016'
# Creates csv files which contain the mails where words (lexical words and punctuation) are represented as integers

import csv
import pandas as pd
from nltk import word_tokenize, sent_tokenize, FreqDist

# Choose the respective corpus and set it to True, the other to False
CSC = True
SAD = False

if CSC:
    print('Chosen corpus: CSC')
    read_mails = '../data_CSDMC2010_SPAM/CSC_mails.csv'
    mails_plain_txt = 'CSC_plain.txt'
    mails_to_int = 'CSC_to_int.csv'
if SAD:
    print('Chosen corpus: SAD')
    read_mails = '../data_SpamAssassin/SAD_mails.csv'
    mails_plain_txt = 'SAD_plain.txt'
    mails_to_int = 'SAD_to_int.csv'

# create plain mail texts
print('Creating plain mail texts...')
d = pd.read_csv(read_mails)
mails_text = ''
for mail_content in d.content:
    words = word_tokenize(str(mail_content))
    for word in words:
        mails_text += word+' '
mails_plain = open(mails_plain_txt, mode='w')
mails_plain.write(mails_text)
mails_plain.close()

# create doc of int
print('Creating list of docs, each doc a list of integers...')
sent_list = list()
mails_plain = open(mails_plain_txt, mode='r').read()
sentences = sent_tokenize(mails_plain)
for sent in sentences:
    tokens = word_tokenize(sent)
    for word in tokens:
        sent_list.append(word)
fd = FreqDist(sent_list)
word_to_num = dict()  # maps each of the 5,000 most frequent words to an integer
count = 1  # zero reserved for words with lower frequency (these words not included in this implementation) and padding
for (word,freq) in fd.most_common(5000):
    word_to_num[word]=count
    count += 1

docs_list = list()  # contains documents represented as lists of integers
for mail_content in d.content:
    document = list()  # the list of integers, all non-zero
    words = word_tokenize(str(mail_content))
    for word in words:
        if not (word_to_num.get(word) == None):
            document.append(word_to_num.get(word))
    docs_list.append(document)


print('Creating csv ...')
csv_header = ''
i = 0
while i < 500:
    csv_header += 'x'+str(i)+' '
    i += 1
csvfile = open(mails_to_int, 'w')
writer = csv.writer(csvfile)
writer.writerow(csv_header.split())

doc_count = 0
for document in docs_list:
    index = 0
    line=''
    while index < 500:
        while index < len(document) and index < 500:
            line += str(document[index])+' '
            index += 1
        while index >= len(document) and index < 500:
            line += '0 '
            index += 1
    writer.writerow(line.split())
    doc_count += 1
csvfile.close()