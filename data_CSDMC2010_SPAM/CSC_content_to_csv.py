__author__ = 'Pat, Daniela'
__date__ = '09.08.2016'
# Extract content and get label from spam/ham corpus from data/CSC_corpus

import codecs
import csv
import email.parser
import os
from bs4 import BeautifulSoup


srcdir = 'TRAINING'
print('Input source directory: ' + str(srcdir) + '\n')

# save labels in dict()
labels = open('SPAMTrain.label',mode='r')
label_dict = dict()
for line in labels:
    line = line.split()
    label_dict[line[1]] = line[0]

# Extract the subject and payload from the .eml file.
def extract_sub_payload(file_dir, file_name):
    if not os.path.exists(file_dir):  # path does not exist
        print('ERROR: input file does not exist:', file_dir)
    with codecs.open(file_dir, 'r', encoding='utf-8', errors='ignore') as fp:
        msg = email.message_from_file(fp)
        payload = msg.get_payload()
        if type(payload) == type(list()):
            payload = payload[0]  # only use the first part of payload
        sub = msg.get('subject')
        sub = str(sub)
        if type(payload) != type(''):
            payload = str(payload)
        payload = BeautifulSoup(payload, "html.parser")
        payload = payload.get_text()
        sub = BeautifulSoup(sub, "html.parser")
        sub = sub.get_text()
    text = sub + payload
    mail_cont = list()
    mail_cont.append(file_name)
    mail_cont.append(label_dict[file_name])
    mail_cont.append(text)
    return mail_cont  # list: filename, label, e-mail content (subject plus text body)

# Extract subject line and body information from all .eml files in the srcdir
files = os.listdir(srcdir)
file_info = list()  # contains all files and their respective information
for file in files:
    srcpath = os.path.join(srcdir, file)
    file_info.append(extract_sub_payload(srcpath, file))  # for all files, get sub_cont list and append to file_info

# print all information to csv for further processing
data_csv = open('CSC_mails.csv', mode='w')
writer = csv.writer(data_csv)
header = 'filename label content'
writer.writerow(header.split())
for file in file_info:
    line=str(file[0])+' '+str(file[1])+' '
    cont = ''
    for word in file[2].split():
        cont += str(word+' ')
    writer.writerow(line.rsplit()+[cont])