__author__ = 'Pat'
__date__ = '09.08.2016'
# Extract content and get label from hard-ham corpus
# from data/SAD_corpus/hard_ham and data/SAD_corpus/spam


import csv
import os
from bs4 import BeautifulSoup

directories = ['hard_ham', 'spam']
label_dict = dict()
for directory in directories:
    if directory == 'hard_ham':
        label = 1
    elif directory == 'spam':
        label = 0
    else:
        label = None
    for file in os.listdir(directory):
        label_dict[file] = label

def extract_sub_payload(file_dir, file_name):
    if not os.path.exists(file_dir):
        print('ERROR: input file does not exist:', file_dir)
    with open(file_dir, 'r', encoding='utf-8', errors='ignore') as fp:
        msg = fp.read()
        content = BeautifulSoup(msg, "html.parser")
        content = content.get_text()
    mail_cont = list()
    mail_cont.append(file_name)
    mail_cont.append(label_dict[file_name])
    mail_cont.append(content)
    return mail_cont  # list of format: filename, label, e-mail content

file_info = list()
for directory in directories:
    for file in os.listdir(directory):
        srcpath = os.path.join(directory, file)
        file_info.append(extract_sub_payload(srcpath, file))

# print all information to csv for further processing
data_csv = open('SAD_mails.csv', mode='w')
writer = csv.writer(data_csv)
header = 'filename label content'
writer.writerow(header.split())
for file in file_info:
    line=str(file[0])+' '+str(file[1])+' '
    cont = ''
    for word in file[2].split():
        cont += str(word+' ')
    writer.writerow(line.rsplit()+[cont])
