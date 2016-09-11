'''
Created on 24 Aug 2016

@author: kibs
'''

import csv
import sys
csv.field_size_limit(sys.maxsize)

import builtins
builtins.rd_PATH = sys.argv[1]

from read_data import ttr, open_closed_ratio, each_prp_freq, prp_freq, ad_freq, first, second, third, all, adjn_freq

#headers = open('content_headers.csv')

epf = each_prp_freq
labels = str(sys.argv[2])
labels_csv = open(labels)
labels_csv = csv.DictReader(labels_csv)

labels = list()
filenames = list()

for row in labels_csv:
    filenames.append(row['filename'])
    labels.append(row['label'])

output = str(sys.argv[3])
f = open(output, 'w')

#for line in headers.readlines():
#    f.write(line)
#headers.close()

for x in range(0, len(filenames)):
    f.write(labels[x]+','+
            str(ttr[x])+','+
            str(open_closed_ratio[x])+',')
    for prp in all:
        f.write(str(epf[x][0][prp])+',') # 31
    for prp in first:
        f.write(str(epf[x][1][prp])+',') # 31
    for prp in second:
        f.write(str(epf[x][2][prp])+',')
    for prp in third:
        f.write(str(epf[x][3][prp])+',')
    for prp in prp_freq[x]:
        f.write(str(prp)+',') # 6
    for ad in ad_freq[x]: # 6
        f.write(str(ad)+',')
    for fr in adjn_freq[x][:-1]:
        f.write(str(fr)+',') # 16
    f.write(str(adjn_freq[x][-1]))
    f.write('\n')

f.close()

print(str(sys.argv[1]) + " has been written to " + str(sys.argv[3]))