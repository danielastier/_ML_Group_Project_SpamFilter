'''
Created on 21 Aug 2016

@author: kibs

Reading and preprocessing code by __author__ = 'Daniela Stier'
'''

import string 
from nltk import corpus
import builtins

if hasattr(builtins, "rd_PATH"):
    PATH = builtins.rd_PATH
# read labels
#d = pd.read_csv('mails_bs_sm.csv')
#labels = d.iloc[:, 1].as_matrix()

# read preprocessed data (sentences segmented, tokenized, pos-tagged)
filename = str(PATH)
#print(filename)
reader = open(PATH, 'r')
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

#print(clean_data)

# number of words
# sentence lengths
num_words = [[len(sent) for sent in text] for text in clean_data]
# text lengths
num_words_summed = [sum(sent) for sent in num_words]

# clean_data excluding stopwords
stops = corpus.stopwords.words("english")
clean_data_wo_stops = [[[token for token in sent if not token[0:token.index("/")].lower() in stops] for sent in text] for text in clean_data]

# list of words only (excluding pos-tags)
pure_words = [[[word[0:word.index("/")] for word in sent] for sent in text] for text in clean_data]
# list of words only excluding stopwords (excluding pos-tags)
pure_words_wo_stops = [[[word[0:word.index("/")] for word in sent] for sent in text] for text in clean_data_wo_stops]

## Vocabulary Richness
## Type Token Ratio
## Number of distinct words / Total number of words (max 100)
num_tokens = num_words_summed
num_types = list()
for text in clean_data:
    text_types = list()
    for sent in text:
        for word in sent:
            if len(text_types)<100: text_types.append(word)
    text_types = set(text_types)
    num_types.append(len(text_types))

ttr = list()
for x in range(0,len(num_types)):
    if num_tokens[x] < 100:
        divisor = num_tokens[x]
    else:
        divisor = 100
    text_ttr = num_types[x] / divisor
    #print(text_ttr)
    ttr.append(text_ttr)

## TTR DONE

# list tags only (excluding tokens)
pure_tags_sent = [[[tag[tag.index("/")+1:] for tag in sent] for sent in text] for text in clean_data]
pure_tags_text = list()
for text in pure_tags_sent:
    temp = list()
    for sent in text:
        temp += sent
    pure_tags_text.append(temp)
num_tags = [[len(sent) for sent in text] for text in pure_tags_sent]
num_tags_summed = [sum(sent) for sent in num_tags]
pos_tags = "CC,CD,DT,EX,FW,IN,JJ,JJR,JJS,LS,MD,NN,NNS,NNP,NNPS,PDT,POS,PRP,PRP$,RB,RBR,RBS,RP,SYM,TO,UH,VB,VBD,VBG,VBN,VBP,VBZ,WDT,WP,WP$,WRB"
pos_tags = pos_tags.split(",")

## Function Words vs Content Words
## (Closed class)    (Open class)
## num_cont / num_funct
# closed class
funct_tags = "JJ,JJR,JJS,RB,RBR,RBS,NN,NNS,NNP,NNPS,VB,VBD,VBG,VBN,VBP,VBZ,FW"
funct_tags = funct_tags.split(',')
# open class
cont_tags = "CD,CC,DT,EX,IN,LS,MD,PDT,POS,PRP,PRP$,RP,TO,UH,WDT,WP,WP$,WRB"
cont_tags = cont_tags.split(',')
# one unaccounted tag: SYM

num_funct = 0
num_cont = 0
numfc_data = list()
for text in pure_tags_text:
    for tag in text:
        if tag in funct_tags:
            num_funct += 1
        elif tag in cont_tags:
            num_cont += 1
    numfc_data.append((num_funct, num_cont))
# num_funct > num_cont

open_closed_ratio = [cont/funct for (funct,cont) in numfc_data]
#print(open_closed_ratio)


## Personal Pronouns
## 1st, 2nd and 3rd person
prp_data = list()
prp_set = list()
for text in clean_data:
    prp_text = list()
    for sent in text:
        for toktag in sent:
            if toktag[-3:] == "PRP" or toktag[-4:] == "PRP$": 
                prp_text.append(toktag)
                prp_set.append(toktag[:toktag.find('/')])
    #print(prp_text)
    prp_data.append(prp_text)
#print(set(prp_set))
first = ['i','me','my','mine','myself','we','us','our','ours','ourselves']
second = ['you','your','yours','yourself','yourselves']
third = ['he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves']

all = first + second + third 
# for each text
# for each personal pronoun
# get count
# [[text1{prpa:0,...},{prp1:0,...},{prp2:0,...},{prp3:0,...}],[text2{...}]...]
# init_all = {prp:0 for prp in all}
# init_f = {prp:0 for prp in first}
# init_s = {prp:0 for prp in second}
# init_t = {prp:0 for prp in third}
each_prp_count_text = list()
for text in prp_data:
    #print(text)
    epc = list()
    a={prp:0 for prp in all}
    f={prp:0 for prp in first}
    s={prp:0 for prp in second}
    t={prp:0 for prp in third}
    for prp in text:
        prp = prp[:prp.find('/')].lower()
        if prp.find("'") != -1:
            prp = prp[:prp.find("'")]
        if prp in all:
            a[prp] += 1
            if prp in first:
                f[prp] += 1
            elif prp in second:
                s[prp] += 1
            elif prp in third:
                t[prp] += 1
    epc.append(a)    
    epc.append(f)
    epc.append(s)
    epc.append(t)
    #print(epc)
    each_prp_count_text.append(epc)

# [[text1{prpa:count/sum(prpa)...},{prp1:count/sum(prp1)...}...]...]
each_prp_freq = list()
for text in each_prp_count_text:
    epf = list()
    for prp_dict in text:
        freq = dict()
        sump = sum(prp_dict.values())
        for prp in prp_dict.keys():
            if sump!=0:
                freq[prp] = prp_dict[prp]/sump
            else:
                freq[prp] = 0
        epf.append(freq)
    each_prp_freq.append(epf)



prp_count_text = list()
for text in prp_data:
    num_f = 0
    num_s = 0
    num_t = 0
    for prp in text:
        prp = prp[:prp.find('/')].lower()
        if prp.find("'") != -1:
            prp = prp[:prp.find("'")]
        if prp in first:
            num_f +=1
        elif prp in second:
            num_s += 1
        elif prp in third:
            num_t += 1
        #else:
            #print(prp)
            # assume wrongly tagged, ignore
    prp_count_text.append((num_f, num_s, num_t))

prp_freq = list()
for x in range(0, len(num_words_summed)):
    num_prp = prp_count_text[x]
    t_len = num_words_summed[x]
    if sum(num_prp) != 0:
        prp_freq.append((num_prp[0]/t_len, num_prp[1]/t_len, num_prp[2]/t_len, num_prp[0]/sum(num_prp), num_prp[1]/sum(num_prp), num_prp[2]/sum(num_prp)))
    else:
        prp_freq.append((num_prp[0]/t_len, num_prp[1]/t_len, num_prp[2]/t_len, 0.0, 0.0, 0.0))


## Adjectives and Adverbs 
## Normal Form = JJ/RB, Comparative Form = JJR/RBR, Superlative Form = JJS/RBS
ad_count_text = list()
for text in pure_tags_text:
    num_norm = 0
    num_comp = 0
    num_supl = 0
    for tag in text:
        if tag == "JJ" or tag == "RB":
            num_norm += 1
        elif tag == "JJR" or tag == "RBR":
            num_comp += 1
        elif tag == "JJS" or tag == "RBS":
            num_supl += 1
    ad_count_text.append((num_norm, num_comp, num_supl))

ad_freq = list()
for x in range(0, len(num_words_summed)):
    freq = list()
    num_ad = ad_count_text[x]
    t_len = num_words_summed[x]
    freq.append(num_ad[0]/t_len)
    freq.append(num_ad[1]/t_len)
    freq.append(num_ad[2]/t_len)
    if sum(num_ad) != 0:
        freq.append(num_ad[0]/sum(num_ad))
        freq.append(num_ad[1]/sum(num_ad))
        freq.append(num_ad[2]/sum(num_ad))
    else:
        freq.append(0)
        freq.append(0)
        freq.append(0)
    ad_freq.append(freq)


### ADJ-N bigrams
### JJ-NN, JJ-NNS, JJ-NNP, JJ-NNPS
### JJS-NN, JJS-NNS, JJS-NNP, JJS-NNPS
### JJR-NN, JJR-NNS, JJR-NNP, JJR-NNPS

adjn_count_text = list()
for text in pure_tags_sent:
    adjn_count = [0 for x in range(0,12)]
    for sent in text:
        for i in range(0,len(sent)-1):
            if sent[i] == 'JJ' and sent[i+1] == 'NN':
                adjn_count[0] += 1
            elif sent[i] == 'JJ' and sent[i+1] == 'NNS':
                adjn_count[1] += 1
            elif sent[i] == 'JJ' and sent[i+1] == 'NNP':
                adjn_count[2] += 1
            elif sent[i] == 'JJ' and sent[i+1] == 'NNPS':
                adjn_count[3] += 1
            elif sent[i] == 'JJS' and sent[i+1] == 'NN':
                adjn_count[4] += 1
            elif sent[i] == 'JJS' and sent[i+1] == 'NNS':
                adjn_count[5] += 1
            elif sent[i] == 'JJS' and sent[i+1] == 'NNP':
                adjn_count[6] += 1
            elif sent[i] == 'JJS' and sent[i+1] == 'NNPS':
                adjn_count[7] += 1
            elif sent[i] == 'JJR' and sent[i+1] == 'NN':
                adjn_count[8] += 1
            elif sent[i] == 'JJR' and sent[i+1] == 'NNS':
                adjn_count[9] += 1
            elif sent[i] == 'JJR' and sent[i+1] == 'NNP':
                adjn_count[10] += 1
            elif sent[i] == 'JJR' and sent[i+1] == 'NNPS':
                adjn_count[11] += 1
    adjn_count_text.append(adjn_count)

adjn_freq = list()
for x in range(0, len(num_words_summed)):
    freq = list()
    suma = sum(adjn_count_text[x])
    freq.append(suma/num_words_summed[x]) # total adj-n / total words
    for adjn in adjn_count_text[x]:
        if suma != 0:
            freq.append(adjn/suma) # each combi total / total adj-n
        else:
            freq.append(0.0)
    if suma != 0:
        freq.append(sum(adjn_count_text[x][0:4])/suma) # JJ
        freq.append(sum(adjn_count_text[x][4:8])/suma) # JJS
        freq.append(sum(adjn_count_text[x][8:])/suma) # JJR
    else: 
        freq.append(0.0)
        freq.append(0.0)
        freq.append(0.0)
    adjn_freq.append(freq)

    
    

