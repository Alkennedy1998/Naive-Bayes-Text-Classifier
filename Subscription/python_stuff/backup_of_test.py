from __future__ import print_function, division

from textblob import TextBlob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import operator
import string


#Import data from the file into a SERIES OF LISTS?
raw_data = pd.read_csv("Update_Case_Data_V2.csv")

clean_data=raw_data.copy()
for index, row in clean_data.iterrows():
    if pd.isnull(row['SUBJ_TEXT']) or pd.isnull(row['LABEL']) or row['LABEL']!='PAYMENT ISSUE':
        clean_data.drop(index,inplace=True)


clean_data['word_count']=clean_data['SUBJ_TEXT'].apply(lambda x: len(str(x).split(" ")))

#review_data[['Review Text','word_count']].head()

clean_data['n_grams']=clean_data['SUBJ_TEXT'].apply(lambda x: TextBlob(x).ngrams(4))

words=[]

word_freq={}

for word in clean_data['SUBJ_TEXT']:
    words.extend(word.split(" "))

#Clean the text
things_to_remove=string.punctuation
table = str.maketrans('', '', things_to_remove)
cleaned_words = [w.translate(table) for w in words]
cleaned_words = [word.lower() for word in cleaned_words]
#print(cleaned_words)


complex_words=[]
for word in cleaned_words[:]:
    if(len(word)>4 and word!=''):
        complex_words.append(word)
        if word in word_freq:
            word_freq[word]+=1;
        else:
            word_freq[word]=1;


word_freq_cleaned={}

for key in word_freq:
    for word in complex_words:
        if key == word and word_freq[key]>10:
            word_freq_cleaned.update({key:word_freq[key]})

#create list of all cleaned words
list_of_keys=[]

for key in word_freq_cleaned:
    list_of_keys.append(key)

#create list of all frequency
list_of_freq=[]
for key in word_freq_cleaned:
    list_of_freq.append(word_freq[key])

#print(list_of_freq)


df=pd.DataFrame(list_of_freq,index=list_of_keys,columns=['Word Frequency for Payment Issues'])
df=df.sort_values(by='Word Frequency for Payment Issues')
print(df)

blob1=TextBlob(raw_data['SUBJ_TEXT'][2]).ngrams(1)
blob2=TextBlob(raw_data['SUBJ_TEXT'][1]).ngrams(1)

clean_data['n_grams'][:5]

#Import data from the file into a SERIES OF LISTS?
raw_data = pd.read_csv("Update_Case_Data_V2.csv")

clean_data=raw_data.copy()
for index, row in clean_data.iterrows():
    if pd.isnull(row['SUBJ_TEXT']) or pd.isnull(row['LABEL']) or row['LABEL']!='PAYMENT ISSUE':
        clean_data.drop(index,inplace=True)


clean_data['word_count']=clean_data['SUBJ_TEXT'].apply(lambda x: len(str(x).split(" ")))

#review_data[['Review Text','word_count']].head()

clean_data['n_grams']=clean_data['SUBJ_TEXT'].apply(lambda x: TextBlob(x).ngrams(4))

words=[]

word_freq={}

for word in clean_data['SUBJ_TEXT']:
    words.extend(word.split(" "))

#Clean the text
things_to_remove=string.punctuation
table = str.maketrans('', '', things_to_remove)
cleaned_words = [w.translate(table) for w in words]
cleaned_words = [word.lower() for word in cleaned_words]
#print(cleaned_words)


complex_words=[]
for word in cleaned_words[:]:
    if(len(word)>4 and word!=''):
        complex_words.append(word)
        if word in word_freq:
            word_freq[word]+=1;
        else:
            word_freq[word]=1;


word_freq_cleaned={}

for key in word_freq:
    for word in complex_words:
        if key == word and word_freq[key]>10:
            word_freq_cleaned.update({key:word_freq[key]})

#create list of all cleaned words
list_of_keys=[]

for key in word_freq_cleaned:
    list_of_keys.append(key)

#create list of all frequency
list_of_freq=[]
for key in word_freq_cleaned:
    list_of_freq.append(word_freq[key])

#print(list_of_freq)


df=pd.DataFrame(list_of_freq,index=list_of_keys,columns=['Word Frequency for Payment Issues'])
df=df.sort_values(by='Word Frequency for Payment Issues')

blob1=TextBlob(raw_data['SUBJ_TEXT'][2]).ngrams(1)
blob2=TextBlob(raw_data['SUBJ_TEXT'][1]).ngrams(1)

print(clean_data['n_grams'][:5])
