from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import operator
import string

sns.set()

#Import data from the file into a SERIES OF LISTS?
review_data = pd.read_csv("all_data.csv")

#max_age=review_data['Age'].max()

review_data['word_count']=review_data['SUBJ_TEXT'].apply(lambda x: len(str(x).split(" ")))

#review_data[['Review Text','word_count']].head()

words=[]

word_freq={}

for word in review_data['SUBJ_TEXT']:
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

#print(word_freq)
#print(max(word_freq.items(), key=operator.itemgetter(1))[0])

#Create a new list of lists sorted by word frequency
sorted_by_value =sorted(word_freq.items(), key=lambda kv: kv[1])




#remove item if number of reports under 3

word_freq_cleaned={}

for key in word_freq:
    for word in complex_words:
        if key == word and word_freq[key]>300:
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


df=pd.DataFrame(list_of_freq,index=list_of_keys,columns=['Word Frequency'])
df=df.sort_values(by='Word Frequency')
print(df)



df.plot.bar()
