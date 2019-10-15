#Alexander Kennedy 8/2/18
#Things to tweak:
#train_size vs test_size
#vocab_size
#length of subj_text entries used


import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from collections import Counter
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import matplotlib
import matplotlib.pyplot as plt
import textblob, string

from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn.metrics import confusion_matrix

#Read in Data into a pandas data frame
raw_data = pd.read_csv("Update_Case_Data_V2.csv")
#raw_data = pd.read_csv("test_data.csv")

#Create a copy of data
clean_data=raw_data.copy()

#Drop all empty rows
for index, row in clean_data.iterrows():
    if pd.isnull(row['SUBJ_TEXT']) or pd.isnull(row['LABEL']):
        clean_data.drop(index,inplace=True)


#Finish cleaning data
clean_data= clean_data.loc[:, clean_data.columns.intersection(['SUBJ_TEXT','LABEL'])]
clean_data=clean_data[:800]

#remove FORIEGN labels
for index, row in clean_data.iterrows():
    #if row['LABEL']== 'FOREIGN' or row['LABEL']=='GENERAL' or row['LABEL']=='RENEWAL ISSUE' or row['LABEL']=='LICENSING ISSUE' or len(row['SUBJ_TEXT'])<15:
    if row['LABEL']== 'FOREIGN' or row['LABEL']=='GENERAL' or row['LABEL']=='RENEWAL ISSUE' or len(row['SUBJ_TEXT'])<15:
        clean_data.drop(index,inplace=True)

clean_data=clean_data.reset_index(drop=True)

#Split data into 80% train and 20% test
train_size = int(len(clean_data) * .8)
train_posts = clean_data['SUBJ_TEXT'][:train_size]
train_tags = clean_data['LABEL'][:train_size]
test_posts = clean_data['SUBJ_TEXT'][train_size:]
test_tags = clean_data['LABEL'][train_size:]

#train_posts, test_posts2, train_tags2, test_tags2 = train_test_split(train_posts,train_tags,test_size=0.33,random_state=42)

#Encode the test_tags
encoder = preprocessing.LabelEncoder()
tags_x = encoder.fit_transform(train_tags)
tags_y = encoder.fit_transform(test_tags)

#Vectorize the sentences with and inverse word frequency vectorizer
vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=2000)
vectorizer.fit(clean_data['SUBJ_TEXT'])
data_x =  vectorizer.transform(train_posts)
data_y =  vectorizer.transform(test_posts)


gnb=MultinomialNB()
model=gnb.fit(data_x,tags_x)
prediction=gnb.predict(data_y)

print('\n----------------------------------------\n','Average Accuracy for the model:',np.mean(prediction == tags_y)*100,'%\n----------------------------------------\n')

#print(clean_data)

labels=[]
for label in train_tags:
    if label not in labels:
        labels.append(label)


counters={}
for index, row in clean_data.iterrows():
#    counters[row['LABEL']]
    counters[row['LABEL']]=0

for index, row in clean_data.iterrows():
#    counters[row['LABEL']]
    counters[row['LABEL']]+=1;

#Show how many of each type of issue there was
print(counters)

#Create a confusion matrix to better visualize what the model was mislabeling
cm = confusion_matrix(tags_y, prediction)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
