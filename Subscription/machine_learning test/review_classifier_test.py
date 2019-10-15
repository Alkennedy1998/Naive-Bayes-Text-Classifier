#Alexander Kennedy 8/2/18
#Things to tweak:
#train_size vs test_size
#vocab_size
#


import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from collections import Counter
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers



#Read in Data into a pandas data frame
raw_data = pd.read_csv("clothing_review.csv")
#raw_data = pd.read_csv("test_data.csv")

headers=raw_data.columns.values.tolist()
review_text=headers[4]
#Create a copy of data
clean_data=raw_data.copy()

#Drop all empty rows
for index, row in clean_data.iterrows():
    if pd.isnull(row['Rating']) or pd.isnull(row['Review Text']):
        clean_data.drop(index,inplace=True)

#Finish cleaning data
clean_data= clean_data.loc[:, clean_data.columns.intersection(['Review Text','Rating'])]

clean_data=clean_data.reset_index(drop=True)

train_size = int(len(clean_data) * .8)
train_posts = clean_data[review_text][:train_size]
train_tags = clean_data['Rating'][:train_size]
test_posts = clean_data[review_text][train_size:]
test_tags = clean_data['Rating'][train_size:]

#train_posts, test_posts2, train_tags2, test_tags2 = train_test_split(train_posts,train_tags,test_size=0.33,random_state=42)

#Encode the test_tags
encoder = preprocessing.LabelEncoder()
tags_x = encoder.fit_transform(train_tags)
tags_y = encoder.fit_transform(test_tags)

#Vectorize the sentences with and inverse word frequency vectorizer
vectorizer = TfidfVectorizer(analyzer='word',token_pattern=r'\w{1,}', max_features=2000)
#vectorizer.fit_transform(clean_data[review_text].values.astype('U'))
vectorizer.fit(clean_data['Review Text'])
data_x =  vectorizer.transform(train_posts)
data_y =  vectorizer.transform(test_posts)


mnb=MultinomialNB()
model=mnb.fit(data_x,tags_x)
prediction=mnb.predict(data_y)

print('\n----------------------------------------\n','Average Accuracy for the model:',np.mean(prediction == tags_y)*100,'%\n----------------------------------------\n')
