#Alexander Kennedy 8/2/18
#Things to tweak:
#train_size vs test_size
#vocab_size
#

import numpy as np
import pandas as pd
import operator
import string
import keras
import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

#Read in Data into a pandas data frame
raw_data = pd.read_csv("small_data.csv")

#Create a copy of data
clean_data=raw_data.copy()

#Drop all empty rows
for index, row in clean_data.iterrows():
    if pd.isnull(row['LABEL']):
        clean_data.drop(index,inplace=True)

#Finish cleaning data
clean_data= clean_data.loc[:, clean_data.columns.intersection(['SUBJ_TEXT','LABEL'])]

#Split data into training and testing
train_size = int(len(clean_data) * .8)
train_posts = clean_data['SUBJ_TEXT'][:train_size]
train_tags = clean_data['LABEL'][:train_size]
test_posts = clean_data['SUBJ_TEXT'][train_size:]
test_tags = clean_data['LABEL'][train_size:]

#tokenize sentences
vocab_size = 1000
tokenize = Tokenizer(num_words=vocab_size)
tokenize.fit_on_texts(train_posts)
x_train = tokenize.texts_to_matrix(train_posts)

#
encoder = LabelBinarizer()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

#Build the model
model = Sequential()

#
model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))


model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=2,
                    verbose=1,
                    validation_split=0.1)


score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])









print(clean_data['SUBJ_TEXT'],clean_data['LABEL'])
