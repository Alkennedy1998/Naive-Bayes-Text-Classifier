from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

review_data = pd.read_csv("case_data.csv")

analyser = SentimentIntensityAnalyzer()

def print_sentiment_scores(sentence):
    snt =analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(snt)))

for word in review_data['SUBJ_TEXT']:
    #print(word)
    #print("\n Sentiment: ")
    print_sentiment_scores(word)
