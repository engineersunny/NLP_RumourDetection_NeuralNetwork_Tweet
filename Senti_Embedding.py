from textblob import TextBlob

def senti(sentence):
    res = TextBlob(sentence).sentiment
    sentiment = ''
    if res.polarity <= 0:
        sentiment = 'negative'
    else:
        sentiment = 'positive'
    return sentiment

def sentimentCF(listTwts):
    for i, twt in enumerate(listTwts):
        senti(twt)
        listTwts[i] = listTwts[i] + ' ' + senti(twt)
    return listTwts


#Diff Method - Not using
"""import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
res = sid.polarity_scores("I am not happy")
print (res)

res = sid.polarity_scores("I am so so happy")
print (res)
"""