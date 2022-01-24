from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re, string

def process(xtrain):
    #Tokenise
    tt = TweetTokenizer()
    tokenised_lst = []
    for row in xtrain:
        tk_row = tt.tokenize(row)
        tokenised_lst.append(tk_row)

    #lemmatise
    lemmatizer = WordNetLemmatizer()

    for i, line in enumerate(tokenised_lst):
        lemmatized_sentence = []

        for word, tag in pos_tag(line):
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
        tokenised_lst[i] = lemmatized_sentence

    #remove noise
    stop_words = stopwords.words('english')

    for i, line in enumerate(tokenised_lst):
        cleaned_tokens = []
        for token, tag in pos_tag(line):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
            token = re.sub("(@[A-Za-z0-9_]+)", "", token)

            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())

        tokenised_lst[i] = ' '.join(cleaned_tokens) # token to string for TfidfVectorizer in LSTM.py

    return tokenised_lst #concated string


"""##TEST CODE
test = [':( :) You can\'t condemn an entire race, nation or religion based on the actions of a few radicals, please keep that in mind #sydneysiege', 'You can\'t condemn an entire race, nation or religion based on the actions of a few radicals, please keep that in mind #sydneysiege']
tokens = process(test)
print(tokens[0])"""

