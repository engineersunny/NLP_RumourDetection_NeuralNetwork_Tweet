import json

import jsonlines
import numpy as np
from keras.layers import LSTM, Embedding
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# Opening JSON file
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import SpatialDropout1D

import Senti_Embedding
import preprocessing
import readData

filepath = './data/covid.data.jsonl'

mLines = []
with jsonlines.open(filepath) as f:
    for line in f.iter():
        aLine = []
        for twt in line:
            d = {}
            d['curr_id'] = twt['id_str']
            d['parent_id'] = twt['in_reply_to_status_id_str']
            d['text'] = twt['text']
            d['user_info'] = twt['user']
            d['created'] = twt['created_at']
            aLine.append(d)
        mLines.append(aLine)

print("Reading Dataset")
xtrain,ytrain, xvalid, yvalid, xtest, xcovid = readData.openfile()
xtrain = xtrain
ytrain = ytrain
print("Reading Dataset - Finished")

#preprocessing
xtrain = preprocessing.process(xtrain)
xcovid = preprocessing.process(xcovid)
#sentiment_test
xtrain = Senti_Embedding.sentimentCF(xtrain)
xcovid = Senti_Embedding.sentimentCF(xcovid)

vectorizer = TfidfVectorizer()
vectorizer.fit(xtrain)
n_features = 0 # determine the number of input features

def vec2array(vec, isShape=False):
    X = vectorizer.transform(vec)
    X = X.toarray()
    # int to float
    res_list = []
    for line in X:
        res_list.append([float(i) for i in line])
    res_array = np.array(res_list)

    if isShape == True:
        n_features = X.shape[1]
        return res_array, n_features
    else: return res_array

#train
X_train_array, n_features = vec2array(xtrain, True)
X_covid_array = vec2array(xcovid)

# int to float
Y_train = [float(i) for i in ytrain]
Y_train_array = np.array(Y_train)

#---------------------------------------------------------------------------------------------------------------------
####################
#RNN TEST
####################
MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 10

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=n_features))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(10, dropout=0.02, recurrent_dropout=0.02)) #100
model.add(Dense(1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  #optimizer='ranger'

epochs = 10
batch_size = 64

print("Start fitting the model")
history = model.fit(X_train_array, Y_train_array, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
print("Fitting model - Finished")

########################################
xtrainID,xvalidID,xtestID,xcovidID = readData.getID()
##########Covid Data###################

res_json = {}
covid_lst = model.predict(X_covid_array)
#print(covid_lst)

for i, x in enumerate(xcovidID):
    if covid_lst[i][0] > 0: res = "non-rumour"
    else:
        res ="rumour"
        print(xcovidID[i])
    res_json[xcovidID[i]] = res

with open('covid_pred.json', 'w') as json_file:
    json.dump(res_json, json_file, separators=(",", ":"))
print('done')


# define model
model = Sequential()
model.add(Dense(20, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit the model
history = model.fit(X_train_array, Y_train_array, epochs=20, batch_size=32, verbose=0,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
xtrainID,xvalidID,xtestID,xcovidID = readData.getID()

res_json = {}
covid_lst = model.predict(X_covid_array)
print(covid_lst)
resarr = []

for i, x in enumerate(xcovidID):
    if covid_lst[i][0] >= 0.5: res = "non-rumour"
    else:
        res ="rumour"
        print(xcovidID[i])

    resarr.append(res)
    res_json[xcovidID[i]] = res

with open('covid_pred.json', 'w') as json_file:
    json.dump(res_json, json_file, separators=(",", ":"))
print('done')