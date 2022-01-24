import json
from timeit import default_timer as timer
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

print("Reading Dataset")
xtrain,ytrain, xvalid, yvalid, xtest, xcovid = readData.openfile()
print("Reading Dataset - Finished")

#preprocessing
xtrain = preprocessing.process(xtrain)
xvalid = preprocessing.process(xvalid)
xtest = preprocessing.process(xtest)
xcovid = preprocessing.process(xcovid)

#sentiment_test
xtrain = Senti_Embedding.sentimentCF(xtrain)
xvalid = Senti_Embedding.sentimentCF(xvalid)
xtest = Senti_Embedding.sentimentCF(xtest)
xcovid = Senti_Embedding.sentimentCF(xcovid)

#############################################################################
# TFID
vectorizer = TfidfVectorizer()
vectorizer.fit(xtrain)
n_features = 0 # determine the number of input features

def vec2array(vec, isShape=False):
    X = vectorizer.transform(vec)
    X = X.toarray()  # 4641x10731
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
X_train_array, n_features = vec2array(xtrain, True) #4641x10731
#valid
X_valid_array = vec2array(xvalid)
#test
X_test_array = vec2array(xtest)
X_covid_array = vec2array(xcovid)

# int to float
Y_train = [float(i) for i in ytrain]
Y_train_array = np.array(Y_train)

Y_valid = [float(i) for i in yvalid]
Y_valid_array = np.array(Y_valid)


####################
#RNN TEST
####################

# The maximum number of words
MAX_NB_WORDS = 5000 #50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 10 #100

start = timer()

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=n_features))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(10, dropout=0.2, recurrent_dropout=0.05)) #100
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 10
batch_size = 128

print("Start fitting the model")
history = model.fit(X_train_array, Y_train_array, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
print("Fitting model - Finished")

accr = model.evaluate(X_valid_array,Y_valid_array)
print('Validation Set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

end = timer()
time = round((end - start),2)
print(time)

print(model.summary())


##########################################
# Dev data prediction json file extracting
##########################################
xtrainID,xvalidID,xtestID,xcovidID = readData.getID()

res_json = {}
dev_lst = model.predict(X_valid_array)

for i, x in enumerate(xvalidID):
    if dev_lst[i][0] >= 0.5: res = "non-rumour"
    else: res ="rumour"
    res_json[xvalidID[i]] = res

for i, x in enumerate(xvalidID):
    res="rumour"
    res_json[xvalidID[i]] = res

with open('result_prediction.json', 'w') as json_file:
    json.dump(res_json, json_file, separators=(",", ":"))

print('done')


##===================
model = Sequential()
model.add(Dense(20, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_array, Y_train_array, epochs=150, batch_size=32, verbose=0)
loss, acc = model.evaluate(X_valid_array, Y_valid_array, verbose=0)
#print('Test Accuracy - classic NN model: %.3f' % acc)

### valid set
res_json = {}
dev_lst = model.predict(X_valid_array)

for i, x in enumerate(xvalidID):
    if dev_lst[i][0] >= 0.5: res = "non-rumour"
    else: res ="rumour"
    res_json[xvalidID[i]] = res

with open('result_orgcf_prediction.json', 'w') as json_file:
    json.dump(res_json, json_file, separators=(",", ":"))

print('done')

######### valid ##############################
### valid
res_json = {}
dev_lst = model.predict(X_test_array)

for i, x in enumerate(xtestID):
    if dev_lst[i][0] >= 0.5: res = "non-rumour"
    else: res ="rumour"
    res_json[xtestID[i]] = res

with open('test-output.json', 'w') as json_file:
    json.dump(res_json, json_file, separators=(",", ":"))

print('test json extration - done')


##########Covid Data###################

res_json = {}
covid_lst = model.predict(X_covid_array)

for i, x in enumerate(xcovidID):
    if covid_lst[i][0] > 0: res = "non-rumour"
    else: res ="rumour"
    res_json[xcovidID[i]] = res

with open('covid_pred.json', 'w') as json_file:
    json.dump(res_json, json_file, separators=(",", ":"))

print('done')

