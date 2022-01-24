from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# Opening JSON file
from tensorflow.python.keras.callbacks import EarlyStopping

import Senti_Embedding
import preprocessing
import readData

print("Reading Dataset")
xtrain,ytrain, xvalid, yvalid, xtest, xcovid = readData.openfile()
print("Reading Dataset - Finished")

xtrain = preprocessing.process(xtrain)
xvalid = preprocessing.process(xvalid)

xtrain = Senti_Embedding.sentimentCF(xtrain)
xvalid = Senti_Embedding.sentimentCF(xvalid)

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



start = timer()
##===================original
# define model
model = Sequential()
model.add(Dense(20, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# fit the model
history = model.fit(X_train_array, Y_train_array, epochs=20, batch_size=32, verbose=0,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
# evaluate the model
loss, acc = model.evaluate(X_valid_array, Y_valid_array, verbose=0)

print('Test Accuracy - classic NN model: %.3f' % acc)


end = timer()
time = round((end - start),2)
print(time)


plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()
