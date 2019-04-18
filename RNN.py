# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

def word_To_Vector(sen_list):
 from gensim.models import word2vec
 wv_model = word2vec.Word2Vec(sen_list,size=50)
 wv_model.wv.syn0.shape
 return wv_model

def fun(sen_list,wv_model):
    word_set = set(wv_model.wv.index2word)
    X = np.zeros([len(sen_list),25,50])
    c = 0
    for sen in sen_list:
        nw=0
        for w in sen:
            if w in word_set:
                X[c,nw] = wv_model[w]
                nw=nw+1
        c=c+1
    return X

def word_count(train_df):
 train_df['word_count'] = train_df['Text'].str.lower().str.split().apply(len)
 print(train_df.head())

def split_train_test(train_df1,X):
 from sklearn.model_selection import train_test_split
 y = train_df1['Sentiment'].values
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
 return X_train,X_test,y_train,y_test


def RNN_Model(train_df,test_df):
    word_count_Threshold = 25
    train_df1 = train_df[:][train_df['word_count'] <= word_count_Threshold]
    test_df1 =  test_df[:][test_df['word_count'] <= word_count_Threshold]

    st_wd = text.ENGLISH_STOP_WORDS

    # Vectorization: Convert a collection of text documents to a matrix of token counts
    c_vector = CountVectorizer(stop_words=st_wd, min_df=.0001, lowercase=1)
    tc_vector = CountVectorizer(stop_words=st_wd, min_df=.0001, lowercase=1)

    c_vector.fit(train_df1['Text'].values)
    tc_vector.fit(test_df1['Text'].values)

    stop_words1 = list(c_vector.stop_words)
    stop_words2= list(tc_vector.stop_words)

    def remove_words(raw_sen, stop_words):
        sen = [w for w in raw_sen if w not in stop_words]
        return sen

    def reviewEdit(raw_sen_list, stop_words):
        sen_list = []
        for i in range(len(raw_sen_list)):
            raw_sen = raw_sen_list[i].split()
            sen_list.append(remove_words(raw_sen, stop_words))
        return sen_list

    train_sen_list = reviewEdit(list(train_df1['Text']), stop_words1)

    wv_model = word_To_Vector(train_sen_list)
    X_train = fun(train_sen_list, wv_model)
    y_train = train_df1['Sentiment'].values

    test_sen_list = reviewEdit(list(train_df1['Text']), stop_words2)
    wv_model = word_To_Vector(test_sen_list)
    X_test=fun(test_sen_list, wv_model)
    y_test=test_df1['Sentiment'].values

    # Splitting into training and test states
    #X_train, X_test, y_train, y_test = split_train_test(train_df1, X)

    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation,LSTM
    import time
    start_time1 = time.time()
    model1= Sequential()
    model1.add(LSTM(100,input_shape=(25,50),activation='relu'))
    model1.add(Dense(50,activation='relu'))
    model1.add(Dense(1,activation='sigmoid'))

    model1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    hist = model1.fit(X_train,y_train,validation_split=0.1,epochs=6,batch_size=64,verbose=1)

    print(model1.evaluate(X_test, y_test, batch_size=128))
    #print(model1.evaluate(X_train, y_train, batch_size=128))
    RNN_Exc_Time = time.time() - start_time1;

    import matplotlib.pyplot as plt
    loss_curve = hist.history['loss']
    epoch_c = list(range(len(loss_curve)))

    plt.xlabel('Epochs')
    plt.ylabel('Loss value')
    plt.plot(epoch_c,loss_curve,label='1 Hidden layer')
    plt.show()

    acc_curve = hist.history['acc']
    epoch_c = list(range(len(loss_curve)))

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy value')
    plt.plot(epoch_c,acc_curve,label='1 Hidden layer')
    plt.show()
    return RNN_Exc_Time



