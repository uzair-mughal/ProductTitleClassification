import os
import pickle
import tkinter
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from UtilityFunctions import *
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from UtilityFunctions import *
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def load_models():

    model_cat_1 = pickle.load(open('h_models_svm/model_c1.sav', 'rb'))
    model_cat_2 = pickle.load(open('h_models_svm/model_c2.sav', 'rb'))
    model_cat_3 = pickle.load(open('h_models_svm/model_c3.sav', 'rb'))
    return model_cat_1, model_cat_2, model_cat_3

def svm_pred_query(query,tfidf_vectorizer,model_cat_1,model_cat_2,model_cat_3):

    corpus_vocabulary = defaultdict(None, copy.deepcopy(tfidf_vectorizer.vocabulary_))
    corpus_vocabulary.default_factory = corpus_vocabulary.__len__

    tfidf_transformer_query = TfidfVectorizer()
    tfidf_transformer_query.fit_transform([query])

    for word in tfidf_transformer_query.vocabulary_.keys():
        if word in tfidf_vectorizer.vocabulary_:
            corpus_vocabulary[word]

    tfidf_transformer_query_sec = TfidfVectorizer(vocabulary=corpus_vocabulary)
    query_tfidf_matrix = tfidf_transformer_query_sec.fit_transform([query])
    return model_cat_1.predict(query_tfidf_matrix), model_cat_2.predict(query_tfidf_matrix),model_cat_3.predict(query_tfidf_matrix)

def rnn_pred_query(query_df):
    token = RegexpTokenizer("[\w']+")
    maxWords = 20000
    MaxWordLength = train_df.title.map(lambda x: len(token.tokenize(x))).max()
    tokenizer = Tokenizer(num_words = maxWords, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(train_df['title'].values)

    cat = []

    seq = tokenizer.texts_to_sequences((query_df['title']).values)
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MaxWordLength)

    pred1 = model_c1.predict(padded)
    cat.append(labels_c1[np.argmax(pred1)])

    query_df['title'] = query_df['title'] + ' ' + labels_c1[np.argmax(pred1)]

    seq = tokenizer.texts_to_sequences((query_df['title']).values)
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MaxWordLength)

    pred2 = model_c2.predict(padded)
    cat.append(labels_c2[np.argmax(pred2)])

    query_df['title'] = query_df['title'] +' '+ labels_c2[np.argmax(pred2)]

    seq = tokenizer.texts_to_sequences((query_df['title']).values)
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MaxWordLength)

    pred3 = model_c3.predict(padded)
    cat.append(labels_c3[np.argmax(pred3)])
    # cat = []

    # seq = tokenizer.texts_to_sequences(query)
    # padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MaxWordLength)

    # pred1 = model_c1.predict(padded)
    # cat.append(labels_c1[np.argmax(pred1)])

    # query = query + ' ' + labels_c1[np.argmax(pred1)]

    # seq = tokenizer.texts_to_sequences(query)
    # padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MaxWordLength)

    # pred2 = model_c2.predict(padded)
    # cat.append(labels_c2[np.argmax(pred2)])

    # query = query +' '+ labels_c2[np.argmax(pred2)]

    # seq = tokenizer.texts_to_sequences(query)
    # padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MaxWordLength)

    # pred3 = model_c3.predict(padded)
    # cat.append(labels_c3[np.argmax(pred3)])

    return cat

def main_window():

    def find_cat():
        title = title_entry.get()
        desc = desc_entry.get()
        
        data = {'title':[title],
                'description':[desc]}
        
        query_df = pd.DataFrame(data)
        PreProcess(query_df)

        query = str(query_df['title'].values+' '+query_df['description'].values)
        query_df['title'] = query_df['title'] + ' ' + query_df['description']
        query_df = query_df.drop(['description'], axis=1)

        svm_pred = svm_pred_query(query,tfidf_vectorizer,model_cat_1,model_cat_2,model_cat_3)
        svm_pred = [p[0] for p in svm_pred]
        svm = ''
        svm += 'Category 1 : '+ svm_pred[0] + '\nCategory 2: ' + svm_pred[1] + '\nCategory 3: ' + svm_pred[2]
        svm_textbar.insert(tkinter.INSERT, svm)

        rnn_pred = rnn_pred_query(query_df)
        rnn = ''
        rnn += 'Category 1 : '+ rnn_pred[0] + '\nCategory 2: ' + rnn_pred[1] + '\nCategory 3: ' + rnn_pred[2]
        rnn_textbar.insert(tkinter.INSERT, rnn)
        

    window = tkinter.Tk()
    window.title('')
    window.config(bg='PeachPuff2')
    width = 1000
    height = 600
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x_cod = (screen_width / 2) - (width / 2)
    y_cod = (screen_height / 2) - (height / 2)
    window.geometry('%dx%d+%d+%d' % (width, height, x_cod, y_cod))
    ptc_label = tkinter.Label(window, text='Product Title Classification', bg='PeachPuff2', fg='black')
    ptc_label.config(font=("Courier", 20, 'bold'))
    ptc_label.place(x=220, y=40)

    mid_label = tkinter.Label(window, text='SVM VS RNN', bg='PeachPuff2', fg='black')
    mid_label.config(font=("Courier", 20, 'bold'))
    mid_label.place(x=380, y=80)

    title_label = tkinter.Label(window, text='Title:', bg='PeachPuff2', fg='black')
    title_label.config(font=("Courier", 12))
    title_label.place(x=100, y=180)

    title_entry = tkinter.Entry(window)
    title_entry.pack()
    title_entry.place(x=260, y=180,height=25 , width=620)

    desc_label = tkinter.Label(window, text='Description:', bg='PeachPuff2', fg='black')
    desc_label.config(font=("Courier", 12))
    desc_label.place(x=100, y=220)

    desc_entry = tkinter.Entry(window)
    desc_entry.pack()
    desc_entry.place(x=260, y=220,height=25 , width=620)

    exe_button = tkinter.Button(window, text="Execute",command=find_cat, bg='white')
    exe_button.place(x=470, y=260, height=30, width= 120)
    

    pred_label = tkinter.Label(window, text='Predicted Categories:', bg='PeachPuff2', fg='black')
    pred_label.config(font=("Courier", 12))
    pred_label.place(x=100, y=300)

    svm_label = tkinter.Label(window, text='SVM', bg='PeachPuff2', fg='black')
    svm_label.config(font=("Courier", 12))
    svm_label.place(x=250, y=360)

    rnn_label = tkinter.Label(window, text='RNN', bg='PeachPuff2', fg='black')
    rnn_label.config(font=("Courier", 12))
    rnn_label.place(x=680, y=360)

    svm_textbar = tkinter.Text(window)
    svm_textbar.pack()
    svm_textbar.place(x=100, y=400,height=120 , width=350)

    rnn_textbar = tkinter.Text(window)
    rnn_textbar.pack()
    rnn_textbar.place(x=530, y=400,height=120 , width=350)

    window.mainloop()


if __name__ == "__main__":

    train_df = pd.read_csv("clean.csv")
    labels_c1=sorted(train_df['c1'].unique())
    labels_c2=sorted(train_df['c2'].unique())
    train_df['c3'] = train_df['c3'].replace(np.nan, 'none', regex=True)
    labels_c3=sorted(train_df['c3'].unique())

    tfidf_vectorizer = pickle.load(open('h_models_svm/tfidf_vectorizer.pickle','rb'))
    model_cat_1, model_cat_2, model_cat_3 = load_models()
    model_c1=tf.keras.models.load_model('h_models_rnn/model_c1/0.922')
    model_c2=tf.keras.models.load_model('h_models_rnn/model_c2/0.881')
    model_c3=tf.keras.models.load_model('h_models_rnn/model_c3/0.800')
   
    main_window()









    # Labels = ['country', 'id', 'title', 'c1', 'c2', 'c3', 'description', 'price', 'type']
    # train_df = pd.read_csv("data_train.csv", names=Labels)
    # PreProcess(train_df)
    # train_df = train_df.drop(['country','id','price','type'], axis=1)
    # train_df['title'] = train_df['title'] + ' ' + train_df['description']
    # train_df = train_df.drop(['description'], axis=1)
    # train_df.to_csv(r'clean.csv', index = False, header = True)

    # DEGUGG
    # Accuracy = []
    # Labels = ['country', 'id', 'title', 'c1', 'c2', 'c3', 'description', 'price', 'type']
    # test_df = pd.read_csv('data_test.csv',names = Labels)

    # tfidf_vectorizer_test = TfidfVectorizer()
    # x_test_tfidf, Y1, Y2, Y3 = cleaning_data(test_df, tfidf_vectorizer_test, tfidf_vectorizer)
    
    # y_pred_cat_1 = model_cat_1.predict(x_test_tfidf)
    # Accuracy.append(round(accuracy_score(Y1, y_pred_cat_1)*100))
    
    # y_pred_cat_2 = model_cat_2.predict(x_test_tfidf)
    # Accuracy.append(round(accuracy_score(Y2, y_pred_cat_2)*100))

    # y_pred_cat_3 = model_cat_3.predict(x_test_tfidf)
    # Accuracy.append(round(accuracy_score(Y3, y_pred_cat_3)*100))

    # print(Accuracy)