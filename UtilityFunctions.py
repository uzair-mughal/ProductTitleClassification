import copy
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_fet_class(df):
    
    df.drop(['country','id','price','type'],inplace=True,axis=1)
    df['c3'] = df['c3'].astype(str) #removing all the non string
    Y1 = df['c1']
    Y2 = df['c2']
    Y3 = df['c3']
    X = df.drop(['c1','c2','c3'],axis=1)
    X = X['title'] +' '+ X['description']
    
    return X, Y1,Y2,Y3

def replace_all(text, sym=['<ul>', '</ul>', '<li>', '</li>']):
    
        for i in sym:
            text = text.replace(i, ' ')

        textSec = []
        for i in text.split(' '):
            textTh = []
            for j in i:
                textTh.append(j if ord(j) < 128 else ' ')
            textTh = ''.join(textTh).strip().split(' ')

            for t in textTh:
                if t != '':
                    textSec.append(t)

        return ' '.join(textSec)

def remove_puncs(text):
    
        punctuations = '''!()–+=-[]{};:'"\,<>./?@#$%^&*_~'''
        no_punct = ""
        for char in text:
            if char not in punctuations:
                no_punct += char

        return no_punct

def PreProcess(df, cols=['title','description']):
        token = RegexpTokenizer("[\w']+")

        for col in cols:
            df[col].fillna('', inplace=True)
            df[col] = df[col].apply(lambda x: (x.lower())) 
            df[col] = df[col].apply(lambda x: replace_all(x))
            if col == 'description':
                df[col] = df[col].apply(lambda x: BeautifulSoup(x, "html.parser").text)
            df[col] = df[col].apply(lambda x: remove_puncs(x))
            df[col] = df[col].str.replace('\d+', ' ')
            df[col] = df[col].apply(lambda x: token.tokenize(x))
            df[col] = df[col].apply(lambda x: ' '.join([w for w in x if len(w)>1]))

def cleaning_data(df, tfidf_vectorizer, tfidf_vectorizer_train=None):
    
    PreProcess(df)
    X, Y1, Y2, Y3 = extract_fet_class(df)
    X_tfidf = tfidf_vectorizer.fit_transform(X)
    
    if tfidf_vectorizer_train:
    
        corpus_vocabulary = defaultdict(None, copy.deepcopy(tfidf_vectorizer_train.vocabulary_))
        corpus_vocabulary.default_factory = corpus_vocabulary.__len__
    
        for word in tfidf_vectorizer.vocabulary_.keys():
            if word in tfidf_vectorizer_train.vocabulary_:
                corpus_vocabulary[word]
                
        tfidf_vectorizer = TfidfVectorizer(vocabulary=corpus_vocabulary)
        X_tfidf = tfidf_vectorizer.fit_transform(X)
    
    return X_tfidf, Y1, Y2, Y3