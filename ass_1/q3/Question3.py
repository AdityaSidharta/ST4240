from gensim.models.word2vec import Word2Vec
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import string
from string import maketrans 
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from tqdm import tqdm

import sys
stdout = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = stdout

df_train = pd.read_csv('spam.csv', encoding='latin-1')
df_train = df_train.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df_train.columns = ['label', 'message']
array_label = df_train['label'].values
Y_full_train = np.where(array_label == 'spam', 1, 0)
df_train['length'] = df_train['message'].apply(len)
df_messages = df_train['message'].copy()
def text_process(s):
    s = str(s)
    s = s.translate(None, string.punctuation)
    s = re.sub(' +',' ',s)
    s = s.decode('utf-8', 'ignore')
    s = s.lower()
    s = [word for word in s.split() if word.lower() not in stopwords.words('english')]
    return " ".join(s)
df_messages = df_messages.apply(lambda x: text_process(x))


list_tokens = []
for idx in range(len(df_messages)):
    list_tokens.append([str(x) for x in df_messages.iloc[idx].split(' ')])

def w2v_to_features(w2v, list_tokens):
    w2v_keys = w2v.keys()
    n_dims = len(w2v[w2v.keys()[0]])
    result = np.zeros((len(list_tokens), n_dims))
    for idx in tqdm(range(len(list_tokens))):
        array_features = np.array([w2v[x] for x in list_tokens[idx] if x in w2v_keys])
        if len(array_features) == 0:
            continue
        else:
            result[idx] = array_features.mean(axis = 0)
    return result.astype('float64')

with open('glove.6B.50d.txt', "rb") as lines:
    glove_word2vec = {line.split()[0]: np.array(map(float, line.split()[1:]))
               for line in lines}
    
X_full_train = w2v_to_features(glove_word2vec, list_tokens)

joblib.dump(X_full_train, "X_full_train_GLOVE.pkl")