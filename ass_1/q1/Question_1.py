
import math
import datetime
import matplotlib.pyplot as plt
import operator
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas_summary import DataFrameSummary
from isoweek import Week
from tqdm import tqdm
from sklearn.externals import joblib
from keras.models import load_model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Activation
from keras.models import Model
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

df_merge_basic_feat = joblib.load('df_merge_basic_feat.pkl')
dict_store = joblib.load('dict_store.pkl')
df_merge_adv_feat = joblib.load('df_merge_adv_feat.pkl')
X_train = joblib.load('X_train.pkl')
Y_train = joblib.load('Y_train.pkl')
X_test = joblib.load('X_test.pkl')
column_names = joblib.load('column_names.pkl')

inputs = Input(shape = (X_train.shape[1],), name='Input')
X = Dense(512, name='Dense1', activation='relu')(inputs)
X = BatchNormalization(name='BN1')(X)
X = Dropout(rate= 0.20)(X)
X = Dense(256, name='Dense2', activation='relu')(X)
X = BatchNormalization(name='BN2')(X)
X = Dropout(rate= 0.10)(X)
X = Dense(128, name='Dense3', activation='relu')(X)
X = BatchNormalization(name='BN3')(X)
X = Dropout(rate= 0.05)(X)
X = Dense(1, name='Dense4', activation='relu')(X)
NN_model = Model(inputs=inputs, outputs=X, name='NN_Model')

NN_model.compile('adam', 'mean_squared_error' , metrics=['accuracy'])

NN_model.fit(X_train, Y_train, epochs=1, batch_size=32)

NN_model.save('NN_model.h5') 

xgb_model = XGBRegressor(silent = False, n_jobs = 8)
xgb_params = {
    'learning_rate' : np.arange(0.01, 0.21, 0.01),
    'max_depth' : np.arange(3, 25, 1),
    'gamma' : np.arange(0., 3., 1.),
    "min_child_weight" : np.arange(1, 6, 1),
    'colsample_bytree' : np.arange(0.7, 1.01 ,0.1)
}
xgb_CV_model = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_params, n_iter= 2, scoring='neg_mean_squared_error', cv=3, verbose=4, n_jobs = 1,  return_train_score = True).fit(X_train, Y_train)

joblib_dump(xgb_CV_model, 'xgb_CV_model.pkl')
