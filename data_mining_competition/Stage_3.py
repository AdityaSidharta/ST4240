from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
from scipy import sparse
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Activation
from keras.models import Model
from keras import backend as K

sX_train_stage0 = joblib.load('sX_train_stage0.pkl')
sX_test_stage0 = joblib.load('sX_test_stage0.pkl')
print sX_train_stage0.shape
print sX_test_stage0.shape

Y_train_price = joblib.load( 'Y_train_price.pkl')
Y_train_duration = joblib.load('Y_train_duration.pkl')
Y_train_trajlength = joblib.load('Y_train_trajlength.pkl')

from scipy.sparse import coo_matrix, hstack
sX_train_stage0_duration = hstack([sX_train_stage0, Y_train_duration.reshape(-1, 1)])
sX_train_stage0_trajlength = hstack([sX_train_stage0, Y_train_trajlength.reshape(-1, 1)])

print sX_train_stage0_duration.shape
print sX_train_stage0_trajlength.shape
print sX_test_stage0.shape
print Y_train_price.shape
print Y_train_duration.shape
print Y_train_trajlength.shape

non_outlier_index_stage1 = np.array(joblib.load('non_outlier_index_stage1.pkl'))
print non_outlier_index_stage1.shape
print non_outlier_index_stage1

Y_dur_stage2 = joblib.load('Y_dur_stage4.pkl')
Y_traj_stage2 = joblib.load('Y_traj_stage4.pkl')

Y_dur_stage2 = Y_dur_stage2.reshape(-1, 1)
Y_traj_stage2 = Y_traj_stage2.reshape(-1, 1)
sX_test_dur = hstack((sX_test_stage0, Y_dur_stage2))
sX_test_traj = hstack((sX_test_stage0, Y_traj_stage2))

sX_train_stage0_duration = sparse.csc_matrix(sX_train_stage0_duration)
sX_train_stage0_trajlength = sparse.csc_matrix(sX_train_stage0_trajlength)
sX_train_dur = sX_train_stage0_duration[non_outlier_index_stage1]
sX_train_traj = sX_train_stage0_trajlength[non_outlier_index_stage1]

Y_train_pri = Y_train_price[non_outlier_index_stage1]
Y_train_dur = Y_train_duration[non_outlier_index_stage1]
Y_train_traj = Y_train_trajlength[non_outlier_index_stage1]

n_train = sX_train_dur.shape[0]
n_test = sX_test_dur.shape[0]

dtrain_dur = xgb.DMatrix(sX_train_traj, label = Y_train_dur)
dtrain_traj = xgb.DMatrix(sX_train_dur, label = Y_train_traj)
dtest_dur = xgb.DMatrix(sX_test_traj)
dtest_traj = xgb.DMatrix(sX_test_dur)

from sklearn.metrics import make_scorer

def rmpse_loss_func(ground_truth, predictions):
    err = np.sqrt(np.mean((np.true_divide(predictions, ground_truth) - 1.)**2))
    return err

rmpse_loss  = make_scorer(rmpse_loss_func, greater_is_better=False)

def rmpse(preds, dtrain):
    labels = dtrain.get_label()
    err = np.sqrt(np.mean((np.true_divide(preds, labels) - 1.)**2))
    return 'error', err

from sklearn.model_selection import KFold
kfold = KFold(n_splits = 3, shuffle = True, random_state=1234)

param = { 'objective' : "reg:linear", 
          'booster' : "gbtree",
          'eta'                 :0.03, 
          'max_depth'           :12, 
          'colsample_bytree'    : 0.7,
          'subsample' : 0.7,
          'n_thread' : 8
        }

bst_traj = xgb.train(param, dtrain_traj, evals=[(dtrain_traj, 'train')], 
                num_boost_round = 5000, feval= rmpse, maximize = False)

rf_traj = RandomForestRegressor(max_depth = 22, max_features = 'sqrt', n_estimators=1000, verbose = 10, n_jobs = -1, criterion='mse', oob_score = True).fit(sX_train_dur, Y_train_traj)

lm_traj = LassoCV(n_jobs = -1, verbose = 3).fit(sX_train_dur, Y_train_traj)

enet_traj = ElasticNetCV(n_jobs = -1, verbose = 3).fit(sX_train_dur, Y_train_traj)

joblib.dump(bst_traj, 'bst_traj_stage3v2.pkl')
joblib.dump(rf_traj, 'rf_traj_stage3v2.pkl')
joblib.dump(lm_traj, 'lm_traj_stage3v2.pkl')
joblib.dump(enet_traj, 'enet_traj_stage3v2.pkl')
