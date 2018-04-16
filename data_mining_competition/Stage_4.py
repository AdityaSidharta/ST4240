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
from tqdm import tqdm
df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test.csv')
df_target = pd.read_csv('stage_3_v2.csv')
pred_log_dur = joblib.load('Y_test_dur_pred_stage3v2.pkl')
pred_log_traj = joblib.load('Y_test_traj_pred_stage3v2.pkl')
pred_dur = np.exp(pred_log_dur)
pred_traj = np.exp(pred_log_traj)
n_test = df_test.shape[0]
non_outlier_index_stage1 = joblib.load('non_outlier_index_stage1.pkl')
df_train = df_train.loc[non_outlier_index_stage1, :]
final_pred_dur = np.zeros(n_test)
final_pred_traj = np.zeros(n_test)
for idx in tqdm(range(len(df_target))):
    x_start, y_start, x_end, y_end = df_test.loc[idx, 'X_START'], df_test.loc[idx, 'Y_START'], \
    df_test.loc[idx, 'X_END'], df_test.loc[idx, 'Y_END']
    df_small = df_train[(abs(abs(df_train.X_START) - abs(x_start)) <= 1) & (abs(abs(df_train.Y_START) - abs(y_start)) <= 1) & (abs(abs(df_train.X_END) - abs(x_end)) <= 1) & (abs(abs(df_train.Y_END) - abs(y_end)) <= 1)]
    md_pred_dur = pred_dur[idx]
    md_pred_traj = pred_traj[idx]
    if df_small.shape[0] == 0:
        final_pred_dur[idx] = md_pred_dur
        final_pred_traj[idx] = md_pred_traj
    elif df_small.shape[0] == 1:
        nb_pred_dur = df_small.DURATION.values[0]
        nb_pred_traj = df_small.TRAJ_LENGTH.values[0]
        final_pred_dur[idx] = (0.2 * nb_pred_dur) + (0.8 * md_pred_dur)
        final_pred_traj[idx] = (0.2 * nb_pred_traj) + (0.8 * md_pred_traj)
    else:
        all_nb_pred_dur = df_small.DURATION.values
        all_nb_pred_traj = df_small.TRAJ_LENGTH.values
        nb_pred_dur = np.median(all_nb_pred_dur)
        nb_pred_traj = np.median(all_nb_pred_traj)
        if np.std(all_nb_pred_dur) > 10.0:
            final_pred_dur[idx] = (0.3 * nb_pred_dur) + (0.7 * md_pred_dur)
        elif np.std(all_nb_pred_dur) > 3.0:
            final_pred_dur[idx] = (0.5 * nb_pred_dur) + (0.5 * md_pred_dur)
        else:
            final_pred_dur[idx] = (0.7 * nb_pred_dur) + (0.3 * md_pred_dur)
            
        if np.std(all_nb_pred_traj) > 10.0:
            final_pred_traj[idx] = (0.3 * nb_pred_traj) + (0.7 * md_pred_traj)
        if np.std(all_nb_pred_traj) > 3.0:
            final_pred_traj[idx] = (0.5 * nb_pred_traj) + (0.5 * md_pred_traj)
        else:
            final_pred_traj[idx] = (0.7 * nb_pred_traj) + (0.3 * md_pred_traj)
joblib.dump(final_pred_dur, 'final_pred_durv1.pkl')
joblib.dump(final_pred_traj, 'final_pred_trajv1.pkl')