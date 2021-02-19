#Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import plotly.express as px
from sklearn.pipeline import make_pipeline
import category_encoders as ce
from joblib import dump
import pathlib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import warnings
warnings.simplefilter('ignore')

#train=pd.read_csv('C:\\Users\\Anuvrat Shukla\\Desktop\\Guided Hackathon\\live2\\train.csv')

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()
train = pd.read_csv(DATA_PATH.joinpath("train.csv"))


ID_COL, TARGET_COL = 'video_id', 'likes'

num_cols=['views','dislikes','comment_count']
cat_cols=['category_id','country_code']
date_cols=['publish_date']
text_cols=['title','channel_title','tags','description']

features = [c for c in train.columns if c not in [ID_COL, TARGET_COL]]
#print("Features : {}".format(features))

#Droppping unwanted columns
train = train.drop(['title','channel_title','tags','description','publish_date'],axis=1)

#Categorical + Numerical Features
cat_num_cols = [c for c in features if c not in text_cols + date_cols]
print("cat_num_cols: {}".format(cat_num_cols))

#Advanced Gradient Boosting
#1) Early Stopping: Stopping the model training, when the model starts to overfit
#2) HyperParameter Tuning

encoder = ce.OneHotEncoder(cols = 'country_code',use_cat_names=True)

# Helper Fn.
def run_gradient_boosting(clf, fit_params, train, features):
    encoder = ce.OneHotEncoder(cols='country_code', use_cat_names=True)
    train = encoder.fit_transform(train)
    # print(train.head(1))

    features = [c for c in train.columns if c not in [ID_COL, TARGET_COL]]
    print("Features after encoding: \n {}".format(features))

    N_SPLITS = 5
    oofs = np.zeros(len(train))
    # preds = np.zeros((len(test_proc)))

    target = train[TARGET_COL]

    folds = StratifiedKFold(n_splits=N_SPLITS)
    stratified_target = pd.qcut(train[TARGET_COL], 10, labels=False, duplicates='drop')

    feature_importances = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, stratified_target)):
        print(f'\n------------- Fold {fold_ + 1} -------------')

        ### Training Set
        X_trn, y_trn = train[features].iloc[trn_idx], target.iloc[trn_idx]

        ### Validation Set
        X_val, y_val = train[features].iloc[val_idx], target.iloc[val_idx]

        ### Test Set
        # X_test = test[features]

        ############# Pipeline Fit-Dump ################
        # pipeline = make_pipeline(
        #   ce.OneHotEncoder(use_cat_names=True),
        #   clf)

        _ = clf.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], **fit_params)
        # dump(pipeline, 'pipeline.joblib')

        fold_importance = pd.DataFrame({'fold': fold_ + 1, 'feature': features, 'importance': clf.feature_importances_})
        feature_importances = pd.concat([feature_importances, fold_importance], axis=0)

        ############# Pipeline Predict ################
        preds_val = clf.predict(X_val)
        # preds_test = pipeline.predict(X_test)

        fold_score = np.sqrt(mean_squared_error(y_val, preds_val))
        fold_model_fitment = round((r2_score(y_val, preds_val)) * 100)

        print(f'\nRMSE for validation set is {fold_score}')
        print(f'Model Fitment for validation set is {fold_model_fitment} %')

        oofs[val_idx] = preds_val
        # preds += preds_test / N_SPLITS

    oofs_score = np.sqrt(mean_squared_error(target, oofs))
    oofs_model_fitment = round((r2_score(target, oofs)) * 100)

    print(f'\n\n------------- Overall  -------------')
    print(f' So, RMSE for oofs is {oofs_score}')
    print(f' & Overall Model Fitment for oofs is {oofs_model_fitment} %')

    # Feature importance
    feature_importances = feature_importances.reset_index(drop=True)
    fi = feature_importances.groupby('feature')['importance'].mean().sort_values(ascending=False)[:20][::-1]
    fi.plot(kind='barh', figsize=(12, 6))

    return oofs, fi

#Calling Fn.
clf = XGBRegressor(n_estimators = 1000,
                    max_depth = 6,
                    learning_rate = 0.01,
                    colsample_bytree = 0.5,
                    random_state=1452,
                    )
fit_params = {'verbose': 200, 'early_stopping_rounds': 200, 'eval_metric': 'rmse'}

lgb_oofs, fi = run_gradient_boosting(clf, fit_params, train, cat_num_cols)

#Graph
fig = fi.plot(kind = 'barh', figsize=(12, 6))
print(fig)