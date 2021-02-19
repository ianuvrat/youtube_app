#Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.pipeline import make_pipeline
import category_encoders as ce
from joblib import dump
import pathlib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.simplefilter('ignore')

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()
train = pd.read_csv(DATA_PATH.joinpath("train.csv"))

#train=pd.read_csv('\data\\train.csv')
# test=pd.read_csv('C:\\Users\\Anuvrat Shukla\\Desktop\\Guided Hackathon\\live2\\test.csv',parse_dates=[4])
# ss=pd.read_csv('C:\\Users\\Anuvrat Shukla\\Desktop\\Guided Hackathon\\live2\\sample_submission.csv')

print("Train shape: {}".format(train.shape))
#print("Test shape: {}".format(test.shape))

train.nunique()

#Dataset
train.head(1)

ID_COL, TARGET_COL = 'video_id', 'likes'

num_cols=['views','dislikes','comment_count']
cat_cols=['category_id','country_code']
date_cols=['publish_date']
text_cols=['title','channel_title','tags','description']

features = [c for c in train.columns if c not in [ID_COL, TARGET_COL]]

print("Features : {}".format(features))

#Droppping unwanted columns
train = train.drop(['title','channel_title','tags','description','publish_date'],axis=1)

train.head(1)

#Categorical + Numerical Features
cat_num_cols = [c for c in features if c not in text_cols + date_cols]
print("cat_num_cols: {}".format(cat_num_cols))

#Pearson Corellation
plt.figure(figsize=(15,10))
sns.heatmap(train.corr(),annot=True,cmap=plt.cm.CMRmap_r,linewidths=2)

train.drop('likes', axis=1).corrwith(train.likes).plot(kind='bar', grid=True, figsize=(12, 8),
                                                   title="Correlation with likes")


#Validation Strategy
def run_clf_kfold(clf, train, features):
    N_SPLITS = 5  # Divide in 5 split (K)

    oofs = np.zeros(len(train))               # train prediction
    # preds = np.zeros((len(test)))           #test prediction

    target = train[TARGET_COL]

    folds = StratifiedKFold(n_splits=N_SPLITS)
    stratified_target = pd.qcut(train[TARGET_COL], 10, labels=False, duplicates='drop')  # splitting target in 10 parts

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, stratified_target)):
        print(f'\n------------- Fold {fold_ + 1} -------------')

        ############# Get train, validation and test sets along with targets ################

        ### Training Set
        X_trn, y_trn = train[features].iloc[trn_idx], target.iloc[trn_idx]
        print("X_train: {}".format(X_trn.shape))
        print("y_train: {}".format(y_trn.shape))

        ### Validation Set
        X_val, y_val = train[features].iloc[val_idx], target.iloc[val_idx]
        print("X_validation: {}".format(X_val.shape))
        print("y_validation: {}".format(y_val.shape))

        ### Test Set
        # X_test = test[features]

        ############# Pipeline Fit-Dump ################
        pipeline = make_pipeline(
            ce.OneHotEncoder(use_cat_names=True),
            clf)

        pipeline.fit(X_trn, y_trn)
        dump(pipeline, 'pipeline_xg.joblib')

        ############# Pipeline Predict ################
        preds_val = pipeline.predict(X_val)
        # preds_test = clf.predict(X_test)

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

    #print(oofs)
    return oofs

clf = XGBRegressor(n_estimators=200, n_jobs=-1)
xgb_oofs = run_clf_kfold(clf, train, cat_num_cols)
