# -*- coding: utf-8 -*-
#import machine learning related libraries
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

from google.cloud import bigquery
import gcsfs
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("airflow.task")


bigqueryClient = bigquery.Client()
query_string = """
SELECT
*
FROM `donexp.onlineRetail.onlineRetailTransformed`
"""
tx_class = (
    bigqueryClient.query(query_string)
    .result()
    .to_dataframe(
        # Optionally, explicitly request to use the BigQuery Storage API. As of
        # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
        # API is used by default.
        create_bqstorage_client=True,
    )
)
    

tx_class = pd.get_dummies(tx_class)

#tx_class['NextPurchaseDayRange'] = 2
#tx_class.loc[tx_class.NextPurchaseDay>20,'NextPurchaseDayRange'] = 1
#tx_class.loc[tx_class.NextPurchaseDay>50,'NextPurchaseDayRange'] = 0
    
logger.info('dataframe head - {}'.format(tx_class.describe()))    
logger.info('dataframe head - {}'.format(tx_class.NextPurchaseDayRange))  
logger.info('dataframe head - {}'.format(tx_class.NextPurchaseDay))  


#train & test split
tx_class = tx_class.drop('NextPurchaseDay',axis=1)
X, y = tx_class.drop('NextPurchaseDayRange',axis=1), tx_class.NextPurchaseDayRange
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

kfold = KFold(n_splits=2,shuffle=True, random_state=22)
cv_result = cross_val_score(xgb.XGBClassifier(),X_train,y_train, cv = kfold,scoring = "accuracy")
#print('Xboost '+ cv_result)
logger.info('Xboost - {}'.format(cv_result))
xgb_model = xgb.XGBClassifier(max_depth=5,min_child_weight=5).fit(X_train,y_train)
logger.info("Accuracy of XBClassifier on training dataset:{:.2f}"
      .format(xgb_model.score(X_train,y_train)))
logger.info("Accuracy of XBClassifier on test dataset:{:.2f}"
      .format(xgb_model.score(X_test,y_test)))

filename = 'gcs://don-onlineretail/predict_purchase_model.joblib.pkl'
fs = gcsfs.GCSFileSystem()
with fs.open(filename, 'wb') as f:
    joblib.dump(xgb_model, f, compress=9)
