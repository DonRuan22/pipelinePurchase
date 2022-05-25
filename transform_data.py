# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:23:20 2022

@author: Donruan
"""
#import libraries
from datetime import datetime, timedelta,date
import pandas as pd
import numpy as np


#import the csv
#tx_data = pd.read_csv('OnlineRetail.csv', encoding= 'unicode_escape')
tx_data = pd.read_csv('gcs://don-onlineretail/OnlineRetail.csv',
                 storage_options={"token": "cloud"}, encoding= 'unicode_escape')

#print first 10 rows
tx_data.head(30)

#convert date field from string to datetime
tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])

#create dataframe with uk data only
tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)
