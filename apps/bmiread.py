# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 00:42:01 2021

@author: Priya
"""
# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# imports
# pandas 
import pandas as pd
# numpy
import numpy as np
# matplotlib 
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
# sns
import seaborn as sns
# util
import utils


##############################################################
# Read Data 
##############################################################

# read dataset
print("\n*** Read Data ***")
df = pd.read_csv('./data/bmi_data.csv')
print("Done ...")

# load model
print("\n*** Load Model ***")
import pickle
filename = './data/bmi-model.pkl'
model = pickle.load(open(filename, 'rb'))
print(model)
print("Done ...")

# load vars
print("\n*** Load Vars ***")
filename = './data/bmi-dvars.pkl'
dVars = pickle.load(open(filename, 'rb'))
print(dVars)
depVars = dVars['depVars'] 
allCols = dVars['allCols']
print("Done ...")

################################
# Prediction
#######N#######################

def getPredict(dfp):
    global depVars, allCols
    X_pred = dfp[allCols].values
    y_pred = dfp[depVars].values
    p_pred = model.predict(X_pred)
    dfp['Predict'] = p_pred
    return (dfp)



# read dataset
print("\n*** Read Data For Prediction ***")
dfp = pd.read_csv('./data/bmiprd.csv')
print(dfp.head())

##############################################################
# Data Transformation Needs to be done same as for Original Data
##############################################################

# drop cols
# change as required
print("\n*** Drop Cols ***")
dfp = dfp.drop(['Sex'], axis=1)
print("Done ...")

# handle nulls if required
print('\n*** Handle Nulls ***')
colNames = dfp.columns.tolist()
for colName in colNames:
    vmed= min(dfp[colName].mean(),dfp[colName].median())
    dfp[colName] = dfp[colName].fillna(vmed)


# check variance
print('\n*** Variance In Columns ***')
print(dfp.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(dfp.std())

# check mean
print('\n*** Mean In Columns ***')
print(dfp.mean())

# predict
print("\n*** Predict Data ***")
dfp = getPredict(dfp)
print(dfp.head())

