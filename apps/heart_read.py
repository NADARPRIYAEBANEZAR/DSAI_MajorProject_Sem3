# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 00:34:36 2021

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
df = pd.read_csv('./data/heart.csv')
print("Done ...")

# load model
print("\n*** Load Model ***")
import pickle
filename = './data/heart-model.pkl'
model = pickle.load(open(filename, 'rb'))
print(model)
print("Done ...")

# load vars
print("\n*** Load Vars ***")
filename = './data/heart-dvars.pkl'
dVars = pickle.load(open(filename, 'rb'))
print(dVars)
clsVars = dVars['clsvars'] 
allCols = dVars['allCols']
print("Done ...")

################################
# Prediction
#######N#######################


################################
# Prediction
#######N#######################

def getPredict(dfp):
    global clsVars, allCols
    X_pred = dfp[allCols].values
    y_pred = dfp[clsVars].values
    # predict from model
    p_pred = model.predict(X_pred)
    # update data frame
    #print("\n*** Update Predict Data ***")
    dfp['Predict'] = p_pred
    #print("Done ...")
    return (dfp)


# read dataset
print("\n*** Read Data For Prediction ***")
dfp = pd.read_csv('./data/heartprd.csv')
print(dfp.head())


# predict
print("\n*** Predict Data ***")
dfp = getPredict(dfp)
print(dfp.head())

# imports
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# accuracy
print("\n*** Accuracy ***")
accuracy = accuracy_score(dfp[clsVars], dfp['Predict'])*100
print(accuracy)

# confusion matrix - actual
cm = confusion_matrix(dfp[clsVars], dfp[clsVars])
print("\n*** Confusion Matrix - Original ***")
print(cm)

# confusion matrix - predicted
cm = confusion_matrix(dfp[clsVars], dfp['Predict'])
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
print("\n*** Classification Report ***")
cr = classification_report(dfp[clsVars], dfp['Predict'])
print(cr)