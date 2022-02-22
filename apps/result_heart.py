# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 01:11:18 2021

@author: Priya
"""
# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# imports
import pandas as pd
# matplotlib 
import matplotlib.pyplot as plt
# sns
import seaborn as sns
# plotly ex
import plotly.express as px
# import streamlit
import streamlit as st

# import Image from pillow to open images
from PIL import Image

def app():
    ################################
    # model
    ################################
    
    # load model
    #print("\n*** Load Model ***")
    import pickle
    filename = './apps/data/heart-model.pkl'
    model = pickle.load(open(filename, 'rb'))
    #print(model)
    #print("Done ...")
    
    # load vars
    #print("\n*** Load Vars ***")
    filename = './apps/data/heart-dvars.pkl'
    dVars = pickle.load(open(filename, 'rb'))
    #print(dVars)
    clsVars = dVars['clsvars'] 
    allCols = dVars['allCols']
    
    #print("Done ...")
    
    ################################
    # predict
    #######N#######################
    
    def getPredict(dfp):
        global clsVars, allCols
        X_pred = dfp[allCols].values
        p_pred = model.predict(X_pred)
        dfp['Predict'] = p_pred
        return (dfp)
    
    ########################################################
    # headers
    ########################################################
    
    
    # title
    st.markdown("<h2 style='text-align: center; color: purple;'>Heart Disease Prediction - Online</h2>", unsafe_allow_html=True)    
    
    
    with st.sidebar.expander("Expand to Enter the value of Heart Features"):
    	# title
        #st.title("Enter the value of Heart Features")
    
        #user inputs
        a=st.number_input("Age: ",1,100,34,1)
        b = st.radio("Sex: '0-Female & 1-Male'",('0','1'))
    #      "Sex: '0-Female & 1-Male'",
    #      ('0', '1'))
        b=int(b)
        c=st.radio("Chest Pain Type: '1-Asympt,2-Atypical,3-Non & 4-Typical'",
         
                   ('1','2','3','4'))
       
        c=int(c)
        #st.write('<br></br>', unsafe_allow_html=True)
        d=st.number_input("Resting Blood Pressure",80,200,94,1)
        e=st.number_input("Cholesterol",40,600,126,1)
        f=st.radio(
          "Fasting Blood Sugar > 120 mg/dl : '0-False & 1-True '",
        ('0', '1'))
        f=int(f)
        
        g = st.radio("Resting electrocardiographic Result: '0-Normal,1-having ST-T & 2-Hypertrophy'",
         ('0','1','2'))
        g=int(g)
        h=st.number_input("Max. Heart Rate Achieved",60,202,126,1)
        i=st.radio("Exercise Induced Angina: '0-No & 1-Yes'",('0','1'))
        i=int(i)
        j=st.number_input("Exercise induced ST segment",0.0,8.0,2.0,0.1)
        k=st.radio("Slope of the peak exercise ST segment: '1-Upsloping,2-Flat & 3-Downsloping'",('1','2','3'))
        k=int(k)
        l=st.radio(
            "Slope of the peak exercise ST segment: '0-Zero,1-One,2-Two & 3-Three'",
         ('0', '1','2','3'))
        l=int(l)
        m=st.radio(
          "Thalassemia: '3-Normal,6-Fixed Defect & 7-Reversable Defect'",
         ('3', '6','7'))
        m=int(m)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        n=st.button("Analyze Heart")
    
    # # submit
    if n:
    #     # create data dict ... colNames should be kay
            data = {'age':a,
                'sex' : b,
                'cp': c,
                'trestbps' : d,
                'chol': e,
                'fbs' : f,
                'restecg': g,
                'thalach' : h,
                'exang':i,
                'oldpeak' : j,
                'slope': k,
                'ca' : l,
                'thal': m}
            
            # create data frame for predict
            dfp = pd.DataFrame(data, index=[0])
            #predict
            dfp = model.predict(dfp)
    
           
                # show dataframe
            st.subheader('Result')
                #st.write('Target \n1-Yes 0-No : ', dfp['Predict'][0])
            if(dfp[0]==0):
                    img = Image.open("./apps/data/noheart.jpg")
                    # display image using streamlit
                    # width is used to set the width of an image
                    st.image(img, width=500)
            else:
                    img = Image.open("./apps/data/heartdis.jpg")
                    # display image using streamlit
                    # width is used to set the width of an image
                    st.image(img, width=500)
                # reset    
            st.button("Reset")
