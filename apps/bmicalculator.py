# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 00:52:28 2021

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
import streamlit.components.v1 as components
#from streamlit.ScriptRunner import StopException, RerunException
# import Image from pillow to open images
from PIL import Image

################################
# model
################################
def app():
    # load model
    #print("\n*** Load Model ***")
    import pickle
    filename = './apps/data/bmi-model.pkl'
    model = pickle.load(open(filename, 'rb'))
    #print(model)
    #print("Done ...")
    
    # load vars
    #print("\n*** Load Vars ***")
    filename = './apps/data/bmi-dvars.pkl'
    dVars = pickle.load(open(filename, 'rb'))
    #print(dVars)
    depVars = dVars['depVars'] 
    allCols = dVars['allCols']
    #print("Done ...")
    
    ###########################################
    # headers
    ########################################################
    
    # title
    st.markdown("<h2 style='text-align: center; color: purple;'>BMI Calculation</h2>", unsafe_allow_html=True)    
    
    with st.form('Form1'):
        
        temp1="""
<style>
img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
</style>
</head>
<body>



<img src="https://www.bajajfinservmarkets.in/content/dam/bajajfinserv/calculators/bmi/bmi-result-p.png"  style="width:50%;">
"""
        components.html(temp1,height=230)
        
        a=st.number_input('Height(Inches)',13,100,65,1,key='a')
        b=st.number_input('Weight(Pounds)',3,300,112,1,key='b')
        # Now add a submit button to the form:
        c=st.form_submit_button("Submit")
    
    # submit
    if c:
        # create data dict ... colNames should be kay
        data = {'Height(Inches)':a,
                'weight(Pounds)' :b
                }
        # create data frame for predict
        dfp = pd.DataFrame(data, index=[0])
        
        
        # predict
        dfp = model.predict(dfp)
        # show dataframe
        #st.subheader('Prediction')
        #st.write('Your BMI is:', round(dfp[0],2))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pass
        with col3:
            pass
        
        with col2 :
           
            st.subheader('RESULT')
            st.write('Your BMI is:', round(dfp[0],2))
            d=round(dfp[0],2)
            if d<= 18.5:  
                st.write("Oops! You are underweight.")  
            elif d <= 24.9:  
                st.write("Awesome! You are healthy.")  
            elif d <= 29.9:  
                st.write("Eee! You are over weight.")  
            else:  
                st.write("Seesh! You are obese.")  
        # reset    
            st.button("Reset")