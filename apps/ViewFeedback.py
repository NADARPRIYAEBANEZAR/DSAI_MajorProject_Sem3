# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 02:22:52 2022

@author: Priya
"""
import pandas as pd
import streamlit as st
import plotly.express as px
def app():
#st.set_page_config(page_title='View Feedback')
    st.markdown("<h2 style='text-align: center; color: purple;'>View Feedback Form</h2>", unsafe_allow_html=True)    
            
            
    dt = st.radio("Please Select your choice !!!",
     ('Exploatory Data Analysis', 'Visual Data Analysis'))
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    if dt == 'Exploatory Data Analysis':
    
    ### --- LOAD DATAFRAME
        excel_file = './data/Feedback.xlsx'
        sheet_name = 'Sheet1'
        
        df = pd.read_excel(excel_file,
                           sheet_name=sheet_name,
                           usecols='A:E'
                           )
        st.dataframe(df,1000,1000)
    elif dt == 'Visual Data Analysis':
        excel_file = './data/Feedback.xlsx'
        sheet_name = 'Sheet1'
        df = pd.read_excel(excel_file,
                           sheet_name=sheet_name,
                           usecols='A:E'
                           )
    
        df_grouped = df.groupby('Rating').count().reset_index()
        df_grouped.rename(columns = {'Name':'Person'}, inplace = True)
        
        
        
        
        st.subheader("Ratings")
        # --- PLOT PIE CHART
        pie_chart = px.pie(df_grouped,
                       
                        values='Person',
                       names='Rating')
        
        st.plotly_chart(pie_chart)