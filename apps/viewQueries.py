# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 04:11:38 2022

@author: Priya
"""
import pandas as pd
import streamlit as st

def app():
#st.set_page_config(page_title='View Feedback')
        st.markdown("<h2 style='text-align: center; color: purple;'>View Queries</h2>", unsafe_allow_html=True)    
            
            
    
    
        excel_file = './apps/data/ContactUs.xlsx'
        sheet_name = 'Sheet1'
        
        df = pd.read_excel(excel_file,
                           sheet_name=sheet_name,
                           usecols='A:E'
                           )
        st.dataframe(df,1000,1000)