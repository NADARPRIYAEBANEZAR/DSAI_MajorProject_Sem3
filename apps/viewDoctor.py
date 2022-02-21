# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:47:03 2022

@author: Priya
"""
import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
def app():


    st.markdown("<h2 style='text-align: center; color: purple;'>Doctors List</h2>", unsafe_allow_html=True)
    
    db_file = r'./apps/data/Doctorsdb.db'
    engine = create_engine(r"sqlite:///{}" .format(db_file))
    sql = 'SELECT * from  Doctor'
    
    data_df = pd.read_sql(sql, engine)
    
    st.dataframe(data_df,1000,1000)
    