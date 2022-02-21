# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 00:33:14 2021

@author: Priya
"""
# -*- coding: utf-8 -*-

import streamlit as st
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import warnings
warnings.filterwarnings('ignore')
import base64

def app():
    st.markdown("<marquee style='width:100%; direction:left;height:100%;color: black; font-weight: bold;behavior:scroll;'>WELCOME TO THE HEART DISEASE ASSISTANT WEBSITE!!!</marquee>",unsafe_allow_html=True)
    training_data_quesans = open('./apps/hearttxt.txt',encoding='utf-8').read().splitlines()

    
         
    
    bot = ChatBot(name = 'Hearty', read_only = True, 
             logic_adapters=['chatterbot.logic.MathematicalEvaluation',
        {   
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry! I do not understand you',
            'maximum_similarity_threshold': 0.90
        }
    ])
    ind = 1
    
    
        
    file_ = open("./data/chat.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
        
    st.sidebar.markdown(f'<img src="data:image/gif;base64,{data_url}"  width=280 height=200 >',unsafe_allow_html=True)
    st.sidebar.markdown("<br><br>",unsafe_allow_html=True)
    
    
    def get_text():
        
            
            #global count
            user_response = st.sidebar.text_input("You: ","What's up")
            
            #user_response=input()
            #print('The Value of Count is',count)
            #count=count+1
            return user_response
    
    
    
    
    st.markdown("<h2 style='text-align: center; color: purple;'>So Glad You're Here!!!!</h2>", unsafe_allow_html=True)    
                
                
    st.write("Hello, My name is Hearty. I will answer your queries. If you want to exit, type Bye!")
               
    user_response = get_text()
    if (st.sidebar.button("Let's Talk")):
            
            training_data = training_data_quesans
            trainer2 = ListTrainer(bot) 
            trainer2.train(training_data)
            trainer2 = ListTrainer(bot) 
            
            #word=user_response
    
   
    st.text_area("Hearty:", value=bot.get_response(user_response), height=200, max_chars=None, key=None)
   