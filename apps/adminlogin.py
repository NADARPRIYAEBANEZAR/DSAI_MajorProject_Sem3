# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 00:59:08 2022

@author: Priya
"""
from PIL import Image
import streamlit as st
def app():
    st.markdown("<marquee style='width:100%; direction:left;height:100%;color: black; font-weight: bold;behavior:scroll;'>WELCOME TO THE ADMIN PANEL!!!</marquee>",unsafe_allow_html=True)
    
    from apps import adddoct,viewuser,trainingdata,ViewFeedback,viewQueries
    from multiapp import MultiApp
   
    image = Image.open('./apps/data/admin.png')

    st.sidebar.image(image,width=150)


    
    
    col1, col2, col3,col4,col5= st.columns(5)
    with col5:
                my_expander = st.sidebar.expander(label='LOGIN')
                with my_expander:
                
                    form = st.form("my_form",clear_on_submit=True)
                    username = st.text_input("User Name")
                    password = st.text_input("Password",type='password')
                    c=st.checkbox("Login")
    if c:
        if username=='' or password=='':
            e=st.error('* All Fields are Mandatory')
        else:
            if username == 'Admin' and password=='Admin@123':
                
                        
                    
                       
                            st.success("Logged In as Admin")
                            
                            app = MultiApp()
                            app.add_app("ADD/VIEW DOCTORS",adddoct.app)
                            app.add_app("VIEW USER",viewuser.app)
                            app.add_app("ADD/VIEW TRAINING DATA",trainingdata.app)
                            app.add_app("VIEW FEEDBACK",ViewFeedback.app)
                            app.add_app("VIEW QUERIES",viewQueries.app)
        
        
        # The main app
                            app.run()
                
                    
            else:
                            st.warning("Incorrect Username/Password")
        
    
    
        
    # 	
        
    ####################################################
    


        



	
		
		
			
			
			

				

				
			
				





	

		


