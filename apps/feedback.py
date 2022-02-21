# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 23:08:45 2022

@author: Priya
"""
import streamlit as st
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import pandas as pd
def app():
    st.markdown("<marquee style='width:100%; direction:left;height:100%;color: black; font-weight: bold;behavior:scroll;'>WELCOME TO THE HEART DISEASE ASSISTANT WEBSITE!!!</marquee>",unsafe_allow_html=True)
    
    image = Image.open('./apps/data/feed.png')

    st.sidebar.image(image,width=250)
    df=pd.read_excel('./data/Feedback.xlsx')
    
    
    
    def isValid(s):
        Pattern = re.compile("(0|91)?[7-9][0-9]{9}")
        return Pattern.match(s)
    def isValide(emails):
        regex = re.compile(r"([-!#-'*+/-9=?A-Z^-~]+(\.[-!#-'*+/-9=?A-Z^-~]+)*|\"([]!#-[^-~ \t]|(\\[\t -~]))+\")@([-!#-'*+/-9=?A-Z^-~]+(\.[-!#-'*+/-9=?A-Z^-~]+)*|\[[\t -Z^-~]*])")
        return regex.fullmatch(emails)
    def isValidname(names):
        regex1 = re.compile("[A-Z]+[a-z]+$")
        return regex1.match(names)
    # title
    st.markdown("<h2 style='text-align: center; color: purple;'>Feedback Form</h2>", unsafe_allow_html=True)    
    
    with st.form('Form1',clear_on_submit=True):
        #f=st.success('We value your feedback. Please complete the following form and help to improve the WebApplication')
        col1, col2, col3 = st.columns(3)
        with col1:
            pass
        
        with col2:
            
            img = Image.open("./data/feedback.jpg") 
            q=st.image(img,caption='We value your feedback. Please complete the following form and help to improve the WebApplication')
        with col3:
            pass
    
        
    
        
        name=st.text_input('Enter your Name')
        mob=st.text_input('Enter your Mobile No.')
        email=st.text_input('Enter your Email Address')
        em=str(email)
        rate= st.radio("How satisfied are you with the usability of the WebApplication?",
         ('Not Satisfied', 'Somewhat Satisfied', 'Satisfied','Very Satisfied','Satisfied enough to tell others'))
        sugg=st.text_area('Do you have any suggestions to make the WebApplication better?')  
        email_from = 'adaminhda@gmail.com'
        password = 'admin123@hda'
    
        subject='Thank You!!!'
    
        body='Thanks for the feedback on your experience with us. We read every feedback we get and we take it very seriously and do improvement on the same.\n\nThanks & Regards,\nAdmin Team'
        message = MIMEMultipart()
        message['From'] = email_from
        message['To'] = em
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))
        text = message.as_string()
        # Now add a submit button to the form:
        c=st.form_submit_button("Submit")
    
    
    if c:
        if (isValid(mob)): 
            a='pass'   
        else :
            a=st.error("* Invalid MobileNumber!!Please Enter Valid MobileNumber") 
        if (isValide(email)):   
             b='pass'   
        else:   
             b=st.error("* Invalid EmailAddress!!Please Enter Valid EmailAddress")
        
        if (isValidname(name)):   
             c='pass'   
        else:   
             c=st.error("* Invalid Name!!Please Enter Valid Name")
        if name=='' or email=='' or mob=='':
            e=st.error('* All Fields are Mandatory')
        else:
            e='pass'
        if a==b==c==e =='pass':
            
            new_data={"Name":name,"MobileNumber":mob,"EmailAddress":email,"Rating":rate,"Suggestions":sugg}
            df=df.append(new_data,ignore_index=True)
            df.to_excel("./data/Feedback.xlsx",index=False)
            mail = smtplib.SMTP('smtp.gmail.com', 587)
            mail.ehlo()
            mail.starttls()
            mail.login(email_from,password)
            mail.sendmail(email_from,em, text)
            mail.close()
            st.success("Your feedback recorded successfully!!!")
        else:
            st.error("Failed to record your feedback...")
          
        st.button("Reset")
       
    
