# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 00:42:59 2021

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
    df=pd.read_excel('./apps/data/ContactUs.xlsx')
    # Please replace below with your email address and password
   
    
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
    st.markdown("<h2 style='text-align: center; color: purple;'>Have Some Questions??</h2>", unsafe_allow_html=True)    
    
    with st.form('Form1'):
        #f=st.success('We value your feedback. Please complete the following form and help to improve the WebApplication')
        col1, col2, col3 = st.columns(3)
        with col1:
            pass
        
        with col2:
            
            img = Image.open("./apps/data/Contact.jpg") 
            q=st.image(img,caption='We would love to respond to your queries...')
        with col3:
            pass
    
        
    
        
        fname=st.text_input('Enter your FirstName')
        lname=st.text_input('Enter your LastName')
        mob=st.text_input('Enter your Mobile No.')
        email=st.text_input('Enter your Email Address')
        em=str(email)
        
        quest=st.text_area('Your Questions?')  
        # Now add a submit button to the form:
        c=st.form_submit_button("Send Message")
        email_from = 'adaminhda@gmail.com'
        password = 'admin123@hda'
        subject = 'Thank You for Contacting Us!!!'
        body='Your query has been received and We will get back to you soon...\n\nThanks & Regards,\nAdmin Team'
        message = MIMEMultipart()
        message['From'] = email_from
        message['To'] = em
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))
        text = message.as_string()
    
    
  
    
    
    if c:
        if (isValid(mob)): 
            a='pass'   
        else :
            a=st.error("* Invalid MobileNumber!!Please Enter Valid MobileNumber") 
        if (isValide(email)):   
             b='pass'   
        else:   
             b=st.error("* Invalid EmailAddress!!Please Enter Valid EmailAddress")
        
        if (isValidname(fname)):   
             c='pass'   
        else:   
             c=st.error("* Invalid FirstName!!Please Enter Valid Name")
        if (isValidname(lname)):   
             d='pass'   
        else:   
             d=st.error("* Invalid LastName!!Please Enter Valid Name")
        if fname=='' or lname=='' or email=='' or mob=='' or quest=='':
            e=st.error('* All Fields are Mandatory')
        else:
            e='pass'
        if a==b==c==d==e =='pass':
            
            new_data={"FirstName":fname,"LastName":lname,"MobileNumber":mob,"EmailAddress":email,"Questions":quest}
            df=df.append(new_data,ignore_index=True)
            df.to_excel("./apps/data/ContactUs.xlsx",index=False)
            mail = smtplib.SMTP('smtp.gmail.com', 587)
            mail.ehlo()
            mail.starttls()
            mail.login(email_from,password)
            mail.sendmail(email_from,em, text)
            mail.close()
            st.success("Your Message has been received successfully...")
        else:
            st.error("Error Occured while receiving your message...")
          
        st.button("Reset")
       
    
