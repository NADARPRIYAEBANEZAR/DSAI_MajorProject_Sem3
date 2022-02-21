# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 19:49:28 2021

@author: Priya
"""
# Core Pkgs
import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine
import sqlite3 as my
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def app():
    conn = my.connect('./apps/data/Doctorsdb.db')
    curr = conn.cursor()
    def create_doctor():
     	curr.execute('CREATE TABLE IF NOT EXISTS Doctor(FName TEXT NOT NULL,LName TEXT NOT NULL,Gender TEXT NOT NULL,Age INT NOT NULL,Mobile TEXT NOT NULL UNIQUE,Email TEXT NOT NULL UNIQUE,Address TEXT NOT NULL,Speciality TEXT NOT NULL)')
    def add_doctor(fname,lname,gender,age,mobile,email,address,spec):
     	curr.execute('INSERT INTO Doctor(FName,LName,Gender,Age,Mobile,Email,Address,Speciality) VALUES (?,?,?,?,?,?,?,?)',(fname,lname,gender,age,mobile,email,address,spec))
     	conn.commit()
    
    menu = ["Add Doctors","View Doctors"]
    choice = st.sidebar.selectbox("Menu",menu)
        
    if choice == "Add Doctors":
    
        
        def isValid(s):
            Pattern = re.compile("(0|91)?[7-9][0-9]{9}")
            return Pattern.match(s)
        def isValide(emails):
            regex = re.compile(r"([-!#-'*+/-9=?A-Z^-~]+(\.[-!#-'*+/-9=?A-Z^-~]+)*|\"([]!#-[^-~ \t]|(\\[\t -~]))+\")@([-!#-'*+/-9=?A-Z^-~]+(\.[-!#-'*+/-9=?A-Z^-~]+)*|\[[\t -Z^-~]*])")
            return regex.fullmatch(emails)
        def isValidname(names):
            regex1 = re.compile("[A-Z]+[a-z]+$")
            return regex1.match(names)
        
        
        
        
        st.markdown("<h2 style='text-align: center; color: purple;'>Add Doctor</h2>", unsafe_allow_html=True)    
        with st.form(key='form1',clear_on_submit=True):
            col1,col2=st.columns(2)
            with col1:
                
                
                firstname = st.text_input("First Name")
                mob=st.text_input("Mobile Number")
                
                sex = st.radio("Sex",('Male','Female'))
                add=st.text_area("Address")
                
            with col2:
                lastname = st.text_input("Last Name")
                email=st.text_input("Email Address")
                em=str(email)
                age=st.number_input("Age",25,100)
                spec=st.text_input("Speciality in")
                #st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
                submit_button = st.form_submit_button(label='Add')
                
        if submit_button:
            email_from = 'adaminhda@gmail.com'
            password = 'admin123@hda'
            
            subject='Your Login Id has been created!!!'
            
            body='The login credential details are as follows:UserName : Admin && Password : Admin@123\nPlease use this credential for further login\n\n\nThanks & Regards,\nAdmin Team'
            
            message = MIMEMultipart()
            message['From'] = email_from
            message['To'] = em
            message['Subject'] = subject
            message.attach(MIMEText(body, 'plain'))
            text = message.as_string()
        
            
            if (isValid(mob)): 
                a='pass'   
            else :
                a=st.error("* Invalid MobileNumber!!Please Enter Valid MobileNumber") 
            if (isValide(email)):   
                 b='pass'   
            else:   
                 b=st.error("* Invalid EmailAddress!!Please Enter Valid EmailAddress")
            
            if (isValidname(firstname)):   
                 c='pass'   
            else:   
                 c=st.error("* Invalid FirstName!!Please Enter Valid FirstName")
            if (isValidname(lastname)):   
                 d='pass'   
            else:   
                 d=st.error("* Invalid LastName!!Please Enter Valid LastName")
            if firstname=='' or lastname==''or add=='' or email=='' or mob=='' or spec=='':
                e=st.error('* All Fields are Mandatory')
            else:
                e='pass'
            if a==b==c==d==e =='pass':
                
                try:
                            create_doctor()
                            add_doctor(firstname,lastname,sex,age,mob,email,add,spec)
                            mail = smtplib.SMTP('smtp.gmail.com', 587)
                            mail.ehlo()
                            mail.starttls()
                            mail.login(email_from,password)
                            mail.sendmail(email_from,em, text)
                            mail.close()
                                
                            st.success("Doctor Details Added...")
                except my.DataError as er:
                            st.error("Error occurred... Please try again!!")
                            #st.error(e)
        
                except my.InternalError as er:
                            st.error("Error occurred... Please try again!!")
                            #st.error(e)
        
                except my.IntegrityError as er:
                            st.error("Error occurred... Please try again!!")
                            #st.error(e)
        
                except my.OperationalError as er:
                            st.error("Error occurred... Please try again!!")
                            #st.error(e)
        
                except my.NotSupportedError as e:
                            st.error("Error occurred... Please try again!!")
                            #st.error(e)
        
                except my.ProgrammingError as er:
                            st.error("Error occurred... Please try again!!")
                            #st.error(e)
        
                except :
                            st.error("Error occurred... Please try again!!")
            else:
                st.error("Failed to Add Doctor Details...")
              
            st.button("Reset")
        
        
    elif choice=="View Doctors":
        st.markdown("<h2 style='text-align: center; color: purple;'>Doctors List</h2>", unsafe_allow_html=True)
        db_file = r'./apps/data/Doctorsdb.db'
        engine = create_engine(r"sqlite:///{}" .format(db_file))
        sql = 'SELECT * from  Doctor'
    
        data_df = pd.read_sql(sql, engine)
    
        st.dataframe(data_df,1000,1000)
    
    
        
        
        
    
    
