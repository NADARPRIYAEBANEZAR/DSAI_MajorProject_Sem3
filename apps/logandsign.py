# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 03:35:10 2022

@author: Priya
"""
import streamlit as st
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
from datetime import datetime
import pytz
from apps import result_heart,bmicalculator,viewDoctor,contact
from multiapp import MultiApp
import sqlite3 as my

def app():
    st.markdown("<marquee style='width:100%; direction:left;height:100%;color: black; font-weight: bold;behavior:scroll;'>WELCOME TO THE USER PANEL!!!</marquee>",unsafe_allow_html=True)
    
    col1, col2,col3= st.sidebar.columns(3)

    with col2:
    
        image = Image.open('./apps/data/userlog.png')

        st.sidebar.image(image,width=150)
    conn = my.connect('./apps/data/Usersdb.db')
    c = conn.cursor()
     
    
    
    def create_usertable():
     	c.execute('CREATE TABLE IF NOT EXISTS Userstable(Name TEXT NOT NULL,Address TEXT NOT NULL, Mobile TEXT NOT NULL UNIQUE,Email TEXT NOT NULL UNIQUE,Gender TEXT NOT NULL,Password TEXT NOT NULL,RegisterDate DATE NOT NULL)')
    def add_userdata(name,address,mobile,email,gender,password1,date1):
     	c.execute('INSERT INTO Userstable(Name,Address,Mobile ,Email  ,Gender,Password,RegisterDate ) VALUES (?,?,?,?,?,?,?)',(name,address,mobile,email,gender,password1,date1))
     	conn.commit()
    def login_user(email,password1):
     	c.execute('SELECT * FROM Userstable WHERE Email =? AND Password = ?',(email,password1))
     	data = c.fetchall()
     	return data
    def select_user(email,):
     	c.execute('SELECT Name FROM Userstable WHERE Email =?',(email,))
     	data = c.fetchall()
     	return data
    def view_all_users():
     	c.execute('SELECT * FROM Userstable')
     	data = c.fetchall()
     	return data
    def isValid(s):
        Pattern = re.compile("(0|91)?[7-9][0-9]{9}")
        return Pattern.match(s)
    def isValide(emails):
        regex = re.compile(r"([-!#-'*+/-9=?A-Z^-~]+(\.[-!#-'*+/-9=?A-Z^-~]+)*|\"([]!#-[^-~ \t]|(\\[\t -~]))+\")@([-!#-'*+/-9=?A-Z^-~]+(\.[-!#-'*+/-9=?A-Z^-~]+)*|\[[\t -Z^-~]*])")
        return regex.fullmatch(emails)
    def isValidname(names):
        regex1 = re.compile("[A-Z]+[a-z]+$")
        return regex1.match(names)
    def isValidpass(passwd):
        regex1 = re.compile("^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!#%*?&]{6,20}$")
        return regex1.match(passwd)
    
    
    
    
        #st.title("Simple Login App")
    
    def main():
        
        
        menu = ["Login","SignUp"]
        choice = st.sidebar.selectbox("Menu",menu)
            
        if choice == "Login":
                col1, col2, col3,col4,col5= st.columns(5)
                
                with col5:
                    my_expander = st.sidebar.expander(label='LOGIN')
                    with my_expander:
                   
                        form = st.form("my_form",clear_on_submit=True)
                        userid1 = st.text_input("Email Id")
                        password1 = st.text_input("Password",type='password')
                        ct=st.checkbox("Login")
                    
                
                if ct:
                     if userid1=='' or password1=='':
                         e=st.error('* All Fields are Mandatory')
                     else:
                         create_usertable()
                         result = login_user(userid1,password1)
                         if result:
                            r=select_user(userid1,)
                            
                            list1 = r
                            
                            clist=list1[0]
                            
                            namelog=clist[0]
                            st.success("Logged In as {}".format(namelog))
                            
                            app = MultiApp()
                           
                            app.add_app("ANALYZE HEART",result_heart.app)
                            app.add_app("BMI",bmicalculator.app)
                            app.add_app("View Doctors",viewDoctor.app)
                            app.add_app("Any Queries??",contact.app)
        
        
        # The main app
                            app.run()
                    
                        
                         else:
                            st.warning("Incorrect Username/Password")
        elif choice == "SignUp":
                
                
                #st.subheader("Create New Account")
                st.markdown("<h2 style='text-align: center; color: purple;'>Register Now</h2>", unsafe_allow_html=True)    
                
                with st.form('Form1',clear_on_submit=True):
            #f=st.success('We value your feedback. Please complete the following form and help to improve the WebApplication')
                    
                        
        
            
        
            
                    name=st.text_input('Your Name')
                    address=st.text_area('Your Address')
                    mobile=st.text_input('Your Mobile No.')
                    email=st.text_input('Your Email Address')
                    em=str(email)
                    gender= st.radio("Your Gender",
             ('Male', 'Female'))
                    
                   
                    
                    
                    
                    
                    
                    
                    password1 = st.text_input("Your Password",type='password')
                    # it will get the time zone
                    # of the specified location
                    IST = pytz.timezone('Asia/Kolkata')
                    #date1=datetime.date.today()
                    datetime_ist = datetime.now(IST)
                    date1=datetime_ist.strftime('%Y:%m:%d')

         
                    
            
                    
                    cy=st.form_submit_button("Signup")
                    
                    
                    email_from = 'adaminhda@gmail.com'
                    password = 'admin123@hda'
                    subject='Thanks For Registering with us!!!'
                    body='Your UserId is '+em+' and PassWord is '+password1+'\nYou can use this credential for login.\n\nThanks & Regards,\nAdmin Team'
                    message = MIMEMultipart()
                    message['From'] = email_from
                    message['To'] = em
                    message['Subject'] = subject
                    message.attach(MIMEText(body, 'plain'))
                    text = message.as_string()
                            
                
                
                if cy:
                    if (isValid(mobile)): 
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
                    if (isValidpass(password1)):   
                        d='pass'   
                    else:   
                        d=st.error("* Invalid Password!!Please Enter Valid Password")
                        st.error("Conditions for a valid password are: (1)Should have at least one number. (2)Should have at least one uppercase and one lowercase character. (3)Should have at least one special symbol. (4)Should be between 6 to 20 characters long.")
                    
                    
                    if name=='' or email=='' or mobile=='' or address==''or password1=='':
                        e=st.error('* All Fields are Mandatory')
                    else:
                        e='pass'
                    if a==b==c==e ==d=='pass':
                        
                        try:
                            create_usertable()
                            add_userdata(name,address,mobile,email,gender,password1,date1)
                            mail = smtplib.SMTP('smtp.gmail.com', 587)
                            mail.ehlo()
                            mail.starttls()
                            mail.login(email_from,password)
                            mail.sendmail(email_from,em, text)
                            mail.close()
                            st.success("You have successfully created a valid Account")
                            st.info("Go to Login Menu to login")
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
                    st.button("Reset")
    main()
            
                    
                    
                        
                
          
                
                    
    
    
    
    
    
    
    
    	
    		
    		
    			
    			
    			
    
    				
    
    				
    			
    				
    
    
    
    
    
    	
    
    		
    
