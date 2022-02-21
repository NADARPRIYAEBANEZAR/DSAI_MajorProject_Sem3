# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 01:00:26 2021

@author: Priya
"""
import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import io

def app():
    
        menu = ["Add TrainingData","View TrainingData"]
        choice = st.sidebar.selectbox("Menu",menu)
        if choice == "Add TrainingData":
            df=pd.read_csv('./data/trainingdata.csv')
            st.markdown("<h2 style='text-align: center; color: purple;'>Add Training Data</h2>", unsafe_allow_html=True)    
            
            with st.form(key='form1'):
                col1,col2=st.columns(2)
                with col1:
                    age=st.number_input("Age",1,100)
                    restbp=st.number_input("Resting Blood Pressure [mm Hg]",80,200,94,1)
                    chol=st.number_input("Serum Cholesterol [mm/dl]",40,600,126,1)
                    sex = st.radio("Sex",('Male','Female'))
                    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
                    chest = st.radio("ChestPain Type",('TA','ATA','NAP','ASY'))
                    st.markdown("<p style='color: green;'>[TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]</p>", unsafe_allow_html=True)    
                    
                   
                    f=st.radio("Fasting Blood Sugar",('0', '1'))
                    st.markdown("<p style='color: green;'>[1: if FastingBS > 120 mg/dl, 0: otherwise]</p>", unsafe_allow_html=True)    
                    g = st.radio("Resting electrocardiogram results",('Normal','ST','ST-T','LVH'))
                    st.markdown("<p style='color: green;'>[Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]</p>", unsafe_allow_html=True)    
                with col2:
                    h=st.number_input("Max. Heart Rate Achieved",60,202,126,1)
                    j=st.number_input("ST depression induced by exercise relative to rest",0.0,8.0,2.0,0.1)
                    k=st.radio("Peak exercise ST segment",('1','2','3'))
                    st.markdown("<p style='color: green;'>[1 = upsloping, 2 = flat, 3 = downsloping]</p>", unsafe_allow_html=True)    
                   
                    i=st.radio("ExerciseAngina: exercise-induced angina",('Yes','No'))
                    l=st.radio("Number of major vessels (0–3) colored by flourosopy",('0', '1','2','3'))
                    m=st.radio("Thalassemia",('3','6','7'))
                    st.markdown("<p style='color: green;'>[3 = normal, 6 = fixed defect & 7 = reversible defect]</p>", unsafe_allow_html=True)    
                    n=st.radio("HeartDisease",('0','1'))
                    st.markdown("<p style='color: green;'>[1: heart disease, 0: Normal]</p>", unsafe_allow_html=True)    
                    submit_button = st.form_submit_button(label='Add')
                    
            if submit_button:
              
                    st.success("Training Data Added Successfully...")
                    new_data={"age":age,"sex":sex,"cp":chest,"trestbps":restbp,"chol":chol,"fbs":f,"restecg":g,"thalach":h,"exang":i,"oldpeak":j,"slope":k,"ca":l,"thal":m,"target":n}
                    df=df.append(new_data,ignore_index=True)
                    df.to_csv("./data/trainingdata.csv",index=False)
        elif choice == "View TrainingData":
            st.markdown("<h2 style='text-align: center; color: purple;'>View Training Data</h2>", unsafe_allow_html=True)    
            
            
            dt = st.radio("Please Select your choice !!!",
     ('Exploratory Data Analysis', 'Visual Data Analysis'))
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

            if dt == 'Exploratory Data Analysis':
                df=pd.read_csv('./data/trainingdata.csv')
                menu2 = ["Read Dataset","Columns in Dataset","Structure of Dataset","Summary of Dataset","Count of Class Variables"]
                choice2 = st.selectbox("Exploring Heart Disease Patients Dataset",menu2)
                if choice2 == "Read Dataset":
                    
    
                    st.dataframe(df,width=1000,height=1000)
                    @st.cache
                    def convert_df(df):
                       return df.to_csv().encode('utf-8')
                    
                    
                    csv = convert_df(df)
                    
                    st.download_button(
                       "Press to Download",
                       csv,
                       "./data/file.csv",
                       "text/csv",
                       key='download-csv'
                    )
                   
                if choice2 == "Columns in Dataset":
                    cols = st.multiselect('Columns'
                        ,df.columns.tolist(),df.columns.tolist()
                        )
                    
                if choice2 == "Structure of Dataset":
                    df=pd.read_csv('./data/trainingdata.csv')
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    s = buffer.getvalue()

                    st.text(s)
                if choice2 == "Summary of Dataset":
                    st.write(df.describe())
                if choice2 == "Count of Class Variables":
                    clsVars = "target"
                    st.write(df.groupby(df[clsVars]).size())
                    st.info('[1: Heart Disease, 0: Normal]')
            elif dt=='Visual Data Analysis':
                df=pd.read_csv('./data/trainingdata.csv')
                attr_1=df[df['target']==1]

                attr_0=df[df['target']==0]
                menu1 = ["Percentage of Heart Disease Patients","Gender wise Distribution","Age wise Distribution","ChestPainType wise Dstribution","RestECG wise Distribution","STSlope wise Distribution"]
                choice1 = st.selectbox("Visualization of the Heart Disease Patients Dataset",menu1)
                if choice1 == "Percentage of Heart Disease Patients":
                    
                
                    fig, ax1 = plt.subplots(figsize=(14,6))
                    
                
                    ax1 = df['target'].value_counts().plot.pie( x="Heart disease" ,y ='No.of patients', 
                                    autopct = "%1.0f%%",labels=["Heart Disease","Normal"], startangle = 90,ax=ax1);
                    ax1.set(title = 'Percentage of Heart disease patients in Dataset')
                    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                
                
                # plt.show()
                    st.pyplot(fig)
                elif choice1=="Gender wise Distribution":
                   
                    fig1 = plt.figure(figsize=(14,6))
                    
                    sns.countplot(attr_0['sex'])
                    plt.title('GENDER DISTRIBUTION OF NORMAL PATIENTS' )
                   
                    st.pyplot(fig1)
                    fig2 = plt.figure(figsize=(14,6))
                    
                    sns.countplot(attr_1['sex'])
                    plt.title('GENDER DISTRIBUTION OF HEART DISEASE PATIENTS' )
                    
                    st.pyplot(fig2)
                    
                   

                elif choice1=="Age wise Distribution":
                     # plotting normal patients
                    fig3 = plt.figure(figsize=(14,6))
                    #ax1 = plt.subplot2grid((1,2),(0,0))
                    sns.distplot(attr_0['age'])
                    plt.title('AGE DISTRIBUTION OF NORMAL PATIENTS')
                    st.pyplot(fig3)
                    fig4 = plt.figure(figsize=(14,6))
                    #ax1 = plt.subplot2grid((1,2),(0,0))
                    sns.distplot(attr_1['age'])
                    plt.title('AGE DISTRIBUTION OF HEART DISEASE PATIENTS')
                    st.pyplot(fig4)
                elif choice1=="ChestPainType wise Dstribution":
                    # plotting normal patients
                        fig5 = plt.figure(figsize=(14,6))
                        
                        sns.countplot(attr_0['cp'])
                        plt.title('CHEST PAIN OF NORMAL PATIENTS')
                        st.pyplot(fig5)
                        #plotting heart patients
                        
                        fig6 = plt.figure(figsize=(14,6))
                        sns.countplot(attr_1['cp'])
                        plt.title('CHEST PAIN OF HEART PATIENTS')
                        st.pyplot(fig6)

                elif choice1=="RestECG wise Distribution":
                    # plotting normal patients
                    fig7 = plt.figure(figsize=(14,6))
                    
                    sns.countplot(attr_0['restecg'])
                    plt.title('REST ECG OF NORMAL PATIENTS')
                    st.pyplot(fig7)
                    #plotting heart patients
                    fig8 = plt.figure(figsize=(14,6))
                    sns.countplot(attr_1['restecg'])
                    plt.title('REST ECG OF HEART PATIENTS')
                    st.pyplot(fig8)
                elif choice1=="STSlope wise Distribution":
                    # plotting normal patients
                    fig9 = plt.figure(figsize=(14,6))
                    
                    sns.countplot(attr_0['slope'])
                    plt.title('ST SLOPE OF NORMAL PATIENTS')
                    st.pyplot(fig9)
                    #plotting heart patients
                    fig10 = plt.figure(figsize=(14,6))
                    sns.countplot(attr_1['slope'])
                    plt.title('ST SLOPE OF HEART PATIENTS' )
                    st.pyplot(fig10)
                                    
            
            
            

    
    
    
    
    
    
    
    
    
    
