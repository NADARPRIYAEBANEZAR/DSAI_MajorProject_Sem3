# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 20:53:53 2022

@author: Priya
"""
import streamlit as st
import base64
import streamlit.components.v1 as components
from PIL import Image
def app():
    
    st.sidebar.markdown("<br>",unsafe_allow_html=True)
    st.sidebar.markdown("<h3 style='text-align: center; color: red; font-weight: bold;font-style: italic;'>The beat of your life goes on only if you have a healthy heart</h3>", unsafe_allow_html=True) 
    
    image = Image.open('./apps/data/heart.jpg')

    st.sidebar.image(image,width=250)
    st.sidebar.markdown("<br>",unsafe_allow_html=True)
    file_ = open("./apps/data/hearthome.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown("<marquee style='width:100%; direction:left;height:100%;color: black; font-weight: bold;behavior:scroll;'>WELCOME TO THE HEART DISEASE ASSISTANT WEBSITE!!!</marquee>",unsafe_allow_html=True)
    
    temp = """

                  
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
</head>
<body>

<div class="container">
  
  <div id="myCarousel" class="carousel slide" data-ride="carousel">
    <!-- Indicators -->
   

    <!-- Wrapper for slides -->
    <div class="carousel-inner">

      <div class="item active">
        <img src="https://d3jlfsfsyc6yvi.cloudfront.net/image/mw:1024/q:85/https%3A%2F%2Fhaygot.s3.amazonaws.com%3A443%2Fcheatsheet%2F12745_ec8594be803c474eadefb3fed47f1e99.jpg"  style="width:100%;">
        
      </div>
       <div class="item">
          <img src="https://media.istockphoto.com/vectors/sad-unhealthy-sick-heart-with-alcohol-bottle-and-smoking-cigarette-vector-id1170990161?k=20&m=1170990161&s=612x612&w=0&h=9atdromOLPpJNYdaBf_UJqTIL9rVLghIVInOi6vga9w=" style="width:100%;">
        
      </div>
      
       
     
    
      <div class="item">
        <img src="https://cbsnews1.cbsistatic.com/hub/i/r/2020/02/11/06dc3d0e-5cc6-45e4-bf9d-b01bfc9b7fd3/thumbnail/1200x630/64f73c0b4fc7b4cb2d420bf0b80b1475/0211-cbsn-heartdiseasewomen-vvk-2025287-640x360.jpg" style="width:100%;">
        
      </div>
       <div class="item">
          <img src="https://health.mil/-/media/Images/MHS/Infographics/Winter-Safety/HeartDiseasePrevention.ashx?h=638&la=en&w=1200&hash=C92FF76B9A714A67F794AEAE470A25190385990A981BF77F91F9BF12CF552118" style="width:100%;">
        
      </div>
      
  <div class="item">
          <img src="https://api.benefits.gov//sites/default/files/2020-08/qRfdcUR - Imgur.jpg" style="width:100%;">
        
      </div>
        <div class="item">
          <img src="https://www.techexplorist.com/wp-content/uploads/2018/04/heart-life.jpg" style="width:100%;">
        
      </div>
        
      
    </div>

   
  </div>
</div>

			"""
    components.html(temp,height=350)
        
    st.sidebar.markdown(f'<img src="data:image/gif;base64,{data_url}"  width=250 height=250>',unsafe_allow_html=True,
            )
    st.markdown("<h5 style='text-align: center; color:green; font-weight: bold;font-style: italic;'>“It’s never too late to take your heart health seriously and make it a priority.”<br>&ensp;&ensp; &ensp;&ensp;— Jennie Garth</h5>", unsafe_allow_html=True) 
    
    col1, col2= st.columns(2)

    with col1:
        st.markdown("<br><br>",unsafe_allow_html=True)
        image = Image.open('./apps/data/images.jpeg')

        st.image(image,width=250)
        st.markdown("<br>",unsafe_allow_html=True)
        image = Image.open('./apps/data/save.jpeg')
        

        st.image(image,width=300)
        
    with col2:
        st.markdown("<h3 style='text-align: center; color: purple;'>OVERVIEW</h3>", unsafe_allow_html=True) 
        st.write('Healthcare sectors has huge amount of data that contains hidden information. This information supports decision making process on related area. In this project, it describes about various approaches of machine learning techniques which are useful for predicting the heart disease. One of the complex tasks in healthcare sectors is predicting of heart disease and it requires more experience and knowledge. Some of the ways of predicting heart diseases are ECG, stress test and heart MRI etc. Here the system uses 14 parameters for predicting the heart disease which includes blood pressure, cholesterol, chest pain, heart rate, etc. These parameters are used to improve the accuracy level. And last but not the least, the main aim of the project is to provide an analysis of machine learning techinques on diagnosing heart disease.')
    
    
        
    st.markdown("<h5 style='text-align: center; color: black;font-weight: bold;font-style: italic;'>“BEAT the HEART DISEASE And feel the HEALTHY BEAT”</h5>", unsafe_allow_html=True)
    footer_temp = """

	 <!-- CSS  -->
	  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
	  <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	  <link href="static/css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
	   <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">


	 <footer class="page-footer grey darken-4">
	    <div class="container" id="aboutapp">
	      <div class="row">
	        <div class="col l6 s12">
	          <h5 class="white-text">CONTACT US</h5>
	          <p class="grey-text text-lighten-4">Address : ABC Building, 361 Strawberry Road, Mumabi-22</p>
               <p class="grey-text text-lighten-4">Mobile : (+91) 9991288867</p>
                <p class="grey-text text-lighten-4">Mail To : adaminhda@gmail.com</p>


	        </div>
	      
	   <div class="col l3 s12">
	          <h5 class="white-text">Connect With Me</h5>
	          <ul>
	           
	           <a href="https://github.com/Jcharis/" target="_blank" class="white-text">
	            <i class="fab fa-github-square fa-4x"></i>
	          </a>
	          </ul>
	        </div>
	      </div>
	    </div>
	    
	  </footer>

	"""
    components.html(footer_temp,height=300)
    
    
        

       
    
        