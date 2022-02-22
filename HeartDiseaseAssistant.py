# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 20:36:35 2022

@author: Priya
"""
import streamlit as st
from multiapp import MultiApp
import streamlit.components.v1 as components
from apps import intro,adminlogin,logandsign,chatbot,feedback

app = MultiApp()
cust_foot="""
<style>
footer{
        visibility:visible;}
footer:after{
    content:'Credit: Created by : Priya 08. Copyright @ 2022: Heart Disease Assistant';
    display:block;
    position:relative;
    color:blue;
    font-weight: bold;font-style: italic;
    }
<style>
"""

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand"  target="_blank">HEART DISEASE ASSISTANT</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">HOME<span class="sr-only">(current)</span></a>
      </li>
 <li class="nav-item">
        <a class="nav-link" href="https://drive.google.com/file/d/13FzdqVKU3Ol6SaGkbRM154RrxytyHJ3r/view?usp=sharing" target="_blank">SYNOPSIS</a>
      </li>
 <li class="nav-item">
        <a class="nav-link" href="https://drive.google.com/file/d/1rxx63btpp6xVTrX75T7qJd9znWstjpD2/view?usp=sharing" target="_blank">DOCUMENTATION</a>
      </li>
        </ul>
  </div>
</nav>
""", unsafe_allow_html=True)



st.markdown("<h1 style='text-align: center; color: red;font-weight: bold;border:6px groove #0096FF;border-radius: 8px;padding: 5px;'>** HEART DISEASE ASSISTANT **</h1>", unsafe_allow_html=True)    
st.markdown("<h5 style='text-align: center; color: blue; font-style: italic;'>A Healthy Heart is a key to a Healthy Life</h5>", unsafe_allow_html=True) 
         


# Add all your application here
app.add_app("OVERVIEW", intro.app)
app.add_app("ADMIN",adminlogin.app)
app.add_app("USER", logandsign.app)
app.add_app("CHATBOT",chatbot.app)
app.add_app("FEEDBACK", feedback.app)

# The main app
app.run()

st.markdown(cust_foot,unsafe_allow_html=True)
