# Q&A Chatbot
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
import os
from dotenv import find_dotenv,load_dotenv

#from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

import streamlit as st
import os


## Function to load OpenAI model and get respones

def get_openai_response(question):
    llm=GoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    response=llm.invoke(question)
    return response

##initialize our streamlit app

st.set_page_config(page_title="Q&A Demo")

st.header("Langchain Application")

input=st.text_input("Input: ")
submit=st.button("Ask the question")





## If ask button is clicked

if submit:
    if input is not None:
        response=get_openai_response(input)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.write("input cannot be empty")