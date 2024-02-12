from dotenv import find_dotenv,load_dotenv
from transformers import pipeline
import google.generativeai as genai
import os
import streamlit as st
from PIL import Image
import requests
import soundfile as sf

load_dotenv(find_dotenv())
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
HUGGINFACEHUB_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")







def img2text(url):
    image_to_text=pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text=image_to_text(url)
    return text[0]['generated_text']


def generate_story(scenario):
    template="you are a story teller. you can generate a short story based on a simple narrative, the story should be no more than 20 words. CONTEXT:{scenario} STORY:"
    model=genai.GenerativeModel('gemini-pro')

    story=model.generate_content(template)
    return story.text

def text2speech(text):
    # API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-eng"
    # headers = {"Authorization": f"Bearer {HUGGINFACEHUB_API_TOKEN}"}
    # response = requests.post(API_URL, headers=headers,json=text)

    pipe = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    response=pipe(text)
    # print(response['audio'][0])
    # with open('audio.wav','wb') as file:
        # file.write(response.content)
        # file.write(response['audio'],samplerate=response['sampling_rate'])
    sf.write("audio.wav", response['audio'][0], samplerate=response['sampling_rate'])



st.title("story behind a pic")

uploaded_file=st.file_uploader("Upload Your image",type=["jpg", "jpeg", "png"],help="Please uplaod the image")

if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")

submit = st.button("Generate results")

if submit:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        text=img2text(image)
        st.image(image, caption=text, use_column_width=True)
        story=generate_story(text)
        st.subheader("story")
        st.write(story)

        model=pipeline("text-generation", model="bigscience/bloom-560m")
        result=model(text)
        st.subheader("another story")
        st.write(result[0]['generated_text'])

        text2speech(text)
        st.audio("audio.wav")

    else:
        st.write("Please upload the pic")






