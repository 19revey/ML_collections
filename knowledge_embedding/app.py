import streamlit as st
from pypdf import PdfReader


# from langchain.text_splitter import CharacterTextSplitter
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

# from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
# from langchain_community.llms import HuggingFaceHub


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain

from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from dotenv import find_dotenv,load_dotenv
import os

load_dotenv(find_dotenv())

HUGGINGFACEHUB_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # model_name = 'maidalun1020/bce-embedding-base_v1'
    # model_kwargs = {'device': 'cuda'}
    # encode_kwargs = {'batch_size': 64, 'normalize_embeddings': True, 'show_progress_bar': False}
    # embed_model = HuggingFaceEmbeddings(
    #     model_name=model_name,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs
    # )

    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings  = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")


def get_conversation_chain():

    prompt_template = """
    You are a researcher know how to write scientific paper on topics related to STEM. 
    I will share related texts from previous documents with you and you will rewrite the provided texts in an academic language.

    1/ the generated texts should be very similar to the style of the documents, 
    in terms of ton of voice, logical arguments and other details

    2/ If the past texts are irrelevant, then try to mimic the style of the documents to rewrite the paragraph

    Below is the texts I want to rewrite:
    {paragraph}

    Here is a list of previous documents:
    {document}

    Please rewrite the paragraph:
    """

   
    llm=GoogleGenerativeAI(model="gemini-pro", temperature=0.3,google_api_key=GOOGLE_API_KEY)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    prompt=PromptTemplate(template=prompt_template,input_variables=["paragraph","document"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question,k=3)
    chain = get_conversation_chain()

    # response = chain(
    #     {"input_documents":docs, "question": user_question}
    #     , return_only_outputs=True)
    response=chain.run(paragraph=user_question,document=docs)

    print(response)
    st.write("Reply: ", response)




def main():
    st.set_page_config("Help you write academic paper")
    st.header("Rewriter paper in academic language")

    user_question = st.text_input("paragraph to be rephrased")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vectorstore(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()