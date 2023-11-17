import streamlit as st
import torch
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space 
from PyPDF2 import PdfReader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader


with st.sidebar:
    
    st.title('🤖💬 OpenAI Chatbot')
    st.write('💬 Nowcerts Fresh Desk Help!')
    st.write('Hello! how can I help you?')

def main():
    st.header("Chat with PDF 💬")

    load_dotenv()

    loader = DirectoryLoader('Knowledge', glob="./*.pdf",  loader_cls=PyPDFLoader)
    docs = loader.load()
    #st.write(docs)
    text = ""
    for filename in os.listdir('Knowledge'):
        if filename.endswith(".pdf"):
            pdf_file_path = os.path.join('Knowledge', filename)
            with open(pdf_file_path, 'rb') as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                #text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    )
    chunks = text_splitter.split_documents(docs)

    st.write(chunks)

    store_name = "Data"
    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_documents(docs, embeddings)
    #with open(f"{store_name}.pkl", "wb") as f:
        #pickle.dump(VectorStore, f)
    VectorStore.save_local(folder_path='Storage')

    if "messages" not in st.session_state:
        st.session_state.messages = []
            

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if prompt := st.chat_input("Say something.."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        
        
    if prompt:
        docs = VectorStore.similarity_search(query=prompt, k=8)
            #st.write(docs)

        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response =chain.run(input_documents=docs, question=prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
                   

if __name__ == '__main__':
    main()