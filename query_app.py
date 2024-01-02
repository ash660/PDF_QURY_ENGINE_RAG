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
    st.write('💬 Nowcerts Policy Details Help!')
   

def main():
    st.header("Chat with PDF 💬")
    st.text("Ask a question about policy documents:")

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

    #st.write(chunks)

    store_name = "Data"
    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_documents(docs, embeddings)
    #with open(f"{store_name}.pkl", "wb") as f:
        #pickle.dump(VectorStore, f)
    VectorStore.save_local(folder_path='Storage')

    if "messages" not in st.session_state:
        st.session_state.messages = []


    first = "Hi, how can I help you?"
    with st.chat_message("assistant"):
            st.markdown(first)
            #st.session_state.messages.append({"role": "assistant", "content": first})        

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    

    if prompt := st.chat_input("Please ask question related to policy docs.."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        
        
    if prompt:
        docs = VectorStore.similarity_search(query=prompt, k=8)
            #st.write(docs)

        llm = OpenAI(model="gpt-3.5-turbo-instruct")
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response =chain.run(input_documents=docs, question=prompt)
        
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
                   

if __name__ == '__main__':
    main()