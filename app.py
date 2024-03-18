from dotenv import load_dotenv
import streamlit as st
import chromadb
import os
from PIL import Image
import google.generativeai as genai
from langchain.document_loaders.pdf import PyPDFium2Loader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores.chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chromadb.config import Settings
from langchain.chains.question_answering import load_qa_chain
import hashlib

from langchain_community.document_loaders.word_document import Docx2txtLoader
from io import BytesIO
load_dotenv()
TMP_DIRECTORY="./temp_doc"
VECTOR_DB=Chroma(persist_directory="./chroma_Store", embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
print(VECTOR_DB._collection.count())



def LoadVectorDB(hash,chunked_document):
    store_name=hash
    #print(store_name)
    client=chromadb.PersistentClient(path="./chroma_Store")
    #print(client.list_collections())
    

    
    collections=client.get_or_create_collection(store_name)
    

    

    vector_db=Chroma.from_documents(
        documents=chunked_document,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        collection_name=store_name,
        persist_directory="./chroma_Store/"
    )
    vector_db.persist()
    return vector_db
        



def DocumnetAgent():
    model=ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True)
    chain=load_qa_chain(model,chain_type="stuff")
    return chain

def PdfVectorDB(file):
    hash=hashlib.sha256(file.getvalue()).hexdigest()[:32]
    
    tmp_path=TMP_DIRECTORY+f"/{hash}.pdf"
    with open(tmp_path,"wb") as f:
        f.write(file.getvalue())
    loader=PyPDFium2Loader(tmp_path)
    documents=loader.load()
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_document=text_splitter.split_documents(documents)
    return LoadVectorDB(hash,chunked_document)
    


    
def DocVectorDB(file):
    hash=hashlib.sha256(file.getvalue()).hexdigest()[:32]
    
    tmp_path=TMP_DIRECTORY+f"/{hash}.docx"
    with open(tmp_path,"wb") as f:
        f.write(file.getvalue())
    loader=Docx2txtLoader(tmp_path)
    documents=loader.load()
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=10)
    chunked_document=text_splitter.split_documents(documents)
    return LoadVectorDB(hash,chunked_document)

    

def DocQuery(file,query):
    
    vectordb=DocVectorDB(file)
    agent=DocumnetAgent()
    matched_document=vectordb.similarity_search(query)
    #print(matched_document)
    answer=agent.run(input_documents=matched_document,question=query)
    return answer

def PdfQuery(file,query):
    
    vectordb=PdfVectorDB(file)
    agent=DocumnetAgent()
    matching_docs=vectordb.similarity_search(query)
    print(matching_docs)
    
    answer=agent.run(input_documents=matching_docs, question=query)
    return answer




def main():
    with st.sidebar:
        st.title("Document Chat")
        st.write("Made by Anas Fodkar")
    st.header("File Chat Bot")

    file=st.file_uploader(label="Input a file",type=["jpeg","png","pdf","docx"])
    
    
    if file:
        
        ext=((file.name).split('.'))[-1]
        prompt=st.text_input(label="Enter Prompt")
        if prompt:
            if '.'+ext in ['.doc', '.docm', '.docx', '.dot', '.dotm', '.dotx', '.odt', '.rtf', '.txt', '.wps', '.xml', '.xps']:

                resp=DocQuery(file,prompt)
            else:

                resp=PdfQuery(file,prompt)
        
        
            st.write(resp)
            prompt=None

if __name__=="__main__":
    main()






