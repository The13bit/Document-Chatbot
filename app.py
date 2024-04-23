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
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.word_document import Docx2txtLoader
from io import BytesIO
from langchain.prompts import PromptTemplate
load_dotenv()
TMP_DIRECTORY="./temp_doc"
EMBEDDINGS=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
DBSTORE="./chroma_Store"


def LoadVectorDB(hash,chunked_document):
    store_name=hash
    #print(store_name)
    client=chromadb.PersistentClient(path="./chroma_Store")
    #print(client.list_collections())
    

    
    client.get_or_create_collection(store_name)
    
    

    

    vector_db=Chroma.from_documents(
        documents=chunked_document,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        collection_name=store_name,
        persist_directory="./chroma_Store/"
    )
    
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
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_document=text_splitter.split_documents(documents)
    vd=LoadVectorDB(hash,chunked_document)
    return vd
    
    


    
def DocVectorDB(file):
    hash=hashlib.sha256(file.getvalue()).hexdigest()[:32]
    
    tmp_path=TMP_DIRECTORY+f"/{hash}.docx"
    with open(tmp_path,"wb") as f:
        f.write(file.getvalue())
    loader=Docx2txtLoader(tmp_path)
    documents=loader.load()
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunked_document=text_splitter.split_documents(documents)
    return LoadVectorDB(hash,chunked_document)

    



def PdfQuery(query,vec):
    
    vec.persist()
    
    #vec=Chroma(persist_directory=DBSTORE,embedding_function=EMBEDDINGS)
    print(vec.get())

    retriver=vec.as_retriever()
    docs=retriver.get_relevant_documents(query)
    print(docs)
    qa_chain =RetrievalQA.from_chain_type(llm=ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True), 
                                  chain_type="stuff", 
                                  retriever=retriver, 
                                  return_source_documents=True)

    
    #vectordb=PdfVectorDB(file)
    
    return qa_chain(query)




def main():
    with st.sidebar:
        st.title("Document Chat")
        st.write("Made by Anas Fodkar")
    st.header("File Chat Bot")

    file=st.file_uploader(label="Input a file",type=["jpeg","png","pdf","docx","odt"])
    
    
    if file:
        
        ext=((file.name).split('.'))[-1]
        
        if st.button(label="Process"):
            
            if '.'+ext in ['.doc', '.docm', '.docx', '.dot', '.dotm', '.dotx', '.odt', '.rtf', '.txt', '.wps', '.xml', '.xps']:
               if "vecdb" not in st.session_state:
                    st.session_state.vecdb = DocVectorDB(file)
            else:

                if "vecdb" not in st.session_state:
                    st.write("PRocessing")
                    st.session_state.vecdb = PdfVectorDB(file)
        if "vecdb" in st.session_state:
            vecdb = st.session_state.vecdb
        else:
            vecdb = None  # or some other default value

        prompt=st.text_input(label="Enter Query")
        if prompt and vecdb is not None:
            resp=PdfQuery(prompt,vecdb)
            st.write(resp["result"])
            prompt=None

if __name__=="__main__":
    main()







