import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def create_medical_db():
    DATA_PATH = "medical_data/"
    DB_PATH = "./chroma_db_medical"
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        return

    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()
    
    if not docs:
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print("✅ Database Created!")

if __name__ == "__main__":
    create_medical_db()