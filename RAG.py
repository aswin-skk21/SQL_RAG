import os
import chromadb
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
   
def main():
    loadDocs()
    
    
def loadDocs():
    loader = DirectoryLoader("./JSON", glob="**/*.jsonl", loader_cls=JSONLoader, loader_kwargs={
    "jq_schema" : ".",
    "text_content": False,
    "json_lines": True
    })
    docs = loader.load()
    textChunking(docs)

def textChunking(docs): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, 
        chunk_overlap = 200,
        length_function = len,
        add_start_index = True
    )
    chunks = text_splitter.split_documents(docs)
    vector_embedding(chunks)
    
def vector_embedding(chunks):
    api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="SQL_RAG",
    persist_directory="./chroma_langchain_db"
)

if __name__ == "__main__":
    main()