import os
import langchain
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        chunk_overlap = 500,
        length_function = len,
        add_start_index = True
    )
    chunks = text_splitter.split_documents(docs)
    
def vector_embedding():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
def vector_database():
    vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

if __name__ == "__main__":
    main()