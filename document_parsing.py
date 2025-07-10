import document_parsing
import vector_db_store
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

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
    vector_db_store.vector_embedding(chunks)

