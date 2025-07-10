import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

loader = DirectoryLoader(
    "./JSON",  
    glob="**/*.jsonl",  
    loader_cls=JSONLoader,  
    loader_kwargs={  
        "jq_schema": ".",  
        "text_content": False,
        "json_lines": True  
    }
)
docs = loader.load() 
print("loaded")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200,  
    length_function=len, 
    add_start_index=True 
)
chunks = text_splitter.split_documents(docs)
print("chunked")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

persist_directory = "./chroma_langchain_db"

if os.path.exists(persist_directory) and os.listdir(persist_directory):
    print(f"Loading existing vector store from {persist_directory}")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="SQL_RAG"  
    )
else:
    print("Creating new vector store and embedding documents...")
    vector_store = Chroma.from_documents(
        documents=chunks,  
        embedding=embeddings,  
        collection_name="SQL_RAG",  
        persist_directory=persist_directory  
    )
    print("Embedded using transformer and persisted.")

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

user_query = input("What SQL script do you need? \n")

retreived_docs = vector_store.similarity_search(user_query)
docs_content = "\n\n".join([doc.page_content for doc in retreived_docs])

sql_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are an expert SQL developer and a versatile database assistant.

**Your Primary Goal:**
To translate natural language questions into accurate SQL queries using provided database schema.
If the user's question is *not* about generating a SQL query, act as a helpful and knowledgeable standard Large Language Model and respond appropriately.

**Database Schema (provided by RAG context, if available):**
{context}

**User Question:**
{question}

**Instructions for SQL Generation (if applicable):**
1.  **If the user asks for a SQL query and relevant schema IS provided in '{context}':**
    * Generate a complete, valid, and syntactically correct SQL query that directly answers the user's question, using ONLY the tables and columns explicitly present in the provided schema.
    * Ensure proper SQL syntax (SELECT, FROM, WHERE, JOIN, GROUP BY, ORDER BY, LIMIT, etc.).
    * Do NOT invent tables, columns, or relationships not in the provided schema.
    * Respond ONLY with the SQL query and nothing else (no introductory phrases, explanations, or markdown fences).

2.  **If the user asks for a SQL query but NO relevant schema is provided in '{context}' (i.e., 'context' is empty or irrelevant):**
    * Provide a generic SQL script structure that attempts to fulfill the user's request.
    * Leave specific database, table, or column names as placeholders (e.g., `<your_database>`, `<your_table>`, `<your_column>`) where the information is missing from the context.
    * Include comments within the SQL script to guide the user on what information they need to fill in.
    * Respond ONLY with the generic SQL script and nothing else.

3.  **If the user's question is NOT about generating a SQL query (e.g., "What is the capital of France?", "Tell me a joke"):**
    * Act as a standard, helpful Large Language Model. Respond directly and comprehensively to their non-SQL question.
    * Do NOT generate any SQL or SQL-related placeholders.

**SQL Query (or Standard LLM Response):**"""
        ),
        ("human", "{question}")
    ]
)

prompt = sql_prompt_template.invoke({"question": user_query, "context": docs_content})

llm_output = llm.invoke(prompt)

generated_sql = llm_output.content

print(generated_sql)