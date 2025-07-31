import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

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

retreiver = vector_store.as_retriever(search_kwargs={"k": 5})


followup_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
If the user's question is *not* about generating a SQL query, formulate it as a standalone question that can be answered by a general LLM without needing database context.
Do NOT answer the question, just reformulate it if needed, otherwise return it as is."""

follow_up_prompt = ChatPromptTemplate.from_messages([
    ("system", followup_prompt), 
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

hr = create_history_aware_retriever (
    llm,
    retreiver, 
    follow_up_prompt
)

sql_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are an expert SQL developer and a versatile database assistant.

**Your Primary Goal:**
To translate natural language questions into accurate SQL queries using provided database schema.
You can also answer direct questions about the schema if the information is available in the context.

**Database Schema (provided by RAG context):**
{context}

**User Question:**
{input}

**Instructions:**
1.  **If the user asks for a SQL query or a script:**
    * Generate a complete, valid, and syntactically correct SQL query that directly answers the user's question, using ONLY the tables and columns explicitly present in the provided schema.
    * Respond ONLY with the SQL query and nothing else (no introductory phrases, explanations, or markdown fences).
2.  **If the user asks a direct question that can be answered from the provided schema context (e.g., "how many databases are on this server?", "what tables are in the Aimsweb database?"):**
    * Analyze the '{context}' and provide a clear, concise, and direct answer in plain language.
    * Do NOT generate a SQL query. The goal is to provide information, not code.
3.  **If the user's question is NOT about generating a SQL query or is a general knowledge question (e.g., "What is the capital of France?", "Tell me a joke"):**
    * Act as a standard, helpful Large Language Model. Respond directly and comprehensively to their non-SQL question.
    * Do NOT generate any SQL or SQL-related placeholders.

**SQL Query (or Standard LLM Response):**"""
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

combined_chain = create_stuff_documents_chain(llm, sql_prompt_template)

rag_chain = create_retrieval_chain(hr, combined_chain)

chat_history = []

print("\nHow may I help you? Type 'exit' to quit.\n")

while True:
    user_query = input("You: ")
    if user_query.lower() == 'exit':
        print("Exiting conversation.")
        break
    response_obj = rag_chain.invoke({"input": user_query, "chat_history": chat_history})
    generated_content = response_obj['answer']
    print(f"Bot: {generated_content}")

chat_history.extend([HumanMessage(content=user_query), AIMessage(content=generated_content)])

