import os
from dotenv import load_dotenv
<<<<<<< HEAD

# Import Chroma vector store integration from langchain community
from langchain_community.vectorstores import Chroma

# Google Generative AI embeddings and chat models
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain chat model initializer
from langchain.chat_models import init_chat_model

# Prompt templates and placeholders for chat prompts
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Document loaders to load JSON and directory-based documents
from langchain_community.document_loaders import DirectoryLoader, JSONLoader

# Text splitter to chunk documents into manageable pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Chains for retrieval augmented generation
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Message classes representing human and AI chat messages
=======
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
>>>>>>> e25e55cb1eb8c15878216607a18c5a5e25cb526d
from langchain_core.messages import HumanMessage, AIMessage


def initialize_rag():
<<<<<<< HEAD
    # Load environment variables from .env file
    load_dotenv()
    
    # Retrieve the Google API key from environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    # Set up a document loader that recursively loads all .jsonl files from ./JSON directory
    # JSONLoader loads each line as a separate JSON object
    loader = DirectoryLoader(
        "./JSON",
        glob="**/*.jsonl",
        loader_cls=JSONLoader,
        loader_kwargs={
            "jq_schema": ".",           # Use entire JSON object (no filtering)
            "text_content": False,      # Don't treat as plain text
            "json_lines": True          # Each line is a JSON object
        }
    )
    docs = loader.load()  # Load all documents into memory
    print("loaded")

    # Split loaded documents into chunks of max 1000 characters with 200 character overlap
    # This helps improve retrieval quality and context handling
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
=======
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
>>>>>>> e25e55cb1eb8c15878216607a18c5a5e25cb526d
    )
    chunks = text_splitter.split_documents(docs)
    print("chunked")

<<<<<<< HEAD
    # Initialize embeddings using Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    persist_directory = "./chroma_langchain_db"  # Directory to persist vector store data

    # If the vector store directory exists and is not empty, load existing data
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing vector store from {persist_directory}")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="SQL_RAG"
        )
    else:
        # Otherwise create a new vector store by embedding the document chunks
        print("Creating new vector store and embedding documents...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="SQL_RAG",
            persist_directory=persist_directory
        )
        print("Embedded using transformer and persisted.")

    # Initialize the Google Gemini chat model for conversational AI
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

    # Create a retriever interface for vector store to fetch top 5 relevant documents
    retreiver = vector_store.as_retriever(search_kwargs={"k": 5})

    # Define a prompt template to rephrase follow-up questions to standalone questions
    followup_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
If the user's question is *not* about generating a SQL query, formulate it as a standalone question that can be answered by a general LLM without needing database context.
Do NOT answer the question, just reformulate it if needed, otherwise return it as is."""
    
    follow_up_prompt = ChatPromptTemplate.from_messages([
        ("system", followup_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Create a retriever that is aware of chat history to handle context in follow-up questions
    hr = create_history_aware_retriever(
        llm,
        retreiver,
        follow_up_prompt
    )

    # Define the main prompt template for generating SQL queries or responding to questions
    sql_prompt_template = ChatPromptTemplate.from_messages([
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

**SQL Query (or Standard LLM Response):"""
=======
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    persist_directory = "./chroma_langchain_db"

    if os.path.exists(persist_directory) and os.listdir(persist_directory): #place for improvement, update as well not just check if it exists
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

    followup_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\nIf the user's question is *not* about generating a SQL query, formulate it as a standalone question that can be answered by a general LLM without needing database context.\nDo NOT answer the question, just reformulate it if needed, otherwise return it as is."""
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

    # Set up the main SQL prompt
    sql_prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         """You are an expert SQL developer and a versatile database assistant.\n\n**Your Primary Goal:**\nTo translate natural language questions into accurate SQL queries using provided database schema.\nYou can also answer direct questions about the schema if the information is available in the context.\n\n**Database Schema (provided by RAG context):**\n{context}\n\n**User Question:**\n{input}\n\n**Instructions:**\n1.  **If the user asks for a SQL query or a script:**\n    * Generate a complete, valid, and syntactically correct SQL query that directly answers the user's question, using ONLY the tables and columns explicitly present in the provided schema.\n    * Respond ONLY with the SQL query and nothing else (no introductory phrases, explanations, or markdown fences).\n2.  **If the user asks a direct question that can be answered from the provided schema context (e.g., \"how many databases are on this server?\", \"what tables are in the Aimsweb database?\"):**\n    * Analyze the '{context}' and provide a clear, concise, and direct answer in plain language.\n    * Do NOT generate a SQL query. The goal is to provide information, not code.\n3.  **If the user's question is NOT about generating a SQL query or is a general knowledge question (e.g., \"What is the capital of France?\", \"Tell me a joke\"):**\n    * Act as a standard, helpful Large Language Model. Respond directly and comprehensively to their non-SQL question.\n    * Do NOT generate any SQL or SQL-related placeholders.\n\n**SQL Query (or Standard LLM Response):**"""
>>>>>>> e25e55cb1eb8c15878216607a18c5a5e25cb526d
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
<<<<<<< HEAD

    # Create a document chain that stuffs all retrieved documents into the prompt for context
    combined_chain = create_stuff_documents_chain(llm, sql_prompt_template)

    # Combine the retriever and the document chain into one Retrieval Augmented Generation chain
    rag_chain = create_retrieval_chain(hr, combined_chain)

=======
    combined_chain = create_stuff_documents_chain(llm, sql_prompt_template)
    rag_chain = create_retrieval_chain(hr, combined_chain)
>>>>>>> e25e55cb1eb8c15878216607a18c5a5e25cb526d
    return rag_chain


def get_rag_response(user_query, chat_history, rag_chain):
<<<<<<< HEAD
    # Invoke the RAG chain with the current user query and chat history
=======

    # Call the RAG chain with the user query and chat history
>>>>>>> e25e55cb1eb8c15878216607a18c5a5e25cb526d
    response_obj = rag_chain.invoke({
        "input": user_query,
        "chat_history": chat_history
    })
<<<<<<< HEAD

    # Extract the answer from the response
    answer = response_obj['answer']

    # Append the new user question and AI answer to chat history for context in future queries
    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=answer))

    # Return the answer and the updated chat history
    return answer, chat_history


# Allow running this module standalone for quick interactive testing
if __name__ == "__main__":
    rag_chain = initialize_rag()
    chat_history = []

    print("\nHow may I help you? Type 'exit' to quit.\n")

=======
    answer = response_obj['answer']
    # Update chat history with the new question and answer
    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=answer))
    return answer, chat_history

# If you want to test this file directly, you can add a __main__ block here
if __name__ == "__main__":
    rag_chain = initialize_rag()
    chat_history = []
    print("\nHow may I help you? Type 'exit' to quit.\n")
>>>>>>> e25e55cb1eb8c15878216607a18c5a5e25cb526d
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Exiting conversation.")
            break
<<<<<<< HEAD

        answer, chat_history = get_rag_response(user_query, chat_history, rag_chain)
        print(f"Bot: {answer}")
=======
        answer, chat_history = get_rag_response(user_query, chat_history, rag_chain)
        print(f"Bot: {answer}")

  
>>>>>>> e25e55cb1eb8c15878216607a18c5a5e25cb526d
