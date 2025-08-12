import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

def initialize_rag_test():
    """
    Test version that only uses smaller files for faster embedding
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    # Only load smaller files for testing
    test_files = [
        "./JSON/summary.json",
        "./JSON/dataTM1.jsonl", 
        "./JSON/sqlProd1-org.jsonl",
        "./JSON/sqlProd1-sf.jsonl"
    ]
    
    docs = []
    for file_path in test_files:
        if os.path.exists(file_path):
            loader = JSONLoader(
                file_path=file_path,
                jq_schema=".",
                text_content=False,
                json_lines=file_path.endswith('.jsonl')
            )
            docs.extend(loader.load())
    
    print(f"Loaded {len(docs)} documents from test files")

    # Use smaller chunks for faster processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks
        chunk_overlap=50,  # Less overlap
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    persist_directory = "./chroma_langchain_db_test"

    # Always create new test store
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)
    
    print("Creating test vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="SQL_RAG_TEST",
        persist_directory=persist_directory
    )
    print("Test vector store created!")

    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Fewer results

    followup_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
If the user's question is *not* about generating a SQL query, formulate it as a standalone question that can be answered by a general LLM without needing database context.
Do NOT answer the question, just reformulate it if needed, otherwise return it as is."""
    
    follow_up_prompt = ChatPromptTemplate.from_messages([
        ("system", followup_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    hr = create_history_aware_retriever(llm, retriever, follow_up_prompt)

    sql_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert SQL developer and a versatile database assistant.

**Your Primary Goal:**
To translate natural language questions into accurate SQL queries using provided database schema.
You can also answer direct questions about the schema if the information is available in the context.

**Database Schema (provided by RAG context):**
{context}

**User Question:**
{input}

**Instructions:**
1. **If the user asks for a SQL query or a script:**
   * Generate a complete, valid, and syntactically correct SQL query that directly answers the user's question, using ONLY the tables and columns explicitly present in the provided schema.
   * Respond ONLY with the SQL query and nothing else (no introductory phrases, explanations, or markdown fences).
2. **If the user asks a direct question that can be answered from the provided schema context:**
   * Analyze the '{context}' and provide a clear, concise, and direct answer in plain language.
   * Do NOT generate a SQL query. The goal is to provide information, not code.
3. **If the user's question is NOT about generating a SQL query or is a general knowledge question:**
   * Act as a standard, helpful Large Language Model. Respond directly and comprehensively to their non-SQL question.
   * Do NOT generate any SQL or SQL-related placeholders.

**SQL Query (or Standard LLM Response):**"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    combined_chain = create_stuff_documents_chain(llm, sql_prompt_template)
    rag_chain = create_retrieval_chain(hr, combined_chain)
    return rag_chain

def get_rag_response_test(user_query, chat_history, rag_chain):
    """
    Test version of the response function
    """
    response_obj = rag_chain.invoke({
        "input": user_query,
        "chat_history": chat_history
    })
    answer = response_obj['answer']
    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=answer))
    return answer, chat_history

if __name__ == "__main__":
    print("Testing RAG system with smaller dataset...")
    rag_chain = initialize_rag_test()
    chat_history = []
    
    print("\nTest RAG system ready! Type 'exit' to quit.\n")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Exiting test.")
            break
        answer, chat_history = get_rag_response_test(user_query, chat_history, rag_chain)
        print(f"Bot: {answer}")
