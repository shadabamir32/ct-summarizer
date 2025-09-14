from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from sqlalchemy import create_engine
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

app = FastAPI(title="Q&A Chatbot", version="1.0.0")
vectorStore=None
load_dotenv()
engine = create_engine("sqlite:///chat_history.db")


# Endpoint to handle chat messages
@app.post("/chat/{session_id}", tags=["Chat"])
def chat(question: str, session_id: str):
    vectorStore = retrive_document_from_web()
    chain = create_chain(vectorStore)

    response = process_chat(chain, question, session_id)
    return JSONResponse(content={"response": response['output']}, status_code=200)


# Retruve all chat history for a session
@app.get("/history/{session_id}", tags=["Chat"])
def get_history(session_id: str):
    history = create_db(session_id)
    messages = history.messages
    formatted_history = [{"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content} for msg in messages]
    return JSONResponse(content={"history": formatted_history}, status_code=200)

# Clear chat history for a session
@app.delete("/history/{session_id}", tags=["Chat"])
def clear_history(session_id: str):
    history = create_db(session_id)
    history.clear()
    return JSONResponse(content={"message": "History cleared"}, status_code=200)

# Create the chat chain with history
def create_chain(vectorStore):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the tools to answer the user's questions only when required."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])

    llm = ChatOllama(
        model="llama3.1",
        temperature=0.7
    )
    
    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="langgraph_search",
        description="Search for information about LangGraph. For any questions about LangGraph, you must use this tool!",
    )
    search_tool = create_search_tool()
    
    tools = [retriever_tool, search_tool]

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        create_db,
        output_messages_key="output",
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_chat_history

# Initialize or retrieve the database for chat history
def create_db(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id, connection=engine
    )

# Process chat input and generate response
def process_chat(chain, question, session_id: str="default"):
    return chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )

#RAG
def retrive_document_from_web():
    # Load the web page
    loader = WebBaseLoader(
        web_path=(
            "https://langchain-ai.github.io/langgraph/",
        )
    )
    data = loader.load()

    if not data:
        raise ValueError(f"No data found.")
    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitedDocument = text_splitter.split_documents(data)
    if not splitedDocument:
        raise ValueError("No text chunks created from document")

    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    vectorStore = FAISS.from_documents(splitedDocument, embeddings)

    return vectorStore

# Create a search tool using DuckDuckGo
def create_search_tool():
    search_tool = DuckDuckGoSearchResults(max_results=2)
    return search_tool


if __name__ == "__main__":
    vectorStore = retrive_document_from_web()
    chain = create_chain(vectorStore)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = process_chat(chain, user_input)
        # print(type(response))
        print(response['output'])
