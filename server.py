from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import init_chat_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import SQLChatMessageHistory

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
import requests
import json
import os


load_dotenv()
engine = create_engine("sqlite:///chat_history.db")
model = init_chat_model("gemini-2.5-pro", model_provider="google_genai")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)


@tool
def summarize_article(url: str) -> str:
    """Use this tool to summarize article from a given URL which is not a youtube link or a link to video."""
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
    }
    data = {"url": url}
    post_url = f"https://production-sfo.browserless.io/content?token={os.getenv('BROWSERLESS_TOKEN')}"
    response = requests.post(post_url, headers=headers, json=data)
    # print(response)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        if not text.strip():
            raise ValueError("No text found on the webpage.")
        vectorStore = create_docs_and_embed(text)
        return create_summary(vectorStore)
    else:
        raise ValueError(
            f"Failed to retrieve content, status code: {response.status_code}"
        )


def create_map_reduce_chain(docs):
    map_prompt_template = PromptTemplate(
        template="""
                Write a concise summary of the following part of an article or youtube video transcript. 
                If it contains a tutorial or guide, please include and summarize each step as well:
                "{text}"
                SUMMARY:
                """,
        input_variables=["text"],
    )

    combine_prompt_template = PromptTemplate(
        template="""
                Write a summary of the following text, which consists of summaries of parts from a whole article or youtube video transcript.
                Make it coherent and include step by step explanations, here comes the text:
                "{text}"
                SUMMARY:
                """,
        input_variables=["text"],
    )
    chain = load_summarize_chain(
        llm=model,
        chain_type="map_reduce",
        verbose=False,
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
    )

    summary = chain.invoke(docs)
    # print(summary)
    # raise ValueError("Debug: Stopping execution after printing summary.")
    return summary["output_text"]


def create_stuff_summary(docs):
    summary_chain = load_summarize_chain(
        llm=model,
        chain_type="stuff",
        verbose=False,
    )

    summary = summary_chain.invoke(docs)

    # print(summary)
    # raise ValueError("Debug: Stopping execution after printing summary.")
    return summary["output_text"]


def create_summary(vectorStore: FAISS, useStuff: bool = True):
    total_docs = len(vectorStore.index_to_docstore_id)
    docs = vectorStore.similarity_search("", k=total_docs)
    if not docs:
        raise ValueError("No documents available to summarize.")
    # print(docs)
    # docs = vectorStore.similarity_search("summarize the content", k=3)
    if useStuff:
        output = create_stuff_summary(docs)
    else:
        output = create_map_reduce_chain(vectorStore)

    return output


def create_docs_and_embed(text) -> FAISS:
    # Split the text and embed it using  Gemini Embeddings and Store in FAISS
    docs = text_splitter.create_documents([text])
    # print("Documents created:")
    # print(docs)
    # splitted_docs = text_splitter.split_documents(docs)
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorStore = FAISS.from_documents(docs, embeddings)

    return vectorStore


def create_summary_agent():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a world class summarizer, who produces concise but full summaries of any topic based on youtube links or web articles links.
        The user will give you either a youtube video link or a web article link or questions, or a link with an additional question. 
        If it is a link, you decide which tool to use to produce a summary and then you return the summary, and
        also provide helpful and concise answers to questions the user gave you.
        """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    tools = [summarize_article]
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        create_db,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_chat_history


def create_db(session_id: str = "default"):
    return SQLChatMessageHistory(session_id=session_id, connection=engine)


if __name__ == "__main__":
    agent = create_summary_agent()
    # https://python.langchain.com/docs/introduction
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response = agent.invoke(
            {"input": query}, config={"configurable": {"session_id": "default"}}
        )
        print(response["output"])
