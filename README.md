# Test_Book_RAG

from langchain.memory import ConversationSummaryBufferMemory
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
import streamlit as st
import os

st.title("TEXTBOOK RAG ")

llm=init_chat_model(model="gemini-2.0-flash-lite",model_provider="google_vertexai")


embeddings = VertexAIEmbeddings(model_name="text-embedding-005")


memory = ConversationSummaryBufferMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)


loader = PyPDFLoader("C:/Users/sai thumma/Downloads/8th.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    is_separator_regex=True
)
chunks = splitter.split_documents(docs)

if os.path.exists("./newdb/index"):
    vector_store = Chroma(persist_directory="./newdb", embedding_function=embeddings)
else:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./newdb"
    )


query = st.text_input("Ask a question about the textbook 📖")
if query:
    retriever = vector_store.as_retriever()
    res_docs = retriever.invoke(query)

    chat_history = memory.load_memory_variables({})["chat_history"]

    prompt = PromptTemplate(
        template="""
You are a helpful teacher. Answer the following question in detail using the textbook context. if you don't know the answer simply say i dont know
Context: {context}
Question: {question}
Chat history: {chat_history}
""",
        input_variables=["context", "question", "chat_history"]
    )

    chain = prompt | llm

    context = "\n".join(d.page_content for d in res_docs)

    answer = chain.invoke({
    "context": context,
    "question": query,
    "chat_history": chat_history
})


    output_text = answer.content if hasattr(answer, "content") else str(answer)

    memory.save_context({"input": query}, {"output": output_text})

    st.write(output_text)


