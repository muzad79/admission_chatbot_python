# from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import os
import streamlit as st
from constants import openai_key
from langchain.chains.question_answering import load_qa_chain
os.environ['OPENAI_API_KEY'] = openai_key

llm = ChatOpenAI(openai_api_key=openai_key, temperature=0.8)

# st.title('Admission Chat Bot')
# input_text = st.text_input("Hi,how may i assist you today")
# # loader = TextLoader("./data.txt")
# # text=loader.load()
# page_content_string = text[0]["page_content"]
# print(page_content_string)

# This is a long document we can split up.
with open("./data.txt") as f:
    text= f.read()
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_text(text)
# print(len(texts))
embeddings = OpenAIEmbeddings()
document_search = FAISS.from_texts(texts,embeddings)
# print(document_search)

# chain = load_qa_chain(llm=llm,chain_type="stuff")
# query =input_text
# docs = document_search.similarity_search(query)
# print(chain.run(input_documents=docs,question=query))

# if input_text:
#     st.write(chain.run(input_documents=docs,question=query))


def admission_chatbot(input_query):
#     llm = ChatOpenAI(openai_api_key=openai_key, temperature=0.8)
#     # This is a long document we can split up.
# with open("./data.txt") as f:
#     text= f.read()
# text_splitter = CharacterTextSplitter(
#     separator="\n\n",
#     chunk_size=800,
#     chunk_overlap=200,
#     length_function=len,
#     is_separator_regex=False,
# )
# texts = text_splitter.split_text(text)
# # print(len(texts))
# embeddings = OpenAIEmbeddings()
# document_search = FAISS.from_texts(texts,embeddings)
# print(document_search)

    chain = load_qa_chain(llm=llm,chain_type="stuff")
    query =input_query
    docs = document_search.similarity_search(query)
    output=chain.run(input_documents=docs,question=query)
    return output