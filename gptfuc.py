# from langchain.llms import OpenAI
import json
import os
import pickle
from pathlib import Path

import faiss

# from gpt_index import GPTSimpleVectorIndex, LLMPredictor, SimpleDirectoryReader
from langchain.chains import VectorDBQA

# from langchain.chains.question_answering import load_qa_chain
# from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings

# from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAIChat

# from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# from langchain import OpenAI

# import requests
# from llama_index import GPTSimpleVectorIndex, LLMPredictor, SimpleDirectoryReader


# read config from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# get openai api key from config.json
api_key = config["openai_api_key"]

os.environ["OPENAI_API_KEY"] = api_key

uploadfolder = "uploads"
filerawfolder = "fileraw"
fileidxfolder = "fileidx"
backendurl = "http://localhost:8000"

openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key is None:
    print("请设置OPENAI_API_KEY")
else:
    print("已设置OPENAI_API_KEY" + openai_api_key)

# gpt_model="text-davinci-003"
# gpt_model='gpt-3.5-turbo'


# llm_predictor = LLMPredictor(
#     llm=OpenAI(temperature=0, model_name=gpt_model, max_tokens=1024)
# )

# use ChatGPT [beta]
# from gpt_index.langchain_helpers.chatgpt import ChatGPTLLMPredictor

# llm_predictor = ChatGPTLLMPredictor()


# def build_index():
#     documents = SimpleDirectoryReader(filerawfolder, recursive=True).load_data()
#     # index = GPTSimpleVectorIndex(documents)
#     index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor)
#     index.save_to_disk(os.path.join(fileidxfolder, "filedata.json"))


# def gpt_answer(question):
#     filepath = os.path.join(fileidxfolder, "filedata.json")
#     index = GPTSimpleVectorIndex.load_from_disk(filepath, llm_predictor=llm_predictor)

#     # prompt = f'You are a helpful support agent. You are asked: "{question}". Try to use only the information provided. Format your answer nicely as a Markdown page.'
#     prompt = f'您是一位专业顾问。您被问到："{question}"。请尽可能使用提供的信息。'
#     # response = index.query(prompt).response.strip()
#     response=index.query(prompt,llm_predictor=llm_predictor)
#     return response


def build_index():
    """
    Ingests data into LangChain by creating an FAISS index of OpenAI embeddings for text files in a folder "fileraw".
    The created index is saved to a file in the folder "fileidx".
    """

    # Get paths to text files in the "fileraw" folder
    ps = list(Path(filerawfolder).glob("**/*.txt"))

    data = []
    sources = []
    for p in ps:
        with open(p) as f:
            data.append(f.read())
        sources.append(p)

    # Split the documents into smaller chunks as needed due to the context limits of the LLMs
    docs = []
    for d in data:
        splits = split_text(d, chunk_chars=800, overlap=50)
        docs.extend(splits)

    # Create vector store from documents and save to disk
    store = FAISS.from_texts(docs, OpenAIEmbeddings())
    index_filename = f"{fileidxfolder}/docs.index"
    faiss.write_index(store.index, index_filename)
    store.index = None
    with open(f"{fileidxfolder}/faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)


def split_text(text, chunk_chars=4000, overlap=50):
    """
    Pre-process text file into chunks
    """
    splits = []
    for i in range(0, len(text), chunk_chars - overlap):
        splits.append(text[i : i + chunk_chars])
    return splits


def gpt_answer(question):
    # Load the LangChain.
    index = faiss.read_index(f"{fileidxfolder}/docs.index")

    with open(f"{fileidxfolder}/faiss_store.pkl", "rb") as f:
        store = pickle.load(f)

    store.index = index

    prefix_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that is very good at problem solving who thinks step by step.",
        }
    ]
    llm = OpenAIChat(temperature=0, prefix_messages=prefix_messages)

    chain = VectorDBQA.from_chain_type(llm, chain_type="stuff", vectorstore=store)

    result = chain.run(question)

    return result
