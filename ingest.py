"""This is the logic for ingesting Notion data into LangChain."""
import json
import os
import pickle
from pathlib import Path

import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# read config from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# get openai api key from config.json
api_key = config["openai_api_key"]

os.environ["OPENAI_API_KEY"] = api_key


def split_text(text, chunk_chars=4000, overlap=50):
    """
    Pre-process text file into chunks
    """
    splits = []
    for i in range(0, len(text), chunk_chars - overlap):
        splits.append(text[i : i + chunk_chars])
    return splits


# Here we load in the data in the format that Notion exports it in.
ps = list(Path("uploads/").glob("**/*.txt"))

data = []
sources = []
for p in ps:
    with open(p) as f:
        data.append(f.read())
    sources.append(p)

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = split_text(d, chunk_chars=800, overlap=50)
    docs.extend(splits)
    # metadatas.extend([{"source": sources[i]}] * len(splits))


# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings())  # , metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
