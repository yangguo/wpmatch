"""Ask a question to the notion database."""
import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
import argparse
import os
import json
# read config from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# get openai api key from config.json
api_key = config["openai_api_key"]

os.environ["OPENAI_API_KEY"] = api_key

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(model_name="gpt-3.5-turbo",temperature=0), vectorstore=store)
result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")