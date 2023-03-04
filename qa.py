"""Ask a question to the notion database."""
import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain,VectorDBQA
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

from langchain.llms import OpenAIChat
prefix_messages = [{"role": "system", "content": "You are a helpful assistant that is very good at problem solving who thinks step by step."}]
llm = OpenAIChat(model_name="gpt-3.5-turbo",temperature=0,prefix_messages=prefix_messages)
# llm = OpenAIChat(temperature=0)

# chain = VectorDBQAWithSourcesChain.from_llm(llm=openaichat, vectorstore=store)
chain = VectorDBQA.from_chain_type(llm, chain_type="stuff", vectorstore=store)
# result = chain({"question": args.question})
result=chain.run(args.question)
# print(f"Answer: {result['answer']}")
print(result)