# from langchain.llms import OpenAI
import json
import os

from gpt_index import GPTSimpleVectorIndex, LLMPredictor, SimpleDirectoryReader
# from langchain import OpenAI

# import requests
# from llama_index import GPTSimpleVectorIndex, LLMPredictor, SimpleDirectoryReader

from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

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
from gpt_index.langchain_helpers.chatgpt import ChatGPTLLMPredictor
llm_predictor = ChatGPTLLMPredictor()


def build_index():
    documents = SimpleDirectoryReader(filerawfolder, recursive=True).load_data()
    # index = GPTSimpleVectorIndex(documents)
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor)
    index.save_to_disk(os.path.join(fileidxfolder, "filedata.json"))


def gpt_answer(question):
    filepath = os.path.join(fileidxfolder, "filedata.json")
    index = GPTSimpleVectorIndex.load_from_disk(filepath, llm_predictor=llm_predictor)

    # prompt = f'You are a helpful support agent. You are asked: "{question}". Try to use only the information provided. Format your answer nicely as a Markdown page.'
    prompt = f'您是一位专业顾问。您被问到："{question}"。请尽可能使用提供的信息。'
    # response = index.query(prompt).response.strip()
    response=index.query(prompt,llm_predictor=llm_predictor)
    return response

