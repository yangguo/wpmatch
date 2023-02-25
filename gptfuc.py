import os

# import requests
from gpt_index import GPTSimpleVectorIndex, LLMPredictor, SimpleDirectoryReader
from langchain import OpenAI

# os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

uploadfolder = "uploads"
filerawfolder = "fileraw"
fileidxfolder = "fileidx"
backendurl = "http://localhost:8000"

# openai_api_key=os.environ.get("OPENAI_API_KEY")
# if openai_api_key is None:
#     print("请设置OPENAI_API_KEY")
llm_predictor = LLMPredictor(
    llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=1024)
)


def build_index():
    documents = SimpleDirectoryReader(filerawfolder, recursive=True).load_data()
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk(os.path.join(fileidxfolder, "filedata.json"))


def gpt_answer(question):
    filepath = os.path.join(fileidxfolder, "filedata.json")
    index = GPTSimpleVectorIndex.load_from_disk(filepath, llm_predictor=llm_predictor)

    prompt = f'You are a helpful support agent. You are asked: "{question}". Try to use only the information provided. Format your answer nicely as a Markdown page.'
    response = index.query(prompt).response.strip()
    return response


# def gpt_answer(question):
#     try:
#         url = backendurl + "/answer"
#         payload = {
#             "question": question,
#         }
#         headers = {}
#         res = requests.post(url, headers=headers, params=payload)
#         result = res.json()
#         print("成功")
#     except Exception as e:
#         print("错误: " + str(e))
#         result = "错误: " + str(e)
#     return result
